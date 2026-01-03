import torch
import wandb
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn.attention.flex_attention import create_block_mask

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class CllmTrainer(Trainer):
    def __init__(self, *args, accelerator=None, optimizer=None, lr_scheduler=None, train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        targs = kwargs["args"]

        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader

        self.base_model = self.accelerator.unwrap_model(self.model)
        self.cfg = self.base_model.config

        self.train_step_cnt = 0
        self.use_gt_labels = getattr(targs, "use_gt_labels", False)

        # Cache per (B, H, Lmax, device) -> (BlockMask, state_tensors)
        # state_tensors are kept stable (same Tensor objects) and updated via .copy_(),
        # which is important when _compile=True.
        self._blk_mask_cache = {}

        # Pointers used by compiled mask_mod
        self._mask_prompt_lens = None  # [B] long
        self._mask_num_pairs = None    # [B] long
        self._mask_block_lens = None   # [B] long
        self._mask_valid = None        # [B, Lmax] bool

    # ---------- Make HF Trainer use your prepared DataLoader ----------
    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        return super().get_train_dataloader()

    # ---------- Unpack a batch ----------
    def _unpack_batch(self, inputs):
        """
        Expected keys from collator:
          - input_ids:      [B, Lmax]
          - seq_lens:       [B] (true unpadded length per sample; equals P + 2*T*N)
          - prompt_ids_len: [B]
          - num_pairs:      [B] (T)
          - block_len:      [B] (N)
        """
        dev = self.args.device
        input_ids = inputs["input_ids"].to(dev)
        seq_lens = inputs["seq_lens"].to(dev).long()
        prompt_lens = inputs["prompt_ids_len"].to(dev).long()
        num_pairs = inputs["num_pairs"].to(dev).long()
        block_lens = inputs["block_len"].to(dev).long()
        return input_ids, seq_lens, prompt_lens, num_pairs, block_lens

    # ---------- Layout ----------
    @staticmethod
    def _index_layout(prompt_len: int, T: int, N: int):
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    # ---------- Position ids (shared between noisy/clean blocks) ----------
    def _build_shared_position_ids_one(self, Lmax: int, seq_len: int, prompt_len: int, T: int, N: int):
        """
        [Lmax] position ids for ONE sample; indices >= seq_len are 0.
        """
        device = self.args.device
        pos = torch.zeros(Lmax, dtype=torch.long, device=device)

        if prompt_len > 0:
            pos[:prompt_len] = torch.arange(prompt_len, device=device)

        if T > 0 and N > 0:
            k_starts, l_starts = self._index_layout(prompt_len, T, N)
            rel = torch.arange(N, device=device)
            for j in range(T):
                base = prompt_len + j * N
                ks = k_starts[j]
                ls = l_starts[j]
                pos[ks:ks + N] = base + rel
                pos[ls:ls + N] = base + rel

        # do not touch suffix >= seq_len; remains 0
        return pos

    # ---------- Duplicate-prefix padding mask for consistency loss ----------
    def _duplicate_prefix_mask(self, input_ids_1d: torch.Tensor, prompt_len: int, T: int, N: int) -> torch.Tensor:
        """
        [L] bool: True where token should be masked because it's in k_j's prefix identical to last_j.
        Assumes no truncation; input_ids_1d is length L (= seq_len).
        """
        device = input_ids_1d.device
        L = int(input_ids_1d.size(0))
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        if T <= 0 or N <= 0:
            return mask

        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        for j in range(T):
            ks = k_starts[j]
            ls = l_starts[j]
            k_block = input_ids_1d[ks:ks + N]
            l_block = input_ids_1d[ls:ls + N]

            eq = (k_block == l_block)
            if torch.any(~eq):
                first_diff = int(torch.nonzero(~eq, as_tuple=False)[0])
            else:
                first_diff = N

            if first_diff > 0:
                end = min(ks + first_diff, L)
                if end > ks:
                    mask[ks:end] = True
        return mask

    def _build_padding_mask_for_loss(self, input_ids_1d: torch.Tensor, prompt_len: int, T: int, N: int) -> torch.Tensor:
        """
        [L] bool padding mask used for losses:
        True = mask out, False = keep.
        Combines: PAD tokens + duplicate-prefix in each k_j vs last_j.
        """
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        mask = torch.zeros_like(input_ids_1d, dtype=torch.bool, device=input_ids_1d.device)
        if pad_id is not None:
            mask |= (input_ids_1d == pad_id)
        mask |= self._duplicate_prefix_mask(input_ids_1d, prompt_len, T, N)
        return mask

    # ---------- Divergence keep mask (no truncation) ----------
    def _block_keep_mask_divergence(
        self,
        input_ids_1d: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        drop_last_offset: bool = False,
    ) -> torch.Tensor:
        """
        Returns [N] bool (or [N-1] if drop_last_offset): keep offsets from first divergence onward.
        """
        device = input_ids_1d.device
        if N <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        size = (N - 1) if drop_last_offset else N
        if size <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        offs = torch.arange(size, device=device)
        k_block = input_ids_1d[k_start:k_start + N]
        l_block = input_ids_1d[l_start:l_start + N]

        diff = (k_block[:size] != l_block[:size])
        if diff.any():
            first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
            return offs >= first_diff
        return torch.zeros(size, dtype=torch.bool, device=device)

    # ---------- Soft CE ----------
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        """
        predicts/targets: [K, V]; padding_mask: [K] (True=mask out)
        """
        if predicts.numel() == 0:
            return predicts.sum() * 0.0
        keep = ~padding_mask
        denom = keep.sum().clamp(min=1)
        logp = F.log_softmax(predicts, dim=-1)
        q = F.softmax(targets, dim=-1)
        ce = -(q * logp).sum(dim=-1)  # [K]
        ce = ce.masked_fill(padding_mask, 0)
        return ce.sum() / denom

    # =========================
    # FlexAttention BlockMask (batch-aware, per-sample prompt_len/T/N)
    # =========================
    def _mask_mod_varlen(self, b, h, q, k):
        b = b.long()

        p = self._mask_prompt_lens[b]
        T = self._mask_num_pairs[b]
        N = torch.maximum(self._mask_block_lens[b], torch.ones_like(self._mask_block_lens[b]))  # avoid div0
        valid = self._mask_valid

        in_range = valid[b, q] & valid[b, k]

        is_prompt_q = q < p
        is_prompt_k = k < p

        # prompt causal
        mask_prompt = is_prompt_q & (k <= q)

        rel_q = q - p
        rel_k = k - p

        block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
        block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

        is_noisy_q = (~is_prompt_q) & (block_idx_q % 2 == 0)
        is_clean_q = (~is_prompt_q) & (block_idx_q % 2 == 1)
        is_noisy_k = (~is_prompt_k) & (block_idx_k % 2 == 0)
        is_clean_k = (~is_prompt_k) & (block_idx_k % 2 == 1)

        Tmax = torch.maximum(T - 1, torch.zeros_like(T))
        j_q_unclamped = block_idx_q // 2
        j_q = torch.minimum(torch.maximum(j_q_unclamped, torch.zeros_like(j_q_unclamped)), Tmax)

        ks = p + 2 * j_q * N
        ls = p + (2 * j_q + 1) * N

        noisy_in_prev_noisy = is_noisy_k & (block_idx_k < 2 * j_q)
        clean_in_prev_clean = is_clean_k & (block_idx_k < 2 * j_q)

        same_noisy_block = is_noisy_q & is_noisy_k & (block_idx_q == block_idx_k)
        same_clean_block = is_clean_q & is_clean_k & (block_idx_q == block_idx_k)

        mask_noisy = is_noisy_q & (
            is_prompt_k |
            noisy_in_prev_noisy |
            (same_noisy_block & (k >= ks) & (k <= q))
        )

        mask_clean = is_clean_q & (
            is_prompt_k |
            clean_in_prev_clean |
            (same_clean_block & (k >= ls) & (k <= q))
        )

        return in_range & (mask_prompt | mask_noisy | mask_clean)

    def _build_block_mask_batched(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        num_pairs: torch.Tensor,
        block_lens: torch.Tensor,
        heads: int,
    ):
        """
        input_ids: [B, Lmax]
        seq_lens:  [B] true (unpadded) lengths; NO EOS truncation
        """
        B, Lmax = input_ids.shape
        device = input_ids.device

        pad_id = getattr(self.processing_class, "pad_token_id", None)
        arange = torch.arange(Lmax, device=device).unsqueeze(0)
        valid = arange < seq_lens.unsqueeze(1)
        if pad_id is not None:
            valid = valid & (input_ids != pad_id)

        key = (int(B), int(heads), int(Lmax), str(device))
        cached = self._blk_mask_cache.get(key, None)

        if cached is None:
            # allocate stable state tensors once for this shape/device
            state = {
                "prompt_lens": torch.empty((B,), device=device, dtype=torch.long),
                "num_pairs": torch.empty((B,), device=device, dtype=torch.long),
                "block_lens": torch.empty((B,), device=device, dtype=torch.long),
                "valid": torch.empty((B, Lmax), device=device, dtype=torch.bool),
            }
            self._mask_prompt_lens = state["prompt_lens"]
            self._mask_num_pairs = state["num_pairs"]
            self._mask_block_lens = state["block_lens"]
            self._mask_valid = state["valid"]

            blk_mask = create_block_mask(
                self._mask_mod_varlen,
                B=B, H=heads, Q_LEN=Lmax, KV_LEN=Lmax,
                device=device,
                _compile=True,
            )
            self._blk_mask_cache[key] = (blk_mask, state)
        else:
            blk_mask, state = cached
            # re-point pointers (same objects) just in case
            self._mask_prompt_lens = state["prompt_lens"]
            self._mask_num_pairs = state["num_pairs"]
            self._mask_block_lens = state["block_lens"]
            self._mask_valid = state["valid"]

        # update state IN-PLACE (important for compiled mask_mod)
        state["prompt_lens"].copy_(prompt_lens)
        state["num_pairs"].copy_(num_pairs)
        state["block_lens"].copy_(block_lens)
        state["valid"].copy_(valid)

        return blk_mask

    # =========================
    # Training step
    # =========================
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._one_pass_losses_step(model, inputs)

    def _one_pass_losses_step(self, model, inputs):
        input_ids, seq_lens, prompt_lens, num_pairs, block_lens = self._unpack_batch(inputs)
        B, Lmax = input_ids.shape

        eos_id = getattr(self.processing_class, "eos_token_id", None)
        pad_id = getattr(self.processing_class, "pad_token_id", None)

        # ---------- sanity ----------
        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            L = int(seq_lens[b])
            expected = P + 2 * T * N
            if L != expected:
                raise ValueError(f"[b={b}] Length mismatch: L={L}, expected {expected} (P={P}, T={T}, N={N})")

        # ---------- precompute: first EOS location in clean blocks per sample ----------
        # eos_stop_j[b] = first j where clean block contains EOS; -1 if none
        # eos_pos_in_clean[b] = offset within clean block at eos_stop_j; undefined if eos_stop_j=-1
        eos_stop_j = [-1] * B
        eos_pos_in_clean = [0] * B
        if eos_id is not None:
            for b in range(B):
                P = int(prompt_lens[b])
                T = int(num_pairs[b])
                N = int(block_lens[b])
                L = int(seq_lens[b])
                if T <= 0 or N <= 0:
                    continue
                for j in range(T):
                    ls = P + (2 * j + 1) * N
                    if ls >= L:
                        break
                    block = input_ids[b, ls:ls + N]
                    epos = torch.nonzero(block == eos_id, as_tuple=False)
                    if epos.numel() > 0:
                        eos_stop_j[b] = j
                        eos_pos_in_clean[b] = int(epos[0])
                        break

        # ---------- block mask + position ids ----------
        num_heads = getattr(self.cfg, "num_attention_heads", 28)
        blk_mask = self._build_block_mask_batched(
            input_ids=input_ids,
            seq_lens=seq_lens,
            prompt_lens=prompt_lens,
            num_pairs=num_pairs,
            block_lens=block_lens,
            heads=num_heads,
        )

        position_ids = torch.zeros((B, Lmax), dtype=torch.long, device=self.args.device)
        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            L = int(seq_lens[b])
            position_ids[b] = self._build_shared_position_ids_one(Lmax, L, P, T, N)

        # ---------- forward ----------
        outputs = model(
            input_ids=input_ids,
            attention_mask=blk_mask,
            position_ids=position_ids,
            attn_implementation="flex_attention",
        )
        logits = outputs.logits  # [B, Lmax, V]
        V = logits.size(-1)
        flat_logits = logits.reshape(B * Lmax, V)
        flat_ids = input_ids.reshape(B * Lmax)

        # ---------- helper to add next-token pairs ----------
        def add_pairs_for_segment(p_list, t_list, b, seg_start, seg_end):
            # p -> p+1 for p in [seg_start, seg_end-2]
            if seg_end - seg_start < 2:
                return
            p = torch.arange(seg_start, seg_end - 1, device=self.args.device, dtype=torch.long)
            t = p + 1
            p_list.append(p + b * Lmax)
            t_list.append(t + b * Lmax)

        # =========================
        # AR loss (stop after EOS clean block; inside EOS block only up to EOS)
        # =========================
        p_ar, t_ar = [], []

        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            L = int(seq_lens[b])

            add_pairs_for_segment(p_ar, t_ar, b, 0, min(P, L))
            if T <= 0 or N <= 0 or P <= 0:
                continue

            prev_last = P - 1
            if prev_last >= L:
                continue

            j_stop = eos_stop_j[b]
            epos = eos_pos_in_clean[b] if j_stop != -1 else None

            for j in range(T):
                if j_stop != -1 and j > j_stop:
                    break

                ls = P + (2 * j + 1) * N
                if ls >= L:
                    break

                # bridge
                p_ar.append(torch.tensor([prev_last + b * Lmax], device=self.args.device))
                t_ar.append(torch.tensor([ls + b * Lmax], device=self.args.device))

                seg_end = min(ls + N, L)
                if j_stop != -1 and j == j_stop:
                    seg_end = min(seg_end, ls + epos + 1)  # include EOS token

                add_pairs_for_segment(p_ar, t_ar, b, ls, seg_end)
                prev_last = seg_end - 1

        if len(p_ar) == 0:
            loss_ar = torch.zeros((), device=self.args.device)
        else:
            p_all = torch.cat(p_ar, dim=0)
            t_all = torch.cat(t_ar, dim=0)

            ar_logits = flat_logits[p_all]
            ar_targets = flat_ids[t_all]
            if pad_id is not None:
                ar_targets = ar_targets.masked_fill(ar_targets == pad_id, -100)

            loss_ar = F.cross_entropy(
                ar_logits.float(),
                ar_targets,
                reduction="mean",
                ignore_index=-100,
                label_smoothing=0.0,
            ) * 10.0

        # =========================
        # Consistency loss (stop after EOS clean block; in EOS block only offsets < eos_pos)
        # =========================
        T_soft = float(getattr(self.args, "distill_temperature", 1.0))
        student_idx, teacher_idx, pad_masks = [], [], []

        drop_last_offset = False

        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            L = int(seq_lens[b])

            if T <= 0 or N <= 0:
                continue

            # mask for loss (PAD + duplicate prefixes)
            pad_and_dup = self._build_padding_mask_for_loss(input_ids[b, :L], P, T, N)
            offs = torch.arange((N - 1) if drop_last_offset else N, device=self.args.device)

            j_stop = eos_stop_j[b]
            epos = eos_pos_in_clean[b] if j_stop != -1 else None

            for j in range(T):
                if j_stop != -1 and j > j_stop:
                    break

                ks = P + 2 * j * N
                ls = P + (2 * j + 1) * N
                if ls >= L:
                    break

                keep = self._block_keep_mask_divergence(
                    input_ids[b, :L], ks, ls, N, drop_last_offset=drop_last_offset
                )
                if keep.numel() == 0 or not keep.any():
                    continue

                # If EOS is in this clean block, only allow offsets that predict up to EOS:
                # offset t predicts token at t+1 => require t < epos
                if j_stop != -1 and j == j_stop:
                    keep = keep & (offs < epos)
                    if not keep.any():
                        break

                sp = ks + offs[keep]
                tp = ls + offs[keep]

                # in-range (should always hold if layout is correct)
                in_range = (sp < L) & (tp < L)
                sp = sp[in_range]
                tp = tp[in_range]
                if sp.numel() == 0:
                    if j_stop != -1 and j == j_stop:
                        break
                    continue

                student_idx.append(sp + b * Lmax)
                teacher_idx.append(tp + b * Lmax)
                pad_masks.append(pad_and_dup.index_select(0, sp))

                if j_stop != -1 and j == j_stop:
                    break

        if len(student_idx) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            s_all = torch.cat(student_idx, dim=0)
            t_all = torch.cat(teacher_idx, dim=0)
            pad_mask_all = torch.cat(pad_masks, dim=0)

            student_logits = flat_logits[s_all] / T_soft
            teacher_logits = flat_logits[t_all].detach() / T_soft

            loss_consistency = self.soft_cross_entropy(
                student_logits.float(),
                teacher_logits.float(),
                pad_mask_all
            )

            denom = max(int(num_pairs.sum().item()), 1)
            loss_consistency = loss_consistency * (T_soft * T_soft) / float(denom)

        total_loss = loss_ar + loss_consistency

        if getattr(self.args, "qlora", False):
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            wandb.log({
                "ar loss": float(loss_ar.detach().cpu()),
                "consistency loss": float(loss_consistency.detach().cpu()),
            })

        #del outputs, logits torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        return total_loss.detach()
