import torch
import wandb
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn.functional as F
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
        self.default_block_len = int(getattr(targs, "max_new_tokens", 64))
        self.use_gt_labels = getattr(targs, "use_gt_labels", False)

        # state tensors used by compiled flex mask_mod
        self._mask_prompt_lens = None   # [B]
        self._mask_num_pairs = None     # [B]
        self._mask_block_lens = None    # [B]
        self._mask_valid = None         # [B, Lmax] bool

    # ---------- Make HF Trainer use your prepared DataLoader ----------
    def get_train_dataloader(self):
        if self.train_dataloader is not None:
            return self.train_dataloader
        return super().get_train_dataloader()

    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    # ---------- Unpack a batch ----------
    def _unpack_batch(self, inputs):
        """
        Expected keys from collator:
          - input_ids: [B, Lmax]
          - seq_lens: [B] (true unpadded length per sample)
          - prompt_ids_len: [B]
          - num_pairs: [B]
          - block_len: [B]
        """
        input_ids = inputs["input_ids"].to(self.args.device)  # [B, Lmax]
        seq_lens = inputs["seq_lens"].to(self.args.device).long()  # [B]
        prompt_lens = inputs["prompt_ids_len"].to(self.args.device).long()  # [B]
        num_pairs = inputs["num_pairs"].to(self.args.device).long()  # [B]
        block_lens = inputs["block_len"].to(self.args.device).long()  # [B]
        return input_ids, seq_lens, prompt_lens, num_pairs, block_lens

    # ---------- Helpers updated to accept per-sample N ----------
    @staticmethod
    def _index_layout(prompt_len: int, T: int, N: int):
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    def _build_shared_position_ids_one(self, Lmax: int, seq_len: int, prompt_len: int, T: int, N: int):
        """
        [Lmax] position ids for ONE sample; indices >= seq_len can be 0.
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

        # suffix is already 0; masked out by blockmask
        return pos

    def _flip_block_after_eos_to_pad(self, input_ids_1d: torch.Tensor, start: int, N: int, eos_id: int | None, pad_id: int | None) -> int:
        if eos_id is None or pad_id is None or N <= 0:
            return 0
        block = input_ids_1d[start:start + N]
        pos = (block == eos_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return 0
        k = int(pos[0])
        flip_start = start + k + 1
        flip_end = start + N
        if flip_start < flip_end:
            input_ids_1d[flip_start:flip_end] = pad_id
            return flip_end - flip_start
        return 0
    
    def _first_eos_in_clean(self, input_ids_1d: torch.Tensor, P: int, T: int, N: int, eos_id: int | None):
        """
        Returns (stop_j, eos_off) where eos_off is offset within clean block.
        If no EOS, returns (None, None).
        """
        if eos_id is None or T <= 0 or N <= 0:
            return None, None
        for j in range(T):
            ls = P + (2 * j + 1) * N
            block = input_ids_1d[ls:ls + N]
            epos = torch.nonzero(block == eos_id, as_tuple=False)
            if epos.numel() > 0:
                return j, int(epos[0])  # eos offset in [0..N-1]
        return None, None


    def _duplicate_prefix_mask(self, input_ids_1d: torch.Tensor, prompt_len: int, T: int, N: int) -> torch.Tensor:
        device = input_ids_1d.device
        L = input_ids_1d.size(0)
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
                mask[ks:ks + first_diff] = True
        return mask

    def _build_padding_mask_for_loss(self, input_ids_1d: torch.Tensor, prompt_len: int, T: int, N: int) -> torch.Tensor:
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        mask = torch.zeros_like(input_ids_1d, dtype=torch.bool, device=input_ids_1d.device)
        if pad_id is not None:
            mask |= (input_ids_1d == pad_id)
        mask |= self._duplicate_prefix_mask(input_ids_1d, prompt_len, T, N)
        return mask

    def _block_keep_mask_divergence_and_eos(
        self,
        input_ids_1d: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        eos_id: int | None,
        drop_last_offset: bool = False,
    ) -> torch.Tensor:
        device = input_ids_1d.device
        if N <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        size = N - 1 if drop_last_offset else N
        if size <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        offs = torch.arange(size, device=device)
        k_block = input_ids_1d[k_start:k_start + N]
        l_block = input_ids_1d[l_start:l_start + N]

        diff = (k_block[:size] != l_block[:size])
        if diff.any():
            first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
            div_keep = offs >= first_diff
        else:
            div_keep = torch.zeros(size, dtype=torch.bool, device=device)

        return div_keep

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
        # b,h,q,k are tensors inside flex mask land
        b = b.long()

        p = self._mask_prompt_lens[b]     # [..]
        T = self._mask_num_pairs[b]
        # Use torch.maximum instead of clamp for vmap compatibility
        N = torch.maximum(self._mask_block_lens[b], torch.ones_like(self._mask_block_lens[b]))  # avoid div0 in dead rows
        valid = self._mask_valid          # [B, Lmax] bool

        in_range = valid[b, q] & valid[b, k]

        is_prompt_q = q < p
        is_prompt_k = k < p

        # prompt causal (k<=q already implies k is prompt when q is prompt)
        mask_prompt = is_prompt_q & (k <= q)

        rel_q = q - p
        rel_k = k - p

        block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
        block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

        is_noisy_q = (~is_prompt_q) & (block_idx_q % 2 == 0)   # noisy == k_j
        is_clean_q = (~is_prompt_q) & (block_idx_q % 2 == 1)   # clean == last_j
        is_noisy_k = (~is_prompt_k) & (block_idx_k % 2 == 0)
        is_clean_k = (~is_prompt_k) & (block_idx_k % 2 == 1)

        # Replace torch.clamp with element-wise min/max operations
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

    def _build_block_mask_batched(self, input_ids: torch.Tensor, seq_lens: torch.Tensor, prompt_lens: torch.Tensor, num_pairs: torch.Tensor, block_lens: torch.Tensor, heads: int):
        """
        input_ids: [B, Lmax]
        returns BlockMask compatible with flex_attention
        """
        B, Lmax = input_ids.shape

        # valid tokens: idx < seq_len AND not PAD
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        arange = torch.arange(Lmax, device=self.args.device).unsqueeze(0)  # [1, Lmax]
        valid = arange < seq_lens.unsqueeze(1)                             # [B, Lmax]
        if pad_id is not None:
            valid = valid & (input_ids != pad_id)

        self._mask_prompt_lens = prompt_lens
        self._mask_num_pairs = num_pairs
        self._mask_block_lens = block_lens
        self._mask_valid = valid

        blk_mask = create_block_mask(
            self._mask_mod_varlen,
            B=B, H=heads, Q_LEN=Lmax, KV_LEN=Lmax,
            device=self.args.device,
            _compile=False,
        )
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

        # sanity: each sample must match layout: L = P + 2*T*N
        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            L = int(seq_lens[b])
            expected = P + 2 * T * N
            if L != expected:
                raise ValueError(f"[b={b}] Length mismatch: L={L}, expected {expected} (P={P}, T={T}, N={N})")

        # flip post-EOS tokens to PAD on last clean block per sample
        for b in range(B):
            P = int(prompt_lens[b])
            T = int(num_pairs[b])
            N = int(block_lens[b])
            if T > 0 and N > 0:
                _, l_starts = self._index_layout(P, T, N)
                self._flip_block_after_eos_to_pad(input_ids[b], l_starts[-1], N, eos_id, pad_id)

        # Build structural block mask + batched position ids
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

        # =========================
        # AR loss (per-sample pairs, flattened gather)
        # =========================
        p_glob, t_glob = [], []

        def add_pairs_for_segment(b, seg_start, seg_end):
            # p -> p+1 for p in [seg_start, seg_end-2]
            if seg_end - seg_start < 2:
                return
            p = torch.arange(seg_start, seg_end - 1, device=self.args.device, dtype=torch.long)
            t = p + 1
            p_glob.append(p + b * Lmax)
            t_glob.append(t + b * Lmax)

        for b in range(B):
            P = int(prompt_lens[b]); T = int(num_pairs[b]); N = int(block_lens[b]); L = int(seq_lens[b])
            add_pairs_for_segment(b, 0, P)

            if T <= 0 or N <= 0:
                continue

            stop_j, eos_off = self._first_eos_in_clean(input_ids[b, :L], P, T, N, eos_id)

            prev_last_logit_pos = P - 1
            for j in range(T):
                if stop_j is not None and j > stop_j:
                    break

                ls = P + (2 * j + 1) * N
                if ls >= L:
                    break

                # bridge into this clean block (only if we haven't terminated)
                if prev_last_logit_pos >= 0 and ls < L:
                    p_glob.append(torch.tensor([prev_last_logit_pos + b * Lmax], device=self.args.device))
                    t_glob.append(torch.tensor([ls + b * Lmax], device=self.args.device))

                # in-block end
                end = N
                if stop_j is not None and j == stop_j:
                    end = min(end, eos_off + 1)  # include EOS token in segment

                seg_end = min(ls + end, L)
                add_pairs_for_segment(b, ls, seg_end)

                prev_last_logit_pos = seg_end - 1

                if stop_j is not None and j == stop_j:
                    break


        if len(p_glob) == 0:
            loss_ar = torch.zeros((), device=self.args.device)
        else:
            p_all = torch.cat(p_glob, dim=0)
            t_all = torch.cat(t_glob, dim=0)

            ar_logits = flat_logits[p_all]          # [K, V]
            ar_targets = flat_ids[t_all]            # [K]
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
        # Consistency loss (noisy vs clean at kept offsets), flattened gather
        # =========================
        T_soft = float(getattr(self.args, "distill_temperature", 1.0))
        student_glob, teacher_glob, padding_masks = [], [], []
        total_pairs = int(num_pairs.sum().item())

        drop_last_offset = False

        for b in range(B):
            P = int(prompt_lens[b]); T = int(num_pairs[b]); N = int(block_lens[b]); L = int(seq_lens[b])
            if T <= 0 or N <= 0:
                continue

            stop_j, eos_off = self._first_eos_in_clean(input_ids[b, :L], P, T, N, eos_id)

            pad_and_dup = self._build_padding_mask_for_loss(input_ids[b, :L], P, T, N)
            offs = torch.arange(N, device=self.args.device)

            for j in range(T):
                if stop_j is not None and j > stop_j:
                    break

                ks = P + 2 * j * N
                ls = P + (2 * j + 1) * N
                if ls >= L:
                    break

                keep = self._block_keep_mask_divergence_and_eos(input_ids[b, :L], ks, ls, N, eos_id=eos_id, drop_last_offset=False)

                # If this is the EOS-containing clean block, drop offsets >= eos_off
                # (offset eos_off would predict after EOS; we don't want that)
                if stop_j is not None and j == stop_j:
                    keep = keep & (offs < eos_off)
                    if not keep.any():
                        break  # EOS hit very early; stop entirely after this block

                if not keep.any():
                    continue

                sp = ks + offs[keep]
                tp = ls + offs[keep]
                in_range = (sp < L) & (tp < L)
                sp = sp[in_range]; tp = tp[in_range]
                if sp.numel() == 0:
                    continue

                student_glob.append(sp + b * Lmax)
                teacher_glob.append(tp + b * Lmax)
                padding_masks.append(pad_and_dup.index_select(0, sp))

                if stop_j is not None and j == stop_j:
                    break


        if len(student_glob) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            s_all = torch.cat(student_glob, dim=0)
            t_all = torch.cat(teacher_glob, dim=0)
            pad_mask_all = torch.cat(padding_masks, dim=0)  # [K] bool

            student_logits_all = flat_logits[s_all] / T_soft
            teacher_logits_all = flat_logits[t_all].detach() / T_soft

            loss_consistency = self.soft_cross_entropy(
                student_logits_all.float(),
                teacher_logits_all.float(),
                pad_mask_all
            )
            # normalize similar to old code, but across batch total pairs
            denom = max(total_pairs, 1)
            loss_consistency = loss_consistency * (T_soft * T_soft) / float(denom)

        total_loss = loss_ar + loss_consistency

        if getattr(self.args, "qlora", False):
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            wandb.log({
                "ar loss": float(loss_ar.detach().cpu()),
                "consistency loss": float(loss_consistency.detach().cpu()),
            })

        torch.cuda.empty_cache()

        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return total_loss.detach()
