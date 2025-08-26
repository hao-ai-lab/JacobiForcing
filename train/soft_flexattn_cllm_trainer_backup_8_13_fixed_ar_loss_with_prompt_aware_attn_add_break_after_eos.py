import torch
import wandb
from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import torch.nn.functional as F

from torch.nn.attention.flex_attention import create_block_mask

from functools import lru_cache

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(self, *args,  accelerator=None, optimizer=None, lr_scheduler=None, train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader

        self.base_model = self.accelerator.unwrap_model(self.model)
        self.cfg = self.base_model.config

        self.train_step_cnt = 0

        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels
        # cache BlockMasks keyed by (L, prompt_len, T, heads, version)
        self._blockmask_cache = {}

    # Utilities
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def _unpack_sample(self, inputs):
        """
        Extract a single sample. (Assumes per_device_train_batch_size == 1.)
        Required keys:
          - input_ids: [1, L]
          - prompt_ids_len: scalar or [1]
          - T: length of traj_position_indices (last uncorrupted token positions) in [1, T]
        """
        # TODO: support bsz > 1 uppacking
        input_ids = inputs["input_ids"][0]
        prompt_len = inputs["prompt_ids_len"]
        if isinstance(prompt_len, torch.Tensor):
            if prompt_len.dim() > 0:
                prompt_len = prompt_len[0]
        prompt_len = self._to_int(prompt_len)

        traj_position_indices = inputs["traj_position_indices"][0][0]
        traj_position_indices = [int(u) for u in traj_position_indices]
        T = len(traj_position_indices)

        return (
            input_ids.to(self.args.device),
            prompt_len,
            T,
        )
    
    def _padding_mask_1d(self, inputs, input_ids: torch.Tensor) -> torch.Tensor:
        """
        [L] bool mask: True = real token, False = pad.
        """
        device = self.args.device
        pad_id = getattr(self.processing_class, "pad_token_id")
        print(f"[padding mask] pad_id={pad_id}")
        return (input_ids != pad_id).to(device)
    
    def _first_eos_index(self, tokens: torch.Tensor, eos_id: int | None) -> int | None:
        if eos_id is None:
            return None
        pos = torch.nonzero(tokens == eos_id, as_tuple=False)
        return int(pos[0]) if pos.numel() > 0 else None

    def _block_keep_mask_divergence_and_eos(
        self,
        input_ids: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        eos_id: int | None,
        drop_last_offset: bool = True,   # NEW default
    ) -> torch.Tensor:
        """
        Returns [N-1] bool if drop_last_offset else [N] bool.
        True => keep offset t for logits at position (start+t) predicting next token (t+1).
        We: start computing loss after first divergence.
        """
        device = input_ids.device
        size = N - 1 if drop_last_offset else N
        offs = torch.arange(size, device=device)

        k_block = input_ids[k_start : k_start + N]
        l_block = input_ids[l_start : l_start + N]

        # Divergence mask (pairwise): keep from first differing offset onward
        diff = (k_block[:size] != l_block[:size])
        if diff.any():
            first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
            div_keep = offs >= first_diff
        else:
            div_keep = torch.zeros(size, dtype=torch.bool, device=device)

        # UNUSED — EOS mask: keep offsets t such that (t+1) is BEFORE EOS in both blocks
        #def keep_next_before_eos(block):
        #    if eos_id is None:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    pos = torch.nonzero(block == eos_id, as_tuple=False)
        #    if pos.numel() == 0:
        #        return torch.ones(size, dtype=torch.bool, device=device)
        #    e = int(pos[0])          # EOS index within [0..N-1]
        #    return offs < e

        #eos_keep = keep_next_before_eos(k_block) & keep_next_before_eos(l_block)

        return div_keep

    @staticmethod
    def _index_layout(prompt_len: int, T: int, N):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    # FlexAttention BlockMask (single-pass version)
    # Enforces:
    #  - prompt queries: causal within prompt
    #  - k_j queries: causal within *their own* k_j block + prompt
    #  - last_j queries: causal within *their own* last_j block + prompt
    def _build_block_mask(self, L: int, prompt_len: int, T: int, heads: int):
        cache_key = (L, prompt_len, T, heads, "singlepass_prev-last-visible")
        if cache_key in self._blockmask_cache:
            return self._blockmask_cache[cache_key]

        N = self.max_new_tokens
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ks = torch.tensor(k_starts, device=self.args.device)  # [T]
        ls = torch.tensor(l_starts, device=self.args.device)  # [T]

        def mask_mod(b, h, q, k):
            # q, k: any shape, torch tensors
            rel_q = q - prompt_len
            rel_k = k - prompt_len
            block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
            block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

            is_prompt_q = q < prompt_len
            is_prompt_k = k < prompt_len

            is_kj_q    = (q >= prompt_len) & (block_idx_q % 2 == 0)
            is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
            is_kj_k    = (k >= prompt_len) & (block_idx_k % 2 == 0)
            is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)

            # j index for q, clamped to [0, T-1]
            j_q = torch.clamp(block_idx_q // 2, min=0, max=T - 1)

            ks_per_q = ks[j_q]
            ls_per_q = ls[j_q]

            # allow queries to attend to all *previous* last_* blocks.
            # This makes every last_j visible to all future k_{j+i} and last_{j+i}.
            k_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            # Prompt is always causal
            mask_prompt = is_prompt_q & (k <= q)

            # k_j queries: prompt (causal) + same k_j block (causal)
            #            + any *previous* last_* blocks (causal by construction)
            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            mask_kj = is_kj_q & (
                (is_prompt_k & (k <= q)) |
                k_in_prev_last |
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            # last_j queries: prompt (causal) + same last_j block (causal)
            #               + previous last_* blocks
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
            mask_lastj = is_lastj_q & (
                (is_prompt_k & (k <= q)) |
                k_in_prev_last |
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            return mask_prompt | mask_kj | mask_lastj

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device=self.args.device
        )
        self._blockmask_cache[cache_key] = block_mask
        return block_mask


    # Training Step
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._one_pass_losses_step(model, inputs)

    def _one_pass_losses_step(self, model, inputs):
        input_ids, prompt_len, T = self._unpack_sample(inputs)
        L = input_ids.size(0)

        eos_id = getattr(self.processing_class, "eos_token_id")
        pad_id = getattr(self.processing_class, "pad_token_id")
        N = self.max_new_tokens

        expected_len = prompt_len + 2 * T * N
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} (prompt_len={prompt_len}, T={T}, n_token_sequence_size={N})"
            )

        attn_mask = torch.ones(L, dtype=torch.long, device=input_ids.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        # ===== Debug printing =====
        # detok the AR input sequence
        prompt_ids_block = input_ids[:prompt_len]
        l_blocks_concat = torch.cat([input_ids[ls : ls + N] for ls in l_starts], dim=0)
        ar_concat_ids = torch.cat([prompt_ids_block, l_blocks_concat], dim=0)

        print("\n=== AR INPUTS (prompt + concatenated l_j blocks) ===")
        # Decode
        ar_text = self.processing_class.decode(ar_concat_ids, skip_special_tokens=False)

        print("\n[Decoded text]")
        print(ar_text)

        # Print all k_j blocks separately
        #print("\n=== k_j BLOCKS ===")
        #for j, ks in enumerate(k_starts):
        #    block_ids = input_ids[ks : ks + N]
        #    block_text = self.processing_class.decode(block_ids, skip_special_tokens=False)
        #    print(f"[k_{j}]")
        #    print(block_text)
        #    print()
        # ==========================
        
        # mark PAD as 0
        attn_mask[input_ids == pad_id] = 0

        # cut post-EOS inside last_N block (the last last_j)
        for j in range(T):
            starting_pos = l_starts[j]
            block = input_ids[starting_pos : starting_pos + N]
            pos = (block == eos_id).nonzero(as_tuple=False)
            if pos.numel():
                first_eos_pos = starting_pos + int(pos[0])
                attn_mask[first_eos_pos + 1 : starting_pos + N] = 0

        # Build structural block mask
        num_heads = getattr(self.cfg, 'num_attention_heads', 28)
        print(f"num heads from config: {self.cfg.num_attention_heads}")
        print(f"[block mask] num_heads={num_heads}, L={L}, prompt_len={prompt_len}, T={T}, N={N}")
        blk_mask = self._build_block_mask(L, prompt_len, T, num_heads)

        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attn_mask.unsqueeze(0),
            block_mask=blk_mask,
            attn_implementation="flex_attention",
        )
        logits = outputs.logits  # [1, L, V]

        # ========== AR loss ==========
        ar_labels = torch.full((1, L), IGNORE_TOKEN_ID, device=self.args.device)

        # include prompt tokens in AR labels (up to EOS/PAD)
        end = prompt_len
        if eos_id is not None:
            pos = (input_ids[:prompt_len] == eos_id).nonzero(as_tuple=False)
            if pos.numel():
                end = min(end, int(pos[0]) + 1)  # keep through EOS itself
        if pad_id is not None:
            pos = (input_ids[:prompt_len] == pad_id).nonzero(as_tuple=False)
            if pos.numel():
                end = min(end, int(pos[0]))
        if end > 0:
            ar_labels[0, :end] = input_ids[:end]

        # existing last_j labeling (unchanged)
        for j in range(T):
            ls = l_starts[j]
            block = input_ids[ls : ls + N]

            # stop at first PAD too, if present
            first_pad = None
            if pad_id is not None:
                pad_pos = torch.nonzero(block == pad_id, as_tuple=False)
                first_pad = int(pad_pos[0]) if pad_pos.numel() > 0 else None

            # inspect EOS
            eos_pos = None
            if eos_id is not None:
                epos = torch.nonzero(block == eos_id, as_tuple=False)
                eos_pos = int(epos[0]) if epos.numel() > 0 else None

            end = N
            if eos_pos is not None:
                end = min(end, eos_pos + 1)  # keep through EOS
            if first_pad is not None:
                end = min(end, first_pad)

            if end > 0:
                ar_labels[0, ls : ls + end] = input_ids[ls : ls + end]

            if eos_pos is not None or first_pad is not None:
                break

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index=IGNORE_TOKEN_ID)
        loss_ar = label_smoother(outputs, ar_labels, shift_labels=True) * 5

        # ========== Consistency loss ==========
        T_soft = getattr(self.args, "distill_temperature", 1.0)
        offs = torch.arange(N - 1, device=self.args.device)

        student_positions, teacher_positions = [], []
        for j in range(T):
            ks, ls = k_starts[j], l_starts[j]
            pair_keep = self._block_keep_mask_divergence_and_eos(
                input_ids, ks, ls, N, eos_id=eos_id, drop_last_offset=True
            )
            if pair_keep.any():
                sp = ks + offs[pair_keep]
                tp = ls + offs[pair_keep]
                student_positions.append(sp)
                teacher_positions.append(tp)

        if len(student_positions) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            sp = torch.cat(student_positions, dim=0)
            tp = torch.cat(teacher_positions, dim=0)

            pad1d = (input_ids != pad_id) if pad_id is not None else torch.ones_like(input_ids, dtype=torch.bool)
            keep = pad1d.index_select(0, sp) & pad1d.index_select(0, tp)
            keep = keep & pad1d.index_select(0, sp + 1) & pad1d.index_select(0, tp + 1)

            if keep.any():
                student_logits_sel = logits[0, sp[keep], :]
                teacher_logits_sel = logits[0, tp[keep], :].detach()
                log_ps = F.log_softmax(student_logits_sel / T_soft, dim=-1)
                p_t = F.softmax(teacher_logits_sel / T_soft, dim=-1)
                loss_consistency = (-(p_t * log_ps).sum(dim=-1)).mean() * (T_soft * T_soft)
            else:
                loss_consistency = torch.zeros((), device=self.args.device)

        total_loss = loss_ar + loss_consistency
        if self.args.qlora:
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            wandb.log({"ar loss": float(loss_ar.detach().cpu()),
                    "consistency loss": float(loss_consistency.detach().cpu())})

        del outputs, logits, ar_labels, label_smoother
        torch.cuda.empty_cache()

        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return total_loss.detach()
