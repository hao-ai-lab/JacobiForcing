import torch
import wandb
from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from contextlib import contextmanager

import torch
from torch.cuda.amp import autocast

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def mem(tag):
    torch.cuda.synchronize()
    print(f"{tag} | alloc={torch.cuda.memory_allocated()/1e6:.1f}MB "
        f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB "
        f"max={torch.cuda.max_memory_allocated()/1e6:.1f}MB")

class CllmTrainer(Trainer):
    def __init__(self, *args, accelerator=None, optimizer=None, lr_scheduler=None, train_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)

        args = kwargs["args"]
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader

        self.train_step_cnt = 0

        self.ar_scale = float(getattr(args, "ar_scale", 10.0))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 64))                     # N (n-token sequence size)
        self.consistency_window = int(getattr(args, "consistency_window", 64))             # W (sliding window size)
        self.consistency_num_segments = int(getattr(args, "consistency_num_segments", 4))  # K (number of pairs per consistency loss step)
        self.qlora = bool(getattr(args, "qlora", False))

    # ====================== helper private methods ===================== #
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def _unpack_sample(self, inputs):
        """
        Required keys (batch size == 1):
          - input_ids: [1, L] (already on self.args.device)
          - prompt_ids_len: scalar or [1]
        """
        input_ids = inputs["input_ids"][0]
        prompt_len = self._to_int(inputs["prompt_ids_len"][0])
        print(f"[unpacking sample...] input_ids: {input_ids.size()}, prompt_len: {prompt_len}")
        return input_ids, prompt_len

    def _infer_T_from_input(self, input_len: int, prompt_len: int, N: int):
        rem = input_len - prompt_len
        return rem // (2 * N)

    def _compute_starts(self, input_ids, prompt_len: int, N: int):
        """
        Returns:
          - T
          - k_starts_input: starts of k_j in input_ids
          - l_starts_input: starts of last_j in input_ids
        """
        L_in = input_ids.size(0)
        T = self._infer_T_from_input(L_in, prompt_len, N)
        k_starts_input = [prompt_len + 2 * j * N for j in range(T)]
        l_starts_input = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return T, k_starts_input, l_starts_input

    def _infer_u_by_block(self, input_ids, prompt_len: int, N: int):
        """
        u_j = longest common prefix between k_j and last_j, using aligned .
        """
        T, k_starts, l_starts = self._compute_starts(input_ids, prompt_len, N)
        u = []
        for j in range(T):
            k_blk = input_ids[k_starts[j] : k_starts[j] + N]
            l_blk = input_ids[l_starts[j] : l_starts[j] + N]
            assert l_blk.numel() == N, f"labels slice length != N at block {j}: got {l_blk.numel()}, expected {N}."
            eq = (k_blk == l_blk)
            if torch.all(eq):
                u.append(N)
            else:
                u.append(int((~eq).nonzero(as_tuple=False)[0].item()))
        return u
    # ============================================================= #
     
    # Attention impl override
    def _unwrap_base_model(self, model):
        # Accelerate's unwrap (handles DeepSpeed/FSDP)
        return self.accelerator.unwrap_model(model)

    @contextmanager
    def _force_attention_impl(self, model, impl: str):
        base = self._unwrap_base_model(model)

        cfg  = getattr(base, "config")
        prev = getattr(cfg, "_attn_implementation")

        #print(f"[forcing attention impl] {prev} -> {impl}")
        try:
            cfg.attn_implementation = impl
            yield
        finally:
            if prev is not None:
                cfg.attn_implementation = prev

    def _forward_ar_with_flash(self, model, input_ids_ar, attn_1d):
        """
        AR forward with Flash-Attn2.
        `attn_1d` can be None when there is no padding.
        """
        with autocast(dtype=torch.bfloat16):
            with self._force_attention_impl(model, "flash_attention_2"):
                return model(
                    input_ids=input_ids_ar.unsqueeze(0),
                    attention_mask=attn_1d,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
    
    def _build_labels(self, input_ids, prompt_len, N):
        # builds [prompt] + [CONCAT_j last_j]  (length = prompt + T*N)
        device = input_ids.device
        T, _, l_starts = self._compute_starts(input_ids, prompt_len, N)
        idx = list(range(prompt_len))
        for j in range(T):
            ls = l_starts[j]
            idx.extend(range(ls, ls + N))
        idx = torch.tensor(idx, dtype=torch.long, device=device)
        return input_ids.index_select(0, idx)

    def _build_ar(self, input_ids, prompt_len, N):
        """
        Build compact AR sequence (on GPU):
            inputs:  [prompt] + [CONCAT_j last_j]
            labels:  same as inputs (causal LM), with prompt masked to IGNORE
        Returns: (input_ids_ar, labels_ar, attn_mask or None)
        """
        device = input_ids.device
        T, _, l_starts_inp = self._compute_starts(input_ids, prompt_len, N)

        # indicies to extract AR compact sequence: prompt + [CONCAT_j last_j]
        input_idx = list(range(prompt_len))
        for j in range(T):
            ls = l_starts_inp[j]
            input_idx.extend(range(ls, ls + N))
        input_idx = torch.tensor(input_idx, dtype=torch.long, device=device)

        input_ids_ar = input_ids.index_select(0, input_idx)
        labels_ar = input_ids_ar.clone()
        labels_ar[:prompt_len] = IGNORE_TOKEN_ID

        # no padding in compact AR sequence
        attn_1d = None

        print(f"[AR loss, local rank {self.args.local_rank}] input_ids_ar: {input_ids_ar.size()}, labels_ar: {labels_ar.size()}")
        return input_ids_ar, labels_ar, attn_1d

    # Consistency loss helper functions
    @staticmethod
    def _evenly_spaced_indices(total: int, k: int):
        """
        Choose k indices from [0..total-1].
        - If total <= 0: []
        - If total == 1: [0] (regardless of k)
        - If k >= total: all indices
        - If total >= 2 and k == 1: pick midpoint
        - Else: include endpoints and space the remainder.
        """
        if total <= 0:
            return []
        if total == 1:
            return [0]
        if k >= total:
            return list(range(total))
        if k == 1:
            return [total // 2]
        # k >= 2 and total >= 2
        picks = [0]
        for i in range(1, k - 1):
            pos = round(i * (total - 1) / (k - 1))
            picks.append(int(pos))
        picks.append(total - 1)
        seen, uniq = set(), []
        for x in picks:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _consistency_one_pass_local_windows_batch(
        self,
        model,
        input_ids,                # [L] = [prompt] + concat_j(k_j + last_j)
        prompt_len: int,
        N: int = 64,              # block size
        W: int = 16384,           # memory window width from previous last_i
        K: int = 4,               # number of (k_j, last_j) pairs
    ):
        """
        For K selected blocks j (j >= 1), build a row:
            input = [prompt_tail (P)] + [mem_tail (last min(W, j*N) tokens of concat(last_0..last_{j-1}))] + [k_j]
        Attention:
            - prompt tokens: causal among themselves
            - memory tokens: causal among themselves (no memory->prompt)
            - each query in k_j: can attend to prompt_tail + memory_tail (no access to k_j tokens)
        Supervise ALL N positions in k_j with targets last_j[t], t=0..N-1.
        Returns a scalar cross-entropy loss.
        """
        device = input_ids.device
        T, k_starts, l_starts = self._compute_starts(input_ids, prompt_len, N)

        # Labels stream: [prompt] + concat(last_0 .. last_{T-1})
        label_ids = self._build_labels(input_ids, prompt_len, N)  # [prompt_len + T*N]

        # Choose K blocks from {1, ..., T-1}
        candidates = list(range(1, T))
        if len(candidates) >= 2:
            idxs = self._evenly_spaced_indices(len(candidates), min(K, len(candidates)))
        else:
            idxs = list(range(len(candidates)))
        sel_blocks = [candidates[i] for i in idxs]
        B = len(sel_blocks)

        # pad id
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        assert pad_id is not None, "pad_token_id must be set in model config."

        # prompt window (P)
        P = prompt_len

        # Per-row metadata
        prompt_lens = []     # always P
        mem_lens = []        # <= W
        total_lens = []      # P + mem_len + N
        rows_lbl = []        # (mem_start_lbl, mem_end_lbl, kj_start_inp, kj_end_inp)
        label_spans = []     # (start_in_label_ids, end_in_label_ids) for last_j

        for j in sel_blocks:
            # memory window on labels timeline
            full_prev_len = j * N
            mem_len = min(W, full_prev_len)
            mem_end_lbl = prompt_len + j * N
            mem_start_lbl = mem_end_lbl - mem_len

            # k_j in input_ids (interleaved there)
            kj_start = k_starts[j]
            kj_end   = kj_start + N
            assert kj_end <= input_ids.size(0), "k_j slice out of range"

            rows_lbl.append((mem_start_lbl, mem_end_lbl, kj_start, kj_end))
            prompt_lens.append(P)
            mem_lens.append(mem_len)
            total_lens.append(P + mem_len + N)

            # targets
            lbl_start = prompt_len + j * N
            lbl_end   = lbl_start + N
            label_spans.append((lbl_start, lbl_end))

        max_len = int(max(total_lens)) if total_lens else 0

        # Allocate padded batch
        batch_ids    = torch.full((B, max_len), pad_id, dtype=input_ids.dtype, device=device)
        position_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
        attn_keep    = torch.zeros((B, max_len, max_len), dtype=torch.bool, device=device)

        # Gather helpers
        gather_q_positions = torch.zeros((B, N), dtype=torch.long, device=device)
        all_targets        = torch.empty((B, N), dtype=torch.long, device=device)

        for i, (j, (mem_start_lbl, mem_end_lbl, kj_start, kj_end)) in enumerate(zip(sel_blocks, rows_lbl)):
            P_i     = prompt_lens[i]
            mem_len = mem_lens[i]
            kj_len  = kj_end - kj_start  # == N
            L_i     = P_i + mem_len + kj_len

            # ---- Place prompt tail (labels[0:P]) ----
            if P_i > 0:
                prompt_slice = label_ids[0:P_i]   # contiguous prompt
                batch_ids[i, 0:P_i] = prompt_slice
                position_ids[i, 0:P_i] = torch.arange(0, P_i, device=device)

            # ---- Place memory tail from LABELS STREAM ----
            if mem_len > 0:
                mem_slice = label_ids[mem_start_lbl:mem_end_lbl]
                batch_ids[i, P_i:P_i + mem_len] = mem_slice
                position_ids[i, P_i:P_i + mem_len] = torch.arange(0, mem_len, device=device)

            # ---- Place k_j tokens from INPUT stream ----
            batch_ids[i, P_i + mem_len:P_i + mem_len + kj_len] = input_ids[kj_start:kj_end]
            position_ids[i, P_i + mem_len:P_i + mem_len + kj_len] = torch.arange(0, kj_len, device=device)

            # ---- Build 2‑D keep‑mask (L_i x L_i) ----
            # Prompt causal within prompt
            if P_i > 0:
                tri_p = torch.tril(torch.ones((P_i, P_i), dtype=torch.bool, device=device))
                attn_keep[i, 0:P_i, 0:P_i] = tri_p

            # Memory causal within memory (no memory->prompt)
            if mem_len > 0:
                tri_m = torch.tril(torch.ones((mem_len, mem_len), dtype=torch.bool, device=device))
                attn_keep[i, P_i:P_i + mem_len, P_i:P_i + mem_len] = tri_m

            # k_j attends to ALL prompt + ALL memory, not to itself
            k_row_start = P_i + mem_len
            k_row_end   = L_i
            if P_i > 0:
                attn_keep[i, k_row_start:k_row_end, 0:P_i] = True
            if mem_len > 0:
                attn_keep[i, k_row_start:k_row_end, P_i:P_i + mem_len] = True

            # ---- Gather positions across k_j and collect targets last_j ----
            gather_q_positions[i] = torch.arange(k_row_start, k_row_end, device=device)  # N positions
            lbl_start, lbl_end = label_spans[i]
            all_targets[i] = label_ids[lbl_start:lbl_end]

        # Convert keep-mask to additive bias: keep=0, mask=-inf
        attn_bias = torch.zeros((B, max_len, max_len), dtype=torch.float32, device=device)
        attn_bias[~attn_keep] = float("-inf")
        attn_bias = attn_bias.unsqueeze(1)  # [B, 1, L, L]

        with autocast(dtype=torch.bfloat16):
            out = model(
                input_ids=batch_ids,
                attention_mask=attn_bias,   # 4D additive mask
                position_ids=position_ids,  # per-segment reset
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )

        logits = out.logits  # [B, max_len, V]

        # Gather logits over k_j span
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        q_pos = gather_q_positions
        student = logits[b_idx, q_pos, :]          # [B, N, V]
        student = student.reshape(B * N, -1)       # [B*N, V]
        targets = all_targets.reshape(B * N)       # [B*N]

        loss_c = torch.nn.functional.cross_entropy(
            student, targets, reduction="mean", ignore_index=IGNORE_TOKEN_ID
        )
        return loss_c

    # KEY TRAINING STEP
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._step(model, inputs)

    def _step(self, model, inputs):
        input_ids, prompt_len = self._unpack_sample(inputs)
        
        # Infer T: total number of n-token subsequences
        L_in = input_ids.size(0)
        T = self._infer_T_from_input(L_in, prompt_len, self.max_new_tokens)

        # Longest common prefix length u_j by comparing k_j and last_j
        # len(u_by_block) == T
        u_by_block = self._infer_u_by_block(input_ids, prompt_len, self.max_new_tokens)

        with self.accelerator.accumulate(model):
            # ============= AR loss =============
            input_ids_ar, lbls_ar, attn_1d = self._build_ar(
                input_ids,
                prompt_len,
                self.max_new_tokens,
            )
            out_ar = self._forward_ar_with_flash(model, input_ids_ar, attn_1d)

            label_smoother = LabelSmoother(epsilon=0.1, ignore_index=IGNORE_TOKEN_ID)
            loss_ar = label_smoother(out_ar, lbls_ar.unsqueeze(0), shift_labels=True)
            loss_ar = loss_ar * self.ar_scale

            self.accelerator.backward(loss_ar)
            loss_ar_scalar = float(loss_ar.detach())

            # drop torch references to free memory
            del out_ar
            del loss_ar
            del lbls_ar
            del input_ids_ar
            
            # ============= Consistency loss =============
            # ONE forward pass over local windows 
            loss_cons = self._consistency_one_pass_local_windows_batch(
                model,
                input_ids,
                prompt_len,
                N=self.max_new_tokens,
                W=self.consistency_window,
                K=self.consistency_num_segments,
            )

            self.accelerator.backward(loss_cons)
            loss_cons_scalar = float(loss_cons.detach())

            del loss_cons

        if getattr(self.args, "local_rank", -1) in (-1, 0):
            wandb.log(
                {
                    "ar loss": loss_ar_scalar,
                    "consistency loss": loss_cons_scalar,
                }
            )

        total = loss_ar_scalar + loss_cons_scalar
        return torch.tensor(total, device=self.args.device)
