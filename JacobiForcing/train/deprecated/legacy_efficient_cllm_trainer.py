import torch
import wandb
from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from contextlib import contextmanager

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class CllmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0

        self.ar_scale = float(getattr(args, "ar_scale", 10.0))
        self.keep_abs_pos_ar = bool(getattr(args, "keep_abs_pos_ar", True))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 64))             # N (block size)
        self.consistency_window = int(getattr(args, "consistency_window", 64))     # W
        self.consistency_num_segments = int(getattr(args, "consistency_num_segments", 4))  # K (GLOBAL)
        self.qlora = bool(getattr(args, "qlora", False))

    # ---------- Utilities ----------
    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def _unpack_sample(self, inputs):
        """
        Required keys (batch size == 1):
          - input_ids: [1, L] (already on self.args.device)
          - prompt_ids_len: scalar or [1]
        """
        input_ids = inputs["input_ids"][0].to(self.args.device)
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
        u_j = longest common prefix between k_j and last_j, using aligned starts.
        Done on GPU (N is small).
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
    
    # ---------- Attention impl override ----------
    @contextmanager
    def _force_attention_impl(self, model, impl: str):
        cfg = getattr(model, "config", None)
        prev = getattr(cfg, "attn_implementation", None) if cfg is not None else None
        try:
            if cfg is not None:
                cfg.attn_implementation = impl
            yield
        finally:
            if cfg is not None:
                cfg.attn_implementation = prev

    def _forward_ar_with_flash(self, model, ids_ar, attn_1d, pos_ar):
        """
        AR forward with Flash-Attn2.
        `attn_1d` can be None when there is no padding.
        """
        with autocast(dtype=torch.bfloat16):
            with self._force_attention_impl(model, "flash_attention_2"):
                return model(
                    input_ids=ids_ar.unsqueeze(0),
                    attention_mask=attn_1d,
                    position_ids=pos_ar.unsqueeze(0) if pos_ar is not None else None,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )

    # ---------- Label builders (GPU) ----------
    def _build_labels_last_compressed_gpu(self, input_ids, prompt_len, N):
        """
        GPU: Build compressed teacher labels for consistency:
            [prompt] + concat_j last_j   (length = P + T*N)
        """
        device = input_ids.device
        T, _, l_starts = self._compute_starts(input_ids, prompt_len, N)
        idx = list(range(prompt_len))
        for j in range(T):
            ls = l_starts[j]
            idx.extend(range(ls, ls + N))
        idx = torch.tensor(idx, dtype=torch.long, device=device)
        return input_ids.index_select(0, idx)

    # ---------- AR loss: compact causal sequence (NO tails) ----------
    def _build_ar(self, input_ids, prompt_len, u_by_block, N, keep_abs_pos=True):
        """
        Build compact AR sequence (on GPU):
            inputs:  [prompt] + concat_j last_j[:u_j]
            labels:  same as inputs (causal LM), with prompt masked to IGNORE
        Returns: (input_ids_ar, labels_ar, position_ids or None, attn_mask or None)
        """
        device = input_ids.device
        T, _, l_starts_inp = self._compute_starts(input_ids, prompt_len, N)

        # Indices to keep for AR compact sequence: prompt + last_j[:u_j]
        input_idx = list(range(prompt_len))
        for j in range(T):
            u = max(0, min(N, int(u_by_block[j])))
            if u == 0:
                continue
            ls = l_starts_inp[j]
            input_idx.extend(range(ls, ls + u))
        input_idx = torch.tensor(input_idx, dtype=torch.long, device=device)

        input_ids_ar = input_ids.index_select(0, input_idx)  # GPU
        labels_ar = input_ids_ar.clone()
        labels_ar[:prompt_len] = IGNORE_TOKEN_ID

        # Position ids: absolute or compact
        if keep_abs_pos:
            full_pos = torch.arange(input_ids.size(0), device=device, dtype=torch.long)
            position_ids = full_pos.index_select(0, input_idx)
        else:
            position_ids = torch.arange(input_ids_ar.size(0), device=device, dtype=torch.long)

        # No padding in compact AR sequence → omit mask (saves memory)
        attn_1d = None  # or torch.ones(1, input_ids_ar.size(0), dtype=torch.bool, device=device)

        print(f"[AR loss] input_ids_ar: {input_ids_ar.size()}, labels_ar: {labels_ar.size()}")
        return input_ids_ar, labels_ar, position_ids, attn_1d

    # ---------- Consistency: ONE forward over GLOBAL-K local windows ----------
    @staticmethod
    def _evenly_spaced_indices(total: int, k: int):
        """
        Choose exactly k indices in [0..total-1] (or all if total < k),
        evenly spaced with endpoints included.
        """
        if total <= 0:
            return []
        if k >= total:
            return list(range(total))
        if k <= 1:
            return [0]
        picks = []
        for i in range(k):
            pos = round(i * (total - 1) / (k - 1))
            picks.append(int(pos))
        # deduplicate (rare) while preserving order
        seen, uniq = set(), []
        for x in picks:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _consistency_one_pass_local_windows_batch(
        self,
        model,
        input_ids,   # GPU [L]
        label_ids,   # GPU [prompt_len + T*N]  (compressed: [prompt] + all last_j)
        prompt_len,
        u_by_block,
        N=64,   # block size
        W=64,   # local window
        K=4,    # GLOBAL number of supervised tail positions
    ):
        """
        Build a batch of local windows (<=W) around exactly K globally-selected tail positions.
        One forward over [B, max_len], right-padded; read last-token logits per row.

        label_ids should be compressed ([prompt] + concat last_j); aligned also works.
        """
        device = input_ids.device
        T, k_starts, l_starts = self._compute_starts(input_ids, prompt_len, N)

        # Detect labels layout (aligned vs compressed)
        L_lab = label_ids.size(0)
        aligned_len    = prompt_len + 2 * T * N
        compressed_len = prompt_len + T * N
        if L_lab == aligned_len:
            def lab_start(j):  # last_j start in labels
                return l_starts[j]
        elif L_lab == compressed_len:
            def lab_start(j):  # compressed layout
                return prompt_len + j * N
        else:
            def lab_start(j):
                return prompt_len + j * N

        # Compute per-block tail lengths (N - u_j), total candidates M
        tail_lens = [max(0, N - int(u_j)) for u_j in u_by_block[:T]]
        M = sum(tail_lens)
        if M == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        # Precompute cumulative sums to map global index -> (block j, local t)
        # cum[i] = number of candidates up to and including block i
        cum = []
        s = 0
        for tl in tail_lens:
            s += tl
            cum.append(s)

        # Pick exactly K global positions (evenly spaced) without building the full list
        picks = self._evenly_spaced_indices(M, K)

        # Map picks to (j, t, p, li)
        sel_global_pos = []   # global positions p in input_ids (k_j[t])
        sel_label_pos  = []   # indices into label_ids
        for g in picks:
            # find block j by lower_bound on cum
            # (can be done linearly since K is small; cost negligible)
            j = 0
            while j < T and cum[j] <= g:
                j += 1
            # previous cumulative before this block
            prev = 0 if j == 0 else cum[j - 1]
            local_offset = g - prev
            u_j = int(u_by_block[j])
            t = u_j + local_offset  # tail offset within k_j
            p = k_starts[j] + t
            li = lab_start(j) + t
            sel_global_pos.append(p)
            sel_label_pos.append(li)

        # Build windows [max(0,p-W+1) .. p] and right-pad into a batch (GPU)
        B = len(sel_global_pos)
        starts = []
        ends = []
        lengths = []
        for p in sel_global_pos:
            start = max(0, p - (W - 1))
            end = p + 1
            starts.append(start)
            ends.append(end)
            lengths.append(end - start)
        max_len = max(lengths)

        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = int(input_ids[0].item())

        batch_ids = torch.full((B, max_len), pad_id, dtype=input_ids.dtype, device=device)
        batch_pos = torch.zeros((B, max_len), dtype=torch.long, device=device)
        attn_1d   = torch.zeros((B, max_len), dtype=torch.bool, device=device)

        for i, (start, end, L_i) in enumerate(zip(starts, ends, lengths)):
            start_col = max_len - L_i
            batch_ids[i, start_col:max_len] = input_ids[start:end]
            batch_pos[i, start_col:max_len] = torch.arange(start, end, device=device, dtype=torch.long)
            attn_1d[i, start_col:max_len]   = True

        # Gather targets in one shot on GPU
        label_idx = torch.tensor(sel_label_pos, dtype=torch.long, device=device)
        targets = label_ids.index_select(0, label_idx)  # [B]

        # Single forward; take last-token logits per row
        with autocast(dtype=torch.bfloat16):
            out = model(
                input_ids=batch_ids,
                attention_mask=attn_1d,     # bool mask
                position_ids=batch_pos,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
        logits = out.logits  # [B, max_len, V]
        student = logits[:, -1, :]                                              # [B, V]
        loss_c = torch.nn.functional.cross_entropy(
            student, targets, reduction="mean", ignore_index=IGNORE_TOKEN_ID
        )
        return loss_c

    # ---------- Training ----------
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._step(model, inputs)

    def _step(self, model, inputs):
        input_ids, prompt_len = self._unpack_sample(inputs)

        # Infer T (enforces the k/last alternation in input_ids)
        L_in = input_ids.size(0)
        T = self._infer_T_from_input(L_in, prompt_len, self.max_new_tokens)

        # Longest common prefix length u_j by comparing k_j and last_j
        u_by_block = self._infer_u_by_block(input_ids, prompt_len, self.max_new_tokens)
        # len(u_by_block) == T

        # ----- AR loss: compact causal (prompt + last_j[:u_j]) -----
        ids_ar, lbls_ar, pos_ar, attn_1d = self._build_ar(
            input_ids,
            prompt_len,
            u_by_block,
            self.max_new_tokens,
            keep_abs_pos=self.keep_abs_pos_ar,
        )
        out_ar = self._forward_ar_with_flash(model, ids_ar, attn_1d, pos_ar)

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index=IGNORE_TOKEN_ID)
        loss_ar = label_smoother(out_ar, lbls_ar.unsqueeze(0), shift_labels=True)
        loss_ar = loss_ar * self.ar_scale

        # ----- Consistency loss: ONE forward pass over local windows -----
        # Build teacher labels on GPU: [prompt] + concat(last_j)  (full N per block)
        labels_last = self._build_labels_last_compressed_gpu(input_ids, prompt_len, self.max_new_tokens)

        loss_cons = self._consistency_one_pass_local_windows_batch(
            model,
            input_ids,       # GPU
            labels_last,     # GPU, compressed teacher labels
            prompt_len,
            u_by_block,
            N=self.max_new_tokens,
            W=self.consistency_window,
            K=self.consistency_num_segments,
        )

        if getattr(self.args, "local_rank", -1) in (-1, 0):
            wandb.log(
                {
                    "ar loss": float(loss_ar.detach().cpu()),
                    "consistency loss": float(loss_cons.detach().cpu()),
                }
            )

        # Two backwards to free each graph earlier → lower peak VRAM
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_ar)
            self.accelerator.backward(loss_cons)

        total = (loss_ar + loss_cons).detach()
        return total
