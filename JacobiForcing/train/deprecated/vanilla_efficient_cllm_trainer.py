# one_pass_trainer.py
#
# CllmTrainer that computes BOTH AR loss and Consistency loss
# in a SINGLE forward pass by constructing a 2D attention mask.
#
# Assumptions:
# - Inputs per sample (batch size == 1 expected here):
#     inputs["input_ids"]            : Tensor [1, L]  (flattened: prompt + k_0 + last_0 + k_1 + last_1 + ... )
#     inputs["labels_ids"]           : Tensor [1, L]  (teacher/fixed sequence; same shape as input_ids)
#     inputs["prompt_ids_len"]       : Tensor([P]) or int  (length of prompt prefix)
#     inputs["traj_position_indices"]: Tensor [1, T] or list[int]; u_j for each k_j (uncorrupted prefix length in k_j)
#
# - Block layout (N_BLOCK = 64):
#     input_ids =
#       [ prompt(P),
#         k_0(64), last_0(64),
#         k_1(64), last_1(64),
#         ...
#         k_{T-1}(64), last_{T-1}(64) ]
#
# - AR loss:
#     supervise ONLY the first u_j tokens of each last_j block (shifted LM loss),
#     where u_j = traj_position_indices[j].
#     Visibility for a last_j token at offset t (< u_j):
#       prompt + all previous last_m (m<j) + previous tokens in last_j (causal).
#
# - Consistency loss:
#     for each corrupted tail token of k_j (offset t in [u_j, 64)),
#     compare student logits at k_j[t] against teacher token at last_j[t] (unshifted CE).
#     Visibility for a k_j token at offset t:
#       prompt + all previous last_m (m<j) + previous tokens in k_j (causal).
#
# Notes:
# - This builds a boolean [1, L, L] attention mask. For some HF models/kernels,
#   you may need to disable flash attention (e.g., use attn_implementation="eager"/SDPA).
# - Batch size > 1 can be added by looping over batch dimension and stacking masks/logits.

import torch
import wandb
from torch.cuda.amp import autocast
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
N_BLOCK = 64  # tokens per (k_j / last_j) block


class CllmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels

    # ---------------- Utilities ---------------- #

    @staticmethod
    def _to_int(x):
        return x.item() if isinstance(x, torch.Tensor) else int(x)

    def _unpack_sample(self, inputs):
        """
        Extract a single sample. (Assumes per_device_train_batch_size == 1.)
        Required keys:
          - input_ids, labels_ids: [1, L]
          - prompt_ids_len: scalar or [1]
          - traj_position_indices: [1, T] or list[int]
        """
        input_ids = inputs["input_ids"][0]

        label_ids = inputs["labels_ids"][0]
        prompt_len = inputs["prompt_ids_len"]
        if isinstance(prompt_len, torch.Tensor):
            if prompt_len.dim() > 0:
                prompt_len = prompt_len[0]
        prompt_len = self._to_int(prompt_len)

        traj_position_indices = inputs["traj_position_indices"][0]
        if isinstance(traj_position_indices, torch.Tensor):
            traj_position_indices = traj_position_indices.tolist()
        traj_position_indices = [int(u) for u in traj_position_indices]

        return (
            input_ids.to(self.args.device),
            label_ids.to(self.args.device),
            prompt_len,
            traj_position_indices,
        )


    @staticmethod
    def _index_layout(prompt_len: int, T: int, N: int = N_BLOCK):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts


    def _build_full_attention_mask(self, L: int, prompt_len: int, u_list, N: int = N_BLOCK):
        """
        Build a single boolean attention mask of shape [1, L, L] that encodes:
          - prompt causal
          - k_j rows attend to: prompt + all previous last blocks + previous tokens in k_j
          - last_j rows (for t<u_j) attend to: prompt + prev last blocks + previous tokens in last_j
          - last_j rows for t>=u_j keep at least self-attend to avoid NaNs; they don't contribute to loss
        """
        device = self.args.device
        T = len(u_list)
        M = torch.zeros((L, L), dtype=torch.bool, device=device)

        # Prompt causal
        for i in range(prompt_len):
            M[i, : i + 1] = True

        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        def prev_last_sources(j_idx):
            allowed = list(range(prompt_len))
            for m in range(j_idx):
                allowed.extend(range(l_starts[m], l_starts[m] + N))
            return allowed

        for j, u_j in enumerate(u_list):
            u = max(0, min(N, int(u_j)))
            ks, ls = k_starts[j], l_starts[j]
            allowed_k_prefix = prev_last_sources(j)
            allowed_l_prefix = prev_last_sources(j)

            # k_j rows: prompt + prev last_m + own causal k_j
            for t in range(N):
                r = ks + t
                if allowed_k_prefix:
                    M[r, allowed_k_prefix] = True
                M[r, ks : r + 1] = True

            # last_j rows: allow only first u tokens; others (>=u) keep self-attend
            for t in range(N):
                r = ls + t
                if t < u:
                    if allowed_l_prefix:
                        M[r, allowed_l_prefix] = True
                    M[r, ls : r + 1] = True
                else:
                    M[r, r] = True  # numerical safety

        return M.unsqueeze(0)  # [1, L, L]

    # ---------------- Core Training Step ---------------- #
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._one_pass_losses_step(model, inputs)

    def _one_pass_losses_step(self, model, inputs):
        """
        Single forward pass to compute:
          - AR loss: first u_j tokens of each last_j (shifted LM)
          - Consistency loss: corrupted tail of each k_j vs teacher last_j at same offsets
        """
        input_ids, label_ids, prompt_len, u_list = self._unpack_sample(inputs)

        # Basic layout
        L = input_ids.size(0)
        T = len(u_list)
        expected_len = prompt_len + 2 * T * N_BLOCK
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} (prompt_len={prompt_len}, T={T}, N_BLOCK={N_BLOCK})"
            )

        # Build attention mask & forward once
        full_mask = self._build_full_attention_mask(L, prompt_len, u_list, N_BLOCK)  # [1, L, L]
        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=full_mask)
        logits = outputs.logits  # [1, L, V]
        vocab_size = logits.size(-1)

        # AR loss (shifted LM on first u_j tokens of each last_j)
        ar_labels = torch.full((L,), IGNORE_TOKEN_ID, device=self.args.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, N_BLOCK)
        for j, u_j in enumerate(u_list):
            u = max(0, min(N_BLOCK, int(u_j)))
            if u > 0:
                ls = l_starts[j]
                ar_labels[ls : ls + u] = label_ids[ls : ls + u]

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index=IGNORE_TOKEN_ID)
        loss_ar = label_smoother(
            type("obj", (), {"logits": logits}), ar_labels.unsqueeze(0), shift_labels=True
        )
        loss_ar = loss_ar * 10.0  # scale (as in your previous code)

        # Consistency loss (unshifted CE on corrupted tail of k_j)
        student_positions = []
        target_token_ids = []
        for j, u_j in enumerate(u_list):
            u = max(0, min(N_BLOCK, int(u_j)))
            if u < N_BLOCK:
                ks, ls = k_starts[j], l_starts[j]
                # collect k_j offsets [u..N_BLOCK-1]
                offs = range(u, N_BLOCK)
                student_positions.extend(ks + off for off in offs)
                target_token_ids.extend(label_ids[ls + off] for off in offs)

        if student_positions:
            student_logits_sel = logits[0, student_positions, :]  # [M, V]
            targets = torch.stack(target_token_ids)               # [M]
            loss_consistency = torch.nn.functional.cross_entropy(
                student_logits_sel, targets, reduction="mean"
            )
        else:
            loss_consistency = logits.sum() * 0.0  # zero

        total_loss = loss_ar + loss_consistency

        if self.args.qlora:
            total_loss.requires_grad = True

        # Logging
        if self.args.local_rank == 0:
            wandb.log(
                {
                    "ar loss": float(loss_ar.detach().cpu()),
                    "consistency loss": float(loss_consistency.detach().cpu()),
                }
            )

        # Backprop
        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        # Sync & return
        torch.distributed.barrier()
        return total_loss.detach()
