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

        self.train_step_cnt = 0

        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels
        # cache BlockMasks keyed by (L, prompt_len, heads) — no u_list dependence
        self._blockmask_cache = {}

    # ---------------- Utilities ---------------- #

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

    @staticmethod
    def _index_layout(prompt_len: int, T: int, N):
        """Return lists of start indices for all k_j and last_j blocks in flattened sequence."""
        k_starts = [prompt_len + 2 * j * N for j in range(T)]
        l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
        return k_starts, l_starts

    # FlexAttention BlockMask
    def _build_block_mask(self, L: int, prompt_len: int, T: int, heads: int):
        N = self.max_new_tokens
        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        ks = torch.tensor(k_starts, device=self.args.device)  # [T]
        ls = torch.tensor(l_starts, device=self.args.device)  # [T]
        num_traj = T

        def mask_mod(b, h, q, k):
            # q, k: any shape, torch tensors (can be batched)
            rel_q = q - prompt_len
            rel_k = k - prompt_len
            block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
            block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

            is_prompt_q = q < prompt_len
            is_prompt_k = k < prompt_len
            is_kj_q = (q >= prompt_len) & (block_idx_q % 2 == 0)
            is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
            is_kj_k = (k >= prompt_len) & (block_idx_k % 2 == 0)
            is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)
            j_q = torch.clamp(block_idx_q // 2, min=0)
            j_k = torch.clamp(block_idx_k // 2, min=0)

            # k_in_prev_last: is_lastj_k & (block_idx_k < 2 * j_q)
            k_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)

            ks_per_q = ks[torch.clamp(j_q, max=len(ks) - 1)]
            ls_per_q = ls[torch.clamp(j_q, max=len(ls) - 1)]

            mask_prompt = is_prompt_q & (k <= q)

            mask_kj = is_kj_q & (
                is_prompt_k |
                k_in_prev_last |
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            mask_lastj = is_lastj_q & (
                is_prompt_k |
                k_in_prev_last |
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            mask = mask_prompt | mask_kj | mask_lastj
            return mask

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device=self.args.device
        )
        
        return block_mask


    # ---------------- Core Training Step ----------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        self.train_step_cnt += 1
        return self._one_pass_losses_step(model, inputs)

    def _one_pass_losses_step(self, model, inputs):
        """
        Single forward pass to compute:
        - AR loss: first u_j tokens of each last_j (shifted LM)
        - Consistency loss: corrupted tail of each k_j vs teacher last_j at same offsets
        """
        input_ids, prompt_len, T = self._unpack_sample(inputs)

        L = input_ids.size(0)
        expected_len = prompt_len + 2 * T * self.max_new_tokens
        if L != expected_len:
            raise ValueError(
                f"Length mismatch: L={L}, expected {expected_len} (prompt_len={prompt_len}, T={T}, n_token_sequence_size={self.max_new_tokens})"
            )

        num_heads = getattr(getattr(model, "config", None), "num_attention_heads", 1)
        blk_mask = self._build_block_mask(L, prompt_len, T, num_heads)
        
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            block_mask=blk_mask,
            attn_implementation="flex_attention",
        )
        # [1, L, V]
        logits = outputs.logits
        del blk_mask

        # ========== AR loss ==========
        ar_labels = torch.full((L,), IGNORE_TOKEN_ID, device=self.args.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, self.max_new_tokens)
        for j in range(T):
            ls = l_starts[j]
            ar_labels[ls : ls + self.max_new_tokens] = input_ids[ls : ls + self.max_new_tokens]

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index=IGNORE_TOKEN_ID)
        loss_ar = label_smoother(
            outputs, ar_labels.unsqueeze(0), shift_labels=True
        ) * 10.0

        del ar_labels, label_smoother
        torch.cuda.empty_cache()

        # ========== Consistency loss (soft) ==========
        T_soft = getattr(self.args, "distill_temperature", 1.0)
        student_positions, teacher_positions = [], []
        for j in range(T):
            ks, ls = k_starts[j], l_starts[j]
            offs = range(self.max_new_tokens)
            student_positions.extend(ks + off for off in offs)
            teacher_positions.extend(ls + off for off in offs)

        if len(student_positions) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            # [M, V]
            student_logits_sel = logits[0, student_positions, :]              
            teacher_logits_sel = logits[0, teacher_positions, :].detach()

            log_ps = F.log_softmax(student_logits_sel / T_soft, dim=-1)
            p_t   = F.softmax(teacher_logits_sel / T_soft, dim=-1)
            loss_consistency = (-(p_t * log_ps).sum(dim=-1)).mean() * (T_soft * T_soft)

            del student_logits_sel, teacher_logits_sel, log_ps, p_t
            torch.cuda.empty_cache()

        del logits, k_starts, l_starts
        torch.cuda.empty_cache()

        total_loss = loss_ar + loss_consistency

        if self.args.qlora:
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            wandb.log(
                {
                    "ar loss": float(loss_ar.detach().cpu()),
                    "consistency loss": float(loss_consistency.detach().cpu()),
                }
            )

        del loss_ar, loss_consistency, outputs
        torch.cuda.empty_cache()

        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return total_loss.detach()
