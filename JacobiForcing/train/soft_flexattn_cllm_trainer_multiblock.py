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
    
    def _duplicate_prefix_mask(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool: True where token should be masked because it's in k_j's prefix
        identical to last_j (from left to right until first divergence).
        Only k_j tokens are masked; last_j tokens are not.
        """
        device = input_ids.device
        N = self.max_new_tokens
        L = input_ids.size(0)
        mask = torch.zeros(L, dtype=torch.bool, device=device)

        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        for j in range(T):
            ks = k_starts[j]
            ls = l_starts[j]
            k_block = input_ids[ks:ks + N]
            l_block = input_ids[ls:ls + N]

            eq = (k_block == l_block)
            # find first index where they differ
            if torch.any(~eq):
                first_diff = int(torch.nonzero(~eq, as_tuple=False)[0])
            else:
                # fully identical: mask the whole k_j block
                first_diff = N

            if first_diff > 0:
                mask[ks:ks + first_diff] = True

        return mask

    def _build_padding_mask_for_loss(self, input_ids: torch.Tensor, prompt_len: int, T: int) -> torch.Tensor:
        """
        [L] bool padding mask used for losses:
        True = mask, False = keep.
        Combines: (a) PAD tokens, (b) duplicate-prefix in each k_j vs last_j.
        """
        device = input_ids.device
        pad_id = getattr(self.processing_class, "pad_token_id", None)

        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
        if pad_id is not None:
            mask |= (input_ids == pad_id)

        mask |= self._duplicate_prefix_mask(input_ids, prompt_len, T)
        return mask

    def _block_keep_mask_divergence_and_eos(
        self,
        input_ids: torch.Tensor,
        k_start: int,
        l_start: int,
        N: int,
        eos_id: int | None,
        drop_last_offset: bool = False,
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

    def _build_shared_position_ids(self, L: int, prompt_len: int, T: int):
        """
        Build [L] position_ids so each (k_j, last_j) share the same positions.
        Prompt: 0...prompt_len-1
        For each j: both blocks use prompt_len + j*N .. prompt_len + j*N + (N-1)
        """
        device = self.args.device
        N = self.max_new_tokens
        pos = torch.empty(L, dtype=torch.long, device=device)

        # Prompt positions
        pos[:prompt_len] = torch.arange(prompt_len, device=device)

        # Pair-shared positions
        k_starts, l_starts = self._index_layout(prompt_len, T, N)
        rel = torch.arange(N, device=device)
        for j in range(T):
            base = prompt_len + j * N
            ks = k_starts[j]
            ls = l_starts[j]
            pos[ks:ks + N] = base + rel
            pos[ls:ls + N] = base + rel

        return pos
    
    def _flip_block_after_eos_to_pad(self, input_ids: torch.Tensor, start: int, N: int, eos_id: int | None, pad_id: int | None) -> int:
        """Mutate input_ids[start:start+N] so that tokens AFTER first EOS become PAD.
        Returns number of tokens flipped."""
        if eos_id is None or pad_id is None:
            return 0
        block = input_ids[start:start+N]
        pos = (block == eos_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return 0
        k = int(pos[0])                  # eos offset inside block
        flip_start = start + k + 1
        flip_end   = start + N
        if flip_start < flip_end:
            input_ids[flip_start:flip_end] = pad_id
            return flip_end - flip_start
        return 0
    
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        if (~padding_mask).sum() == 0:
            return 0 * predicts[0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy = entropy.masked_fill(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    # FlexAttention BlockMask
    # - prompt queries: causal within prompt
    # - k_j queries: causal within *their own* k_j block + prompt
    # - last_j queries: causal within *their own* last_j block + prompt
    def _build_block_mask(self, L: int, prompt_len: int, T: int, heads: int):
        N = self.max_new_tokens
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ks = torch.tensor(k_starts, device=self.args.device)
        ls = torch.tensor(l_starts, device=self.args.device)

        def mask_mod(b, h, q, k):
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

            # for k_j queries, allow attending to ALL PREVIOUS k_* blocks
            # (k blocks have even block indices: 0, 2, 4, ... 2*(j_q-1))
            k_in_prev_k   = is_kj_k    & (block_idx_k < 2 * j_q)
            # keep old behavior for last_j (still sees previous last_*)
            last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)

            # prompt is always causal
            mask_prompt = is_prompt_q & (k <= q)

            # k_j queries:
            same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
            mask_kj = is_kj_q & (
                is_prompt_k |
                k_in_prev_k |  # attends to "previous k_*"
                (same_kj_block & (k >= ks_per_q) & (k <= q))
            )

            # last_j queries (UNCHANGED):
            same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
            mask_lastj = is_lastj_q & (
                is_prompt_k |
                last_in_prev_last |  # attends to all previous last_{<j}
                (same_lastj_block & (k >= ls_per_q) & (k <= q))
            )

            return mask_prompt | mask_kj | mask_lastj

        block_mask = create_block_mask(
            mask_mod, B=1, H=heads, Q_LEN=L, KV_LEN=L, device=self.args.device, _compile=True,
        )
        return block_mask

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

        # ===== Debug printing =====
        # detok the AR input sequence
        #prompt_ids_block = input_ids[:prompt_len]
        #l_blocks_concat = torch.cat([input_ids[ls : ls + N] for ls in l_starts], dim=0)
        #ar_concat_ids = torch.cat([prompt_ids_block, l_blocks_concat], dim=0)

        #print("\n=== AR INPUTS (prompt + concatenated l_j blocks) ===")
        
        # Decode
        #ar_text = self.processing_class.decode(ar_concat_ids, skip_special_tokens=False)

        #print("\n[Decoded text]")
        #print(ar_text)

        # Print all k_j blocks separately
        #print("\n=== k_j BLOCKS ===")
        #for j, ks in enumerate(k_starts):
        #    block_ids = input_ids[ks : ks + N]
        #    block_text = self.processing_class.decode(block_ids, skip_special_tokens=False)
        #    print(f"[k_{j}]")
        #    print(block_text)
        #    print()
        # ==========================

        attn_mask = torch.ones(L, dtype=torch.long, device=input_ids.device)
        k_starts, l_starts = self._index_layout(prompt_len, T, N)

        ### METHOD 1: PAD post-EOS TOKENS on last_N
        self._flip_block_after_eos_to_pad(input_ids, l_starts[-1], N, eos_id, pad_id)
        
        ###-- METHOD 2: cut post-EOS inside k_N & last_N block
        #--for j in range(T):
        #--    starting_pos_k = k_starts[j]
        #--    starting_pos_l = l_starts[j]
        
        #--    block_k = input_ids[starting_pos_k : starting_pos_k + N]
        #--    block_l = input_ids[starting_pos_l : starting_pos_l + N]

        #--    pos = (block_l == eos_id).nonzero(as_tuple=False)
        #--    if pos.numel():
        #--        first_eos_pos_k = starting_pos_k + int(pos[0])
        #--        first_eos_pos_l = starting_pos_l + int(pos[0])
        #--        attn_mask[first_eos_pos_k + 1 : starting_pos_k + N] = 0
        #--        attn_mask[first_eos_pos_l + 1 : starting_pos_l + N] = 0

        # Build structural block mask
        num_heads = getattr(self.cfg, 'num_attention_heads', 28)
        #print(f"num heads from config: {self.cfg.num_attention_heads}")
        #print(f"[block mask] num_heads={num_heads}, L={L}, prompt_len={prompt_len}, T={T}, N={N}")
        blk_mask = self._build_block_mask(L, prompt_len, T, num_heads)
        # shared position ids for each (k_j, last_j) pair
        position_ids = self._build_shared_position_ids(L, prompt_len, T)

        # ---- DEBUG: mask + positions sanity checks ----
        #-if (self.args.local_rank in (-1, 0)) and self.debug_masks and (self.train_step_cnt % self.debug_every == 0):
        #-    with torch.no_grad():
        #-        # Invariants of your current flex pattern
        #-        self._assert_invariants(blk_mask, prompt_len, N, T, L)
        #-        # RoPE positions are shared within each (k_j, last_j) pair
        #-        self._check_shared_positions(position_ids, prompt_len, T)
                # Human-readable peek
        #-        self._dump_visibility(blk_mask, prompt_len, N, T, L, max_rows=3)
        # -----------------------------------------------

        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=blk_mask,
            #block_mask=blk_mask,
            position_ids=position_ids.unsqueeze(0),
            attn_implementation="flex_attention",
        )
        logits = outputs.logits

        # ========== AR loss ==========
        # gather (logit_pos --> target_pos) pairs
        # run soft CE.
        pair_logit_positions = []
        pair_target_positions = []

        def add_forward_pairs(seg_start: int, seg_end: int):
            """
            Add all in-block next-token pairs for a segment [seg_start, seg_end).
            Produces pairs (p --> p+1) for p in [seg_start .. seg_end-2].
            """
            p = torch.arange(seg_start, seg_end - 1, device=self.args.device, dtype=torch.long)
            t = p + 1
            pair_logit_positions.append(p)
            pair_target_positions.append(t)

        # Prompt segment
        end_prompt = prompt_len
        add_forward_pairs(0, end_prompt)

        # for each j, we need an effective length ('end' from j-1) to compute first-token loss in last_j
        for j in range(T):
            ls = l_starts[j]

            # first append bridging token
            if j == 0:
                # map last token from prompt to first token in last_0
                logit_pos = end_prompt - 1
                target_pos = ls
            elif j > 0:
                prev_ls = l_starts[j - 1]
                logit_pos = prev_ls + (N - 1)
                target_pos = ls

            pair_logit_positions.append(torch.tensor([logit_pos], device=self.args.device))
            pair_target_positions.append(torch.tensor([target_pos], device=self.args.device))

            # handle (k_j, last_j) block
            block = input_ids[ls : ls + N]

            # respect EOS inside the block
            eos_pos = None
            if eos_id is not None:
                epos = torch.nonzero(block == eos_id, as_tuple=False)
                eos_pos = int(epos[0]) if epos.numel() > 0 else None

            end = N
            if eos_pos is not None:
                end = min(end, eos_pos + 1)  # include EOS in segment
                
                # mark PAD as 0 for attn_mask
                mask_block = attn_mask[ls : ls + N]
                mask_block[block == pad_id] = 0
                attn_mask[ls : ls + N] = mask_block

            #print(f"ending position for block {j}: {end}")

            # in-block forward pairs
            add_forward_pairs(ls, ls + end)

            #--if eos_pos is not None:
            #--    break

        # Bridges: (last_{j-1} last token logits) -> (first token of last_j)
        # Note: This skips the intervening k_j block
        #for j in range(1, T):
            # length of the 
        #    prev_end = last_effective_ends[j - 1]

        #    prev_ls = l_starts[j - 1]
            # last token logit from last_{j-1}
        #    logit_pos = prev_ls + (prev_end - 1)
            # first token target from last_j
        #    target_pos = l_starts[j]

            # edge case: skip if target is PAD
        #    if pad_id is not None and input_ids[target_pos].item() == pad_id:
        #        continue

        #    pair_logit_positions.append(torch.tensor([logit_pos], device=self.args.device, dtype=torch.long))
        #    pair_target_positions.append(torch.tensor([target_pos], device=self.args.device, dtype=torch.long))

        # Compute CE over all pairs
        if len(pair_logit_positions) == 0:
            loss_ar = torch.zeros((), device=self.args.device)
        else:
            p_all = torch.cat(pair_logit_positions, dim=0)
            t_all = torch.cat(pair_target_positions, dim=0)

            ar_logits  = logits[0, p_all, :].clone()                          # [K, V]
            ar_targets = input_ids.index_select(0, t_all).clone().detach()    # [K]
            
            # ===== DEBUG PRINTING ===== #
            if self.args.local_rank == 0:
                print(f"===== decoded last_N AR targets =====\n{self.processing_class.decode(ar_targets[-64:])}\n==========\n")
                print(f"===== last_N AR tokens =====\n{ar_targets[-64:]}\n==========\n")
            # ===== DEBUG PRINTING ===== #

            ar_targets[ar_targets == pad_id] = -100  # respect ignore_index

            max_values, max_indices = ar_logits.max(dim=-1)
            #print(f"\ninput ids: {input_ids.tolist()}")

            #print(f"\nlabels length: {len(max_values)}")
            #print(f"\nargmax length: {len(max_indices)}")
            #print("\nmax logits:", max_values.tolist())
            #print("\nargmax indices:", max_indices.tolist())

            #print(f"\nattention mask length: {len(attn_mask.tolist())}")
            #print(f"\nattention mask: {attn_mask.tolist()}")
            #print(f"block attention mask blk_mask: {blk_mask}")

            #print(f"\nprompt length: {prompt_len}")
            #print(f"\nEOS position offset: {eos_pos}")
            #print(f"\ngeneration max logits: {max_values[prompt_len:].tolist()}")
            #print(f"generation indices: {max_indices[prompt_len:].tolist()}")

            loss_ar = F.cross_entropy(
                ar_logits.float(),
                ar_targets,
                reduction="mean",
                label_smoothing=0.0,
                ignore_index=-100,
            ) * 10

        # ========== Consistency loss ==========
        T_soft = getattr(self.args, "distill_temperature", 1.0)

        # learn to predict first token in next block (last_{j+1})  from the last token in k_j
        drop_last_offset = False
        if drop_last_offset:
            offs = torch.arange(N - 1, device=self.args.device)
        else:
            offs = torch.arange(N, device=self.args.device)

        student_positions, teacher_positions = [], []
        for j in range(T):
            ks, ls = k_starts[j], l_starts[j]
            pair_keep = self._block_keep_mask_divergence_and_eos(
                input_ids, ks, ls, N, eos_id=eos_id, drop_last_offset=drop_last_offset
            )
            if pair_keep.any():
                sp = ks + offs[pair_keep]
                tp = ls + offs[pair_keep]
                student_positions.append(sp)
                teacher_positions.append(tp)

        if len(student_positions) == 0:
            loss_consistency = torch.zeros((), device=self.args.device)
        else:
            sp = torch.cat(student_positions, dim=0)  # [K]
            tp = torch.cat(teacher_positions, dim=0)  # [K]

            # build global [L] padding mask: PADs and duplicate k_j prefixes
            global_pad_and_dup_mask = self._build_padding_mask_for_loss(input_ids, prompt_len, T)
            
            # per-pair padding mask for soft CE (True = masking out)
            padding_mask = global_pad_and_dup_mask.index_select(0, sp)

            # Build logits for all candidate pairs
            student_logits_all = logits[0, sp, :]
            teacher_logits_all = logits[0, tp, :].detach()

            student_logits_all = student_logits_all / T_soft
            teacher_logits_all = teacher_logits_all / T_soft

            loss_consistency = self.soft_cross_entropy(
                student_logits_all.float(),
                teacher_logits_all.float(),
                padding_mask
            )

            loss_consistency = loss_consistency * (T_soft * T_soft) / T

        total_loss = loss_ar + loss_consistency

        if self.args.qlora:
            total_loss.requires_grad = True

        if self.args.local_rank == 0:
            wandb.log({"ar loss": float(loss_ar.detach().cpu()),
                    "consistency loss": float(loss_consistency.detach().cpu())})

        #del outputs, logits
        torch.cuda.empty_cache()

        with self.accelerator.accumulate(model):
            self.accelerator.backward(total_loss)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return total_loss.detach()
