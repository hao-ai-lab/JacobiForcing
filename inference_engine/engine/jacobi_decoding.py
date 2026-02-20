from __future__ import annotations

from typing import Callable, List, Optional, Sequence as PySequence, Tuple
import random
import os

import torch
from torch import Tensor

from inference_engine.engine.sequence import Sequence
from inference_engine.engine.block_manager import BlockManager
from inference_engine.sampling_params import SamplingParams


# ===================
# Profiler helper
# ===================
def _get_profiler():
    """Get profiler instance if available and PROFILE=1."""
    if os.environ.get("PROFILE", "0") != "1":
        return None
    from inference_engine.engine.model_runner import get_profiler
    return get_profiler()



# -----------------------------------------------------------------------------
# Forward function types
# -----------------------------------------------------------------------------
# forward_step (single):
#   - takes (seq, draft_tokens: LongTensor[1, L])
#   - draft[0] = seed (last token in seq, already has KV)
#   - draft[1:] = speculative tokens to verify
#   - forwards all L tokens at positions [S-1, S, ..., S+L-2]
#   - returns logits: FloatTensor[1, L-1, vocab] for verifying draft[1:]
LogitsForwardFn = Callable[[Sequence, Tensor], Tensor]

# forward_step_batch (batched):
#   - takes (seqs, draft_tokens: LongTensor[B, L])
#   - draft[:, 0] = seeds (last tokens in seqs, already have KV)
#   - draft[:, 1:] = speculative tokens to verify
#   - forwards all L tokens at positions [S-1, S, ..., S+L-2]
#   - returns logits: FloatTensor[B, L-1, vocab] for verifying draft[:, 1:]
LogitsForwardFnBatch = Callable[[List[Sequence], Tensor], Tensor]


class JacobiDecoder:
    """
    Minimal Jacobi-style greedy verification (K=1) with KV rollback for rejected
    suffix tokens. No multiblock, no n-gram pool, no candidate recycling.

    Supports:
      - single sequence: generate_chunk(seq)
      - batched sequences: generate_chunk_batch(seqs), with group-by-L to avoid
        padding / masking assumptions in nano-vLLM.

    Required from caller:
      - forward_step(seq, draft) for single
      - optionally forward_step_batch(seqs, draft_batch) for real batched model
        execution. If not provided, we fall back to looping forward_step.

    Notes:
      - This is "greedy Jacobi": draft is verified against greedy predictions,
        accept longest exact-match prefix, rollback KV for rejected tail, then
        rebuild next draft from greedy tokens.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        forward_step: Optional[LogitsForwardFn] = None,
        forward_step_batch: Optional[LogitsForwardFnBatch] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if forward_step is None and forward_step_batch is None:
            raise ValueError("Provide at least one of forward_step or forward_step_batch.")

        self.block_manager = block_manager
        self.forward_step = forward_step
        self.forward_step_batch = forward_step_batch
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        # vocab_size should ALWAYS be provided from model config (151643 for Qwen2.5)
        # The old default of 151936 was incorrect
        if vocab_size is None:
            raise ValueError("vocab_size must be provided from model config. Do not use hard-coded values.")
        self.vocab_size = vocab_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Cache debug flag once at init (avoid os.environ.get() per iteration)
        import os
        self.debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        # Statistics tracking for testing/profiling
        self.stats = {
            'num_chunk_calls': 0,
            'num_jacobi_iterations': 0,  # Total iterations across all chunk calls
            'tokens_accepted': 0,
            'tokens_per_call': [],        # Tokens per chunk call
            'tokens_per_iteration': [],   # Tokens per individual Jacobi iteration
            'iterations_per_call': [],    # Iterations per chunk call
        }

    # ----------------- config helpers -----------------

    def _get_sampling_cfg(self, seq: Sequence) -> Tuple[int, int]:
        """Extract Jacobi hyper-params from SamplingParams with safe defaults."""
        sp: Optional[SamplingParams] = getattr(seq, "sampling_params", None)

        def g(name: str, default):
            return getattr(sp, name, default) if sp is not None else default

        block_len = int(g("jacobi_block_len", 64))
        max_iterations = int(g("jacobi_max_iterations", 128))
        return block_len, max_iterations

    # ----------------- token_ids GPU cache -----------------
    
    def _get_token_ids_gpu(self, seq: Sequence) -> Tensor:
        """Get cached GPU version of token_ids, updating only when length changes."""
        current_len = len(seq.token_ids)
        cached_tensor = getattr(seq, '_token_ids_gpu', None)
        
        # Check if cache is valid
        if cached_tensor is None or cached_tensor.size(0) != current_len:
            # Cache miss or stale - update it
            seq._token_ids_gpu = torch.as_tensor(
                seq.token_ids, dtype=torch.long, device=self.device
            )
            return seq._token_ids_gpu
        
        return cached_tensor

    # ----------------- draft helpers -----------------

    def _init_draft(self, seq: Sequence, length: int) -> Tensor:
        """
        Initialize draft with seed + speculative tokens.
        
        draft[0] = seed (last committed token, already has KV at position S-1)
        draft[1:] = speculative tokens (to be verified)
        """
        if length <= 0:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)

        token_ids = getattr(seq, "token_ids", None) or []
        
        # Seed is the last committed token
        if token_ids:
            seed = token_ids[-1]
        else:
            seed = (
                self.pad_token_id
                if self.pad_token_id is not None
                else (self.eos_token_id if self.eos_token_id is not None else 0)
            )
        
        # Pre-allocate draft tensor on GPU
        d = torch.empty((1, length), dtype=torch.long, device=self.device)
        d[0, 0] = seed
        
        if length > 1:
            d[0, 1:] = torch.randint(0, self.vocab_size, (length - 1,), device=self.device)
        
        return d
    
    def _init_draft_from_greedy(self, seq: Sequence, length: int) -> Tensor:
        """
        Initialize draft by forwarding current sequence and taking greedy predictions.

        Note: For simplicity, we use random init.
        """
        # Fall back to random init
        return self._init_draft(seq, length)

    def _resize_or_init_draft(self, seq: Sequence, draft: Optional[Tensor], L: int) -> Tensor:
        """
        Ensure draft is [1, L] on self.device.
        draft[0] must be the seed (last token in seq).
        """
        if L <= 0:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)

        if draft is None or draft.numel() == 0:
            return self._init_draft(seq, L)

        draft = draft.to(self.device)
        if draft.dim() != 2 or draft.size(0) != 1:
            raise ValueError(f"draft must be [1, L], got {tuple(draft.shape)}")

        # Verify/fix seed
        expected_seed = seq.token_ids[-1] if seq.token_ids else 0
        draft[0, 0] = expected_seed

        cur_L = int(draft.size(1))
        if cur_L == L:
            return draft
        if cur_L > L:
            return draft[:, :L]

        pad = draft[:, -1:].expand(1, L - cur_L)
        return torch.cat([draft, pad], dim=-1)

    # ----------------- forward helpers -----------------

    @torch.inference_mode()
    def _forward_batched(self, seqs: List[Sequence], draft_batch: Tensor) -> Tensor:
        """
        Returns logits [B, L-1, vocab] for verifying speculative tokens draft[:, 1:].
        
        Args:
            seqs: List of B sequences
            draft_batch: [B, L] where draft[:, 0] = seed (last committed token)
                                      draft[:, 1:] = speculative tokens
        
        Returns:
            logits: [B, L-1, vocab] predictions for verifying draft[:, 1:]
                    - logits[:, i] verifies draft[:, i+1]
        """
        if draft_batch.dim() != 2:
            raise ValueError(f"draft_batch must be [B, L], got {tuple(draft_batch.shape)}")
        B, L = int(draft_batch.size(0)), int(draft_batch.size(1))
        if B != len(seqs):
            raise ValueError(f"B mismatch: got draft_batch B={B} but len(seqs)={len(seqs)}")

        if self.forward_step_batch is not None:
            logits = self.forward_step_batch(seqs, draft_batch)
        else:
            if self.forward_step is None:
                raise ValueError("No forward_step available for fallback loop.")
            logits_list = []
            for i, seq in enumerate(seqs):
                li = self.forward_step(seq, draft_batch[i : i + 1, :])
                logits_list.append(li)
            logits = torch.cat(logits_list, dim=0)

        # Expect [B, L-1, vocab] for verification of speculative tokens
        if logits.ndim != 3 or logits.size(0) != B or logits.size(1) != (L - 1):
            raise ValueError(
                f"forward must return logits [B, L-1, vocab] for verifying speculative tokens, "
                f"expected [{B}, {L-1}, *], got {tuple(logits.shape)}"
            )
        return logits

    # ----------------- core jacobi logic -----------------

    @staticmethod
    def _accept_lengths(draft: Tensor, greedy: Tensor) -> Tensor:
        """
        Verify speculative tokens against greedy predictions.
        
        Args:
            draft:  [B, L] where draft[:, 0] = seed (always accepted)
                                 draft[:, 1:] = speculative tokens to verify
            greedy: [B, L-1] predictions where greedy[:, i] verifies draft[:, i+1]
        
        Returns:
            acc_len: LongTensor[B] in [1..L]
                     Number of tokens to accept (1 = seed only, L = all accepted)
        
        Verification logic:
            - draft[0] is seed (always accepted)
            - Compare draft[i+1] == greedy[i] for i = 0, 1, 2, ...
            - acc_len = 1 + number of consecutive matches
        """
        B, L = draft.shape
        if L == 0:
            return torch.zeros((B,), device=draft.device, dtype=torch.long)
        
        if L == 1:
            # Only seed token, nothing to verify
            return torch.ones((B,), device=draft.device, dtype=torch.long)

        L_minus_1 = L - 1  # Number of speculative tokens to verify
        if greedy.size(1) != L_minus_1:
            raise ValueError(
                f"Expected greedy.shape[1]={L_minus_1}, got {greedy.size(1)}"
            )

        # Check if draft[1:] matches greedy predictions
        mismatch = (draft[:, 1:] != greedy)  # [B, L-1]
        
        # Count consecutive matches from the start
        num_matches = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)  # [B], range [0..L-1]
        
        # acc_len = 1 (seed) + num_matches
        return num_matches + 1

    # ----------------- public APIs -----------------

    @torch.inference_mode()
    def generate_chunk(self, seq: Sequence) -> List[int]:
        """Single-seq wrapper."""
        return self._generate_single_fast(seq)

    @torch.inference_mode()
    def _generate_single_fast(self, seq: Sequence) -> List[int]:
        """Fast path for single sequence."""
        debug = self.debug
        
        # Get config
        block_len, max_iters = self._get_sampling_cfg(seq)
        sp = getattr(seq, "sampling_params", None)
        max_tokens = getattr(sp, "max_tokens", 2048) if sp else 2048
        max_tokens -= seq.num_completion_tokens  # Remaining tokens
        
        if block_len <= 1:
            return []  # Can't do Jacobi with L=1
        
        accepted: List[int] = []
        q_draft: Optional[Tensor] = None
        eos_reached = False
        iters = 0
        
        # Check for prefill draft
        prefill_draft = getattr(seq, '_prefill_draft', None)
        
        profiler = _get_profiler()
        
        while not eos_reached and len(accepted) < max_tokens and iters < max_iters:
            iters += 1
            L = block_len
            if profiler: profiler.add_iteration()
            
            if profiler: profiler.start("jacobi.draft_build")
            if prefill_draft is not None and len(accepted) == 0 and q_draft is None:
                # Use prefill draft for first iteration
                seed = seq.token_ids[-1]
                d = torch.empty((1, L), dtype=torch.long, device=self.device)
                d[0, 0] = seed
                
                prefill_len = min(len(prefill_draft), L - 1)
                if prefill_len > 0:
                    d[0, 1:prefill_len+1] = torch.as_tensor(prefill_draft[:prefill_len], dtype=torch.long, device=self.device)
                
                if prefill_len < L - 1:
                    num_pad = L - 1 - prefill_len
                    d[0, prefill_len+1:] = torch.randint(0, self.vocab_size, (num_pad,), device=self.device)
                
                seq._prefill_draft = None
                q_draft = d
            else:
                q_draft = self._resize_or_init_draft(seq, q_draft, L)
            
            seq.draft_tokens = q_draft[0].tolist()
            if profiler: profiler.stop("jacobi.draft_build")
            
            logits = self.forward_step(seq, q_draft) if self.forward_step else self.forward_step_batch([seq], q_draft)
            
            if profiler: profiler.start("jacobi.verify")
            greedy = torch.argmax(logits[0], dim=-1)
            
            draft_spec = q_draft[0, 1:]
            mismatch = (draft_spec != greedy)
            acc_len = int((mismatch.cumsum(dim=-1) == 0).sum().item()) + 1
            acc_len = max(1, min(acc_len, L))
            
            if self.eos_token_id is not None and acc_len > 1:
                eos_mask = (q_draft[0, 1:acc_len] == self.eos_token_id)
                if eos_mask.any():
                    first_eos = int(torch.nonzero(eos_mask, as_tuple=False)[0].item())
                    acc_len = 1 + first_eos + 1
                    eos_reached = True
            
            if profiler: profiler.stop("jacobi.verify")
            if profiler: profiler.start("jacobi.commit")
            num_spec_accepted = acc_len - 1
            draft = seq.draft_tokens
            
            if num_spec_accepted > 0:
                accepted_tokens = draft[1:acc_len]
                seq.extend_tokens(accepted_tokens)
                if self.block_manager is not None:
                    self.block_manager.may_append_batch(seq, num_spec_accepted)
                accepted.extend(accepted_tokens)
            
            if acc_len == 1:
                next_token = greedy[0].item()
                seq.append_token(next_token)
                if self.block_manager is not None:
                    self.block_manager.may_append(seq)
                accepted.append(next_token)
                num_spec_accepted = 1
                
                if self.eos_token_id is not None and next_token == self.eos_token_id:
                    eos_reached = True
            
            if profiler: profiler.stop("jacobi.commit")
            
            if profiler: profiler.start("jacobi.trim")
            num_to_trim = L - 1 - num_spec_accepted
            if num_to_trim > 0 and self.block_manager is not None:
                self.block_manager.trim_kv_only_fast(seq, num_to_trim)
            if profiler: profiler.stop("jacobi.trim")
            
            seq.clear_draft()
            
            if len(seq) != seq.num_cached_tokens:
                raise RuntimeError(
                    f"Invariant violated: len(token_ids)={len(seq)} != num_cached_tokens={seq.num_cached_tokens}"
                )
            
            if profiler: profiler.add_tokens(num_spec_accepted if num_spec_accepted > 0 else 1)
            
            if eos_reached or len(accepted) >= max_tokens:
                break
            
            if profiler: profiler.start("jacobi.next_draft")
            new_seed = seq.token_ids[-1]
            d = torch.empty((1, L), dtype=torch.long, device=self.device)
            d[0, 0] = new_seed
            
            if acc_len < L:
                if acc_len == 1:
                    remaining = greedy[1:]
                else:
                    remaining = greedy[acc_len-1:]
                
                copy_len = min(len(remaining), L - 1)
                if copy_len > 0:
                    d[0, 1:copy_len+1] = remaining[:copy_len]
            else:
                d[0, 1] = greedy[-1].item()
                copy_len = 1
            
            if copy_len < L - 1:
                num_pad = L - 1 - copy_len
                d[0, copy_len+1:] = torch.randint(0, self.vocab_size, (num_pad,), device=self.device)
            
            q_draft = d
            if profiler: profiler.stop("jacobi.next_draft")
        
        self.stats['num_chunk_calls'] += 1
        self.stats['num_jacobi_iterations'] += iters
        self.stats['tokens_accepted'] += len(accepted)
        self.stats['tokens_per_call'].append(len(accepted))
        self.stats['iterations_per_call'].append(iters)
        
        return accepted

    @torch.inference_mode()
    def generate_chunk_batch(self, seqs: List[Sequence]) -> List[List[int]]:
        """Generate tokens continuously until max_tokens or EOS for each sequence."""
        if not seqs:
            return []
        
        if len(seqs) == 1:
            return [self._generate_single_fast(seqs[0])]

        B = len(seqs)
        accepted: List[List[int]] = [[] for _ in range(B)]
        q_draft: List[Optional[Tensor]] = [None for _ in range(B)]
        eos_reached = [False for _ in range(B)]
        iters = [0 for _ in range(B)]

        cfg = [self._get_sampling_cfg(s) for s in seqs]
        block_lens = [c[0] for c in cfg]
        max_iters = [c[1] for c in cfg]
        
        max_tokens = []
        for seq in seqs:
            sp = getattr(seq, "sampling_params", None)
            if sp is not None:
                max_tok = getattr(sp, "max_tokens", 2048)
                already_generated = seq.num_completion_tokens
                remaining = max_tok - already_generated
                max_tokens.append(max(0, remaining))
            else:
                max_tokens.append(2048)
        
        num_iterations_this_call = 0
        prev_accepted_lens = [0 for _ in range(B)]
        profiler = _get_profiler()

        while True:
            active = [
                i for i in range(B)
                if (not eos_reached[i])
                and (len(accepted[i]) < max_tokens[i])
                and (iters[i] < max_iters[i])
            ]
            
            debug = self.debug
            if debug:
                print(f"\n[MAIN_LOOP] Active sequences: {active}")
                for i in active:
                    print(f"  seq_{i}: accepted={len(accepted[i])}/{max_tokens[i]}, iters={iters[i]}/{max_iters[i]}, eos={eos_reached[i]}")
            
            if not active:
                if debug:
                    print(f"[MAIN_LOOP] No active sequences, exiting loop")
                break

            groups: dict[int, List[int]] = {}
            for i in active:
                L = block_lens[i]
                if L > 1:
                    groups.setdefault(L, []).append(i)

            if not groups:
                break
            
            num_iterations_this_call += 1
            tokens_this_iteration = 0
            if profiler: profiler.add_iteration()

            sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            for L, idxs in sorted_groups:
                debug = self.debug
                if debug:
                    print(f"\n[JACOBI_ITER] Processing group: L={L}, num_seqs={len(idxs)}, seq_idxs={idxs}")
                
                if profiler: profiler.start("jacobi.draft_build")
                drafts = []
                sub_seqs = []
                for i in idxs:
                    iters[i] += 1
                    if debug:
                        print(f"  [DRAFT_BUILD] seq_idx={i}, iter={iters[i]}, seq_len={len(seqs[i])}, " 
                              f"accepted_so_far={len(accepted[i])}, target={max_tokens[i]}")
                    
                    prefill_draft = getattr(seqs[i], '_prefill_draft', None)
                    if prefill_draft is not None and len(accepted[i]) == 0 and q_draft[i] is None:
                        seed = seqs[i].token_ids[-1]
                        
                        d = torch.empty((1, L), dtype=torch.long, device=self.device)
                        d[0, 0] = seed
                        
                        prefill_len = min(len(prefill_draft), L - 1)
                        if prefill_len > 0:
                            d[0, 1:prefill_len+1] = torch.as_tensor(prefill_draft[:prefill_len], dtype=torch.long, device=self.device)
                        
                        if prefill_len < L - 1:
                            num_pad = L - 1 - prefill_len
                            d[0, prefill_len+1:] = torch.randint(0, self.vocab_size, (num_pad,), device=self.device)
                        
                        seqs[i]._prefill_draft = None
                        if debug:
                            print(f"  [DRAFT_BUILD] Using prefill draft: {d[0, :5].tolist()}...")
                    else:
                        d = self._resize_or_init_draft(seqs[i], q_draft[i], L)
                    
                    if debug:
                        print(f"  [DRAFT_BUILD] draft shape={d.shape}, draft[0:5]={d[0, :min(5, L)].tolist()}")
                    q_draft[i] = d
                    drafts.append(d)
                    sub_seqs.append(seqs[i])

                draft_batch = torch.cat(drafts, dim=0)  # [bg, L]
                if profiler: profiler.stop("jacobi.draft_build")
                if debug:
                    print(f"  [FORWARD] Calling forward with draft_batch.shape={draft_batch.shape}")
                
                logits = self._forward_batched(sub_seqs, draft_batch)
                
                if debug:
                    print(f"  [FORWARD] Returned logits.shape={logits.shape}")
                
                if profiler: profiler.start("jacobi.verify")
                greedy = torch.argmax(logits, dim=-1)
                
                if debug:
                    print(f"  [FORWARD] greedy.shape={greedy.shape}, greedy[0, :5]={greedy[0, :min(5, L-1)].tolist()}")

                acc_lens = self._accept_lengths(draft_batch, greedy)
                
                acc_lens_cpu = acc_lens.cpu().tolist() if not debug else None
                if debug:
                    acc_lens_cpu = acc_lens.tolist()
                    print(f"  [ACCEPT] acc_lens={acc_lens_cpu}")
                if profiler: profiler.stop("jacobi.verify")
                
                if profiler: profiler.start("jacobi.commit")
                for row, i in enumerate(idxs):
                    if debug:
                        print(f"\n  [SEQ_{i}] Processing acceptance for row={row}")
                    
                    seq = sub_seqs[row]
                    
                    draft_gpu = seq.draft_tokens_gpu
                    
                    acc_len = int(acc_lens_cpu[row])
                    acc_len = max(1, min(acc_len, L))
                    
                    if debug:
                        print(f"  [SEQ_{i}] acc_len={acc_len}, L={L}")
                        print(f"  [SEQ_{i}] draft={draft_gpu[:min(10, L)].tolist()}")
                        print(f"  [SEQ_{i}] greedy={greedy[row, :min(10, L-1)].tolist()}")

                    if self.eos_token_id is not None and acc_len > 1:
                        eos_mask = (draft_batch[row, 1:acc_len] == self.eos_token_id)
                        if eos_mask.any():
                            first_eos = int(torch.nonzero(eos_mask, as_tuple=False)[0].item())
                            acc_len = 1 + first_eos + 1
                            eos_reached[i] = True

                    num_spec_accepted = acc_len - 1
                    
                    if debug:
                        print(f"  [SEQ_{i}] Committing {num_spec_accepted} speculative tokens")
                    
                    if num_spec_accepted > 0:
                        accepted_tokens = draft_gpu[1:acc_len].tolist()
                        seq.extend_tokens(accepted_tokens)
                        if self.block_manager is not None:
                            self.block_manager.may_append_batch(seq, num_spec_accepted)
                        accepted[i].extend(accepted_tokens)
                        
                        if debug:
                            print(f"  [SEQ_{i}] Committed: {accepted_tokens}")
                    
                    if acc_len == 1:
                        next_token = greedy[row, 0].item()
                        seq.append_token(next_token)
                        if self.block_manager is not None:
                            self.block_manager.may_append(seq)
                        accepted[i].append(next_token)
                        num_spec_accepted = 1
                        
                        if debug:
                            print(f"  [SEQ_{i}] Autoregressive fallback: greedy[0]={next_token}")
                        
                        if self.eos_token_id is not None and next_token == self.eos_token_id:
                            eos_reached[i] = True
                    
                    new_tokens_this_seq = len(accepted[i]) - prev_accepted_lens[i]
                    tokens_this_iteration += new_tokens_this_seq
                    prev_accepted_lens[i] = len(accepted[i])
                    if profiler: profiler.add_tokens(new_tokens_this_seq)
                    
                    num_tokens_to_trim = L - 1 - num_spec_accepted
                    
                    if num_tokens_to_trim > 0 and self.block_manager is not None:
                        if debug:
                            print(f"  [SEQ_{i}] Trimming KV for {num_tokens_to_trim} tokens")
                        if profiler: profiler.stop("jacobi.commit")
                        if profiler: profiler.start("jacobi.trim")
                        self.block_manager.trim_kv_only_fast(seq, num_tokens_to_trim)
                        if profiler: profiler.stop("jacobi.trim")
                        if profiler: profiler.start("jacobi.commit")
                    
                    seq.clear_draft()
                    
                    if len(seq) != seq.num_cached_tokens:
                        raise RuntimeError(
                            f"Invariant violated: len(token_ids)={len(seq)} != num_cached_tokens={seq.num_cached_tokens}"
                        )
                    
                    if debug:
                        print(f"  [SEQ_{i}] After: len={len(seq)}, num_cached={seq.num_cached_tokens}")
                    
                    if eos_reached[i]:
                        if debug:
                            print(f"  [EOS] Reached EOS for sequence {i}")
                        q_draft[i] = None
                        continue
                    
                    if len(accepted[i]) >= max_tokens[i]:
                        if debug:
                            print(f"  [MAX_TOKENS] Reached max_tokens for sequence {i}")
                        q_draft[i] = None
                        continue
                    
                    if profiler: profiler.stop("jacobi.commit")

                    if profiler: profiler.start("jacobi.next_draft")
                    next_L = block_lens[i]
                    
                    if next_L <= 1:
                        q_draft[i] = None
                        continue
                    
                    new_seed = seq.token_ids[-1]
                    
                    d = torch.empty((1, next_L), dtype=torch.long, device=self.device)
                    d[0, 0] = new_seed
                    
                    if acc_len < L:
                        if acc_len == 1:
                            remaining = greedy[row, 1:]
                            if debug:
                                print(f"  [SEQ_{i}] New draft from greedy[1:] (autoregressive fallback)")
                        else:
                            remaining = greedy[row, acc_len-1:]
                            if debug:
                                print(f"  [SEQ_{i}] New draft from greedy[{acc_len-1}:]")
                        
                        copy_len = min(len(remaining), next_L - 1)
                        if copy_len > 0:
                            d[0, 1:copy_len+1] = remaining[:copy_len]
                    else:
                        d[0, 1] = greedy[row, -1]
                        copy_len = 1
                        
                        if debug:
                            print(f"  [SEQ_{i}] New draft: seed + last greedy")
                    
                    if copy_len < next_L - 1:
                        num_pad = next_L - 1 - copy_len
                        d[0, copy_len+1:] = torch.randint(0, self.vocab_size, (num_pad,), device=self.device)
                    
                    q_draft[i] = d
                    if profiler: profiler.stop("jacobi.next_draft")
                    
                    if debug:
                        print(f"  [SEQ_{i}] new_draft: {d[0, :min(10, next_L)].tolist()}")
            
            self.stats['tokens_per_iteration'].append(tokens_this_iteration)

        total_tokens = sum(len(tokens) for tokens in accepted)
        self.stats['num_chunk_calls'] += 1
        self.stats['num_jacobi_iterations'] += num_iterations_this_call
        self.stats['tokens_accepted'] += total_tokens
        self.stats['tokens_per_call'].append(total_tokens)
        self.stats['iterations_per_call'].append(num_iterations_this_call)

        return accepted
