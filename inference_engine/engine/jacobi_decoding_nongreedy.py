from __future__ import annotations

"""
jacobi_decoding_nongreedy.py

A Jacobi-style rollout decoder that uses *non-greedy* verification via
rejection sampling (speculative-decoding style) instead of greedy equality checks.

Key differences from jacobi_decoding.py:
  1) Verification is stochastic: each drafted token is accepted with probability
     p_target(token | context) (a special-case of speculative decoding where the
     proposal distribution q is a delta on the drafted token). On the first
     rejection, we sample a "bonus" token from the *residual* distribution
     proportional to max(0, p_target - q). Under delta-q, that residual reduces
     to p_target conditioned on "token != drafted_token".
  2) Batched generation can optionally return per-sequence efficiency metrics:
        - tokens_per_iter    = generated_tokens / jacobi_iterations
        - tokens_per_forward = generated_tokens / (target forward calls)
"""

from typing import Callable, Dict, List, Optional, Sequence as PySequence, Tuple
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


def _safe_getattr(obj, name: str, default):
    return getattr(obj, name, default)


@torch.inference_mode()
def _softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature is None or temperature <= 0:
        temperature = 1.0
    if temperature != 1.0:
        logits = logits / float(temperature)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def _apply_top_k(probs: Tensor, top_k: Optional[int]) -> Tensor:
    if top_k is None or top_k <= 0 or top_k >= probs.size(-1):
        return probs
    # Keep only top_k probs
    v, idx = torch.topk(probs, k=int(top_k), dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, idx, v)
    # Renormalize
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


@torch.inference_mode()
def _apply_top_p(probs: Tensor, top_p: Optional[float]) -> Tensor:
    """
    Nucleus filtering. Note: requires sorting the vocab dimension; expensive for very large vocab.
    If you care about speed, prefer top_k or leave top_p=None.
    """
    if top_p is None:
        return probs
    tp = float(top_p)
    if tp <= 0.0 or tp >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= tp
    # Ensure at least one token kept
    keep[..., 0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Unsort back
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, filtered)
    return out


@torch.inference_mode()
def _build_target_probs(logits: Tensor, sp: Optional[SamplingParams]) -> Tensor:
    """
    Convert logits -> target sampling distribution p(·).
    Supports temperature/top_k/top_p if present in SamplingParams.
    """
    temperature = float(_safe_getattr(sp, "temperature", 1.0))
    top_k = _safe_getattr(sp, "top_k", None)
    top_p = _safe_getattr(sp, "top_p", None)

    probs = _softmax_with_temperature(logits, temperature)
    probs = _apply_top_k(probs, top_k)
    probs = _apply_top_p(probs, top_p)
    return probs


@torch.inference_mode()
def _sample_from_probs(probs: Tensor) -> int:
    """
    Sample one token id from a 1D prob vector.
    """
    # torch.multinomial expects float probs and sums need not be 1 (but should be non-negative)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def _sample_from_probs_not_equal(probs: Tensor, avoid_token: int, max_tries: int = 16) -> int:
    """
    Sample y ~ probs conditioned on y != avoid_token.
    Implemented by rejection sampling: sample from probs until y != avoid_token.
    This is efficient in practice because rejection only happens when avoid_token has high mass,
    and then the original acceptance event is very likely, so we rarely enter this path.
    """
    for _ in range(max_tries):
        y = _sample_from_probs(probs)
        if y != int(avoid_token):
            return y
    # Fallback: deterministic best alternative
    probs2 = probs.clone()
    probs2[int(avoid_token)] = 0.0
    if probs2.sum().item() <= 0:
        # If all mass on avoid_token (shouldn't happen with a real softmax), just return avoid_token.
        return int(avoid_token)
    return int(torch.argmax(probs2).item())


class JacobiDecoderNonGreedy:
    """
    Jacobi-style verifier with speculative-decoding-style accept/reject.

    Compared to jacobi_decoding.JacobiDecoder:
      - The "verify" step is stochastic rather than greedy-equality.
      - This is a special case of speculative decoding where the proposal q is a delta
        on the drafted token at each position, so accept prob is p(x), and residual
        sampling upon rejection becomes sampling from p conditioned on x' != x.
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

        self.debug = os.environ.get("JACOBI_DEBUG", "0") == "1"

        # Global statistics (optional, useful for profiling)
        self.stats: Dict[str, object] = {
            "num_chunk_calls": 0,
            "num_jacobi_iterations": 0,
            "tokens_accepted": 0,
            "tokens_per_call": [],
            "tokens_per_iteration": [],
            "iterations_per_call": [],
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

    # ----------------- draft helpers -----------------

    def _init_draft(self, seq: Sequence, length: int) -> Tensor:
        """Initialize draft with seed + random speculative tokens."""
        if length <= 0:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)

        token_ids = getattr(seq, "token_ids", None) or []
        if token_ids:
            seed = token_ids[-1]
        else:
            seed = (
                self.pad_token_id
                if self.pad_token_id is not None
                else (self.eos_token_id if self.eos_token_id is not None else 0)
            )

        d = torch.empty((1, length), dtype=torch.long, device=self.device)
        d[0, 0] = seed
        if length > 1:
            d[0, 1:] = torch.randint(0, self.vocab_size, (length - 1,), device=self.device)
        return d

    def _resize_or_init_draft(self, seq: Sequence, draft: Optional[Tensor], L: int) -> Tensor:
        """Ensure draft is [1, L] on self.device and has correct seed."""
        if L <= 0:
            return torch.empty((1, 0), dtype=torch.long, device=self.device)

        if draft is None or draft.numel() == 0:
            return self._init_draft(seq, L)

        draft = draft.to(self.device)
        if draft.dim() != 2 or draft.size(0) != 1:
            raise ValueError(f"draft must be [1, L], got {tuple(draft.shape)}")

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

        if logits.ndim != 3 or logits.size(0) != B or logits.size(1) != (L - 1):
            raise ValueError(
                f"forward must return logits [B, L-1, vocab], expected [{B}, {L-1}, *], got {tuple(logits.shape)}"
            )
        return logits

    # ----------------- non-greedy verify (spec-dec style) -----------------

    @torch.inference_mode()
    def _verify_block_rejection_sampling(
        self,
        seq: Sequence,
        draft_row: Tensor,   # [L]
        logits_row: Tensor,  # [L-1, vocab]
    ) -> Tuple[List[int], int, bool]:
        """
        Verify draft_row[1:] under target probs derived from logits_row.

        Returns:
          committed_tokens: tokens to append to the sequence (>=1 unless L==1)
          num_keep_for_kv: how many speculative *positions* we treat as kept in KV bookkeeping
                           (used in trim logic, mirrors jacobi_decoding.py's behavior).
          eos_reached: whether EOS is generated/accepted in this commit.
        """
        L = int(draft_row.size(0))
        if L <= 1:
            return [], 0, False

        sp: Optional[SamplingParams] = getattr(seq, "sampling_params", None)
        probs = _build_target_probs(logits_row, sp)  # [L-1, vocab]

        committed: List[int] = []
        eos_reached = False

        # Sequential accept/reject along the block
        for t in range(L - 1):
            proposed = int(draft_row[t + 1].item())
            p_x = float(probs[t, proposed].item())
            u = float(torch.rand((), device=probs.device).item())

            if self.debug:
                print(f"[RS] t={t} proposed={proposed} p={p_x:.6f} u={u:.6f} accept={u < p_x}")

            if u < p_x:
                # Accept proposed token
                committed.append(proposed)
                if self.eos_token_id is not None and proposed == self.eos_token_id:
                    eos_reached = True
                    break
                continue

            # Reject at position t: sample "bonus" token from residual.
            # With delta proposal q=δ_proposed, residual ∝ max(0, p - q) => p conditioned on token != proposed.
            bonus = _sample_from_probs_not_equal(probs[t], avoid_token=proposed)
            committed.append(int(bonus))
            if self.eos_token_id is not None and int(bonus) == self.eos_token_id:
                eos_reached = True
            break

        # KV bookkeeping:
        # We mimic jacobi_decoding.py's behavior by treating *all* committed tokens as if
        # their KV positions are kept (even if a "bonus" replaces the proposed token).
        num_keep_for_kv = len(committed)
        return committed, num_keep_for_kv, eos_reached

    # ----------------- public APIs -----------------

    @torch.inference_mode()
    def generate_chunk(self, seq: Sequence, return_metrics: bool = False) -> object:
        """
        Single-seq generation.

        If return_metrics=False:
            returns List[int] accepted tokens.
        If return_metrics=True:
            returns (tokens, metrics_dict) where metrics includes:
              - tokens_per_iter
              - tokens_per_forward
              - num_iters
              - num_forwards
        """
        tokens, metrics = self._generate_single(seq)
        return (tokens, metrics) if return_metrics else tokens

    @torch.inference_mode()
    def _generate_single(self, seq: Sequence) -> Tuple[List[int], Dict[str, float]]:
        block_len, max_iters = self._get_sampling_cfg(seq)
        sp = getattr(seq, "sampling_params", None)
        max_tokens = getattr(sp, "max_tokens", 2048) if sp else 2048
        max_tokens -= seq.num_completion_tokens

        accepted: List[int] = []
        q_draft: Optional[Tensor] = None
        eos_reached = False
        iters = 0
        forwards = 0

        profiler = _get_profiler()

        while not eos_reached and len(accepted) < max_tokens and iters < max_iters:
            iters += 1
            if profiler: profiler.add_iteration()

            L = int(block_len)
            if L <= 1:
                break

            if profiler: profiler.start("jacobi.draft_build")
            q_draft = self._resize_or_init_draft(seq, q_draft, L)
            seq.draft_tokens = q_draft[0].tolist()
            if profiler: profiler.stop("jacobi.draft_build")

            # Forward target
            if profiler: profiler.start("jacobi.forward")
            logits = self.forward_step(seq, q_draft) if self.forward_step else self.forward_step_batch([seq], q_draft)
            forwards += 1
            if profiler: profiler.stop("jacobi.forward")

            # Verify with RS
            if profiler: profiler.start("jacobi.verify")
            committed, num_keep_for_kv, eos_hit = self._verify_block_rejection_sampling(
                seq, q_draft[0], logits[0]
            )
            eos_reached = eos_reached or eos_hit
            if profiler: profiler.stop("jacobi.verify")

            # Commit
            if profiler: profiler.start("jacobi.commit")
            if committed:
                seq.extend_tokens(committed)
                if self.block_manager is not None:
                    self.block_manager.may_append_batch(seq, len(committed))
                accepted.extend(committed)
            if profiler: profiler.stop("jacobi.commit")

            if profiler: profiler.start("jacobi.trim")
            num_to_trim = (L - 1) - int(num_keep_for_kv)
            if num_to_trim > 0 and self.block_manager is not None:
                self.block_manager.trim_kv_only_fast(seq, num_to_trim)
            if profiler: profiler.stop("jacobi.trim")

            seq.clear_draft()

            if len(seq) != seq.num_cached_tokens:
                raise RuntimeError(
                    f"Invariant violated: len(token_ids)={len(seq)} != num_cached_tokens={seq.num_cached_tokens}"
                )

            if profiler: profiler.add_tokens(len(committed))

            if eos_reached or len(accepted) >= max_tokens:
                break

            # Build next draft cheaply (Jacobi refinement): seed + greedy fill from current logits
            if profiler: profiler.start("jacobi.next_draft")
            greedy = torch.argmax(logits[0], dim=-1)  # [L-1]
            new_seed = seq.token_ids[-1]
            d = torch.empty((1, L), dtype=torch.long, device=self.device)
            d[0, 0] = new_seed

            # "acc_len" in the greedy file counted seed + accepted speculative tokens
            acc_len = 1 + len(committed)
            if acc_len < L:
                remaining = greedy[acc_len - 1 :] if acc_len > 1 else greedy[1:]
                copy_len = min(int(remaining.numel()), L - 1)
                if copy_len > 0:
                    d[0, 1 : copy_len + 1] = remaining[:copy_len]
            else:
                d[0, 1] = greedy[-1].item()
                copy_len = 1

            if copy_len < L - 1:
                d[0, copy_len + 1 :] = torch.randint(
                    0, self.vocab_size, (L - 1 - copy_len,), device=self.device
                )
            q_draft = d
            if profiler: profiler.stop("jacobi.next_draft")

        # Update global stats
        self.stats["num_chunk_calls"] = int(self.stats["num_chunk_calls"]) + 1
        self.stats["num_jacobi_iterations"] = int(self.stats["num_jacobi_iterations"]) + iters
        self.stats["tokens_accepted"] = int(self.stats["tokens_accepted"]) + len(accepted)
        self.stats["tokens_per_call"].append(len(accepted))
        self.stats["iterations_per_call"].append(iters)

        metrics = {
            "tokens_per_iter": (len(accepted) / iters) if iters > 0 else 0.0,
            "tokens_per_forward": (len(accepted) / forwards) if forwards > 0 else 0.0,
            "num_iters": float(iters),
            "num_forwards": float(forwards),
        }
        return accepted, metrics

    @torch.inference_mode()
    def generate_chunk_batch(
        self,
        seqs: List[Sequence],
        return_metrics: bool = False,
    ) -> object:
        """
        Batched generation.

        If return_metrics=False (default): returns List[List[int]] tokens per sequence.
        If return_metrics=True: returns (tokens, metrics_per_sequence)

        metrics_per_sequence[i] contains:
          - tokens_per_iter
          - tokens_per_forward
          - num_iters
          - num_forwards
          - total_tokens
        """
        if not seqs:
            return ([], []) if return_metrics else []

        if len(seqs) == 1:
            toks, m = self._generate_single(seqs[0])
            return ([toks], [m]) if return_metrics else [toks]

        B = len(seqs)
        accepted: List[List[int]] = [[] for _ in range(B)]
        q_draft: List[Optional[Tensor]] = [None for _ in range(B)]
        eos_reached = [False for _ in range(B)]
        iters = [0 for _ in range(B)]
        forwards = [0 for _ in range(B)]

        cfg = [self._get_sampling_cfg(s) for s in seqs]
        block_lens = [c[0] for c in cfg]
        max_iters = [c[1] for c in cfg]

        max_tokens = []
        for seq in seqs:
            sp = getattr(seq, "sampling_params", None)
            if sp is not None:
                max_tok = getattr(sp, "max_tokens", 2048)
                remaining = max_tok - seq.num_completion_tokens
                max_tokens.append(max(0, remaining))
            else:
                max_tokens.append(2048)

        num_iterations_this_call = 0
        profiler = _get_profiler()

        while True:
            active = [
                i for i in range(B)
                if (not eos_reached[i])
                and (len(accepted[i]) < max_tokens[i])
                and (iters[i] < max_iters[i])
            ]
            if not active:
                break

            # Group active sequences by L to avoid padding assumptions
            groups: Dict[int, List[int]] = {}
            for i in active:
                L = int(block_lens[i])
                if L > 1:
                    groups.setdefault(L, []).append(i)
            if not groups:
                break

            num_iterations_this_call += 1
            tokens_this_iteration = 0
            if profiler: profiler.add_iteration()

            for L, idxs in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True):
                if profiler: profiler.start("jacobi.draft_build")
                drafts = []
                sub_seqs = []
                for i in idxs:
                    iters[i] += 1
                    d = self._resize_or_init_draft(seqs[i], q_draft[i], L)
                    q_draft[i] = d
                    seqs[i].draft_tokens = d[0].tolist()
                    drafts.append(d)
                    sub_seqs.append(seqs[i])
                draft_batch = torch.cat(drafts, dim=0)  # [bg, L]
                if profiler: profiler.stop("jacobi.draft_build")

                # Forward (target)
                if profiler: profiler.start("jacobi.forward")
                logits = self._forward_batched(sub_seqs, draft_batch)
                if profiler: profiler.stop("jacobi.forward")

                # Count forwards per involved sequence
                for row, i in enumerate(idxs):
                    forwards[i] += 1

                # Verify + commit per sequence
                if profiler: profiler.start("jacobi.verify_commit")
                for row, i in enumerate(idxs):
                    if eos_reached[i] or len(accepted[i]) >= max_tokens[i]:
                        continue

                    seq = sub_seqs[row]
                    committed, num_keep_for_kv, eos_hit = self._verify_block_rejection_sampling(
                        seq, draft_batch[row], logits[row]
                    )
                    eos_reached[i] = eos_reached[i] or eos_hit

                    # Commit
                    if committed:
                        seq.extend_tokens(committed)
                        if self.block_manager is not None:
                            self.block_manager.may_append_batch(seq, len(committed))
                        accepted[i].extend(committed)
                        tokens_this_iteration += len(committed)
                        if profiler: profiler.add_tokens(len(committed))

                    num_to_trim = (L - 1) - int(num_keep_for_kv)
                    if num_to_trim > 0 and self.block_manager is not None:
                        if profiler: profiler.start("jacobi.trim")
                        self.block_manager.trim_kv_only_fast(seq, num_to_trim)
                        if profiler: profiler.stop("jacobi.trim")

                    seq.clear_draft()

                    if len(seq) != seq.num_cached_tokens:
                        raise RuntimeError(
                            f"Invariant violated: len(token_ids)={len(seq)} != num_cached_tokens={seq.num_cached_tokens}"
                        )

                    # Next draft: seed + greedy fill from current logits
                    if eos_reached[i] or len(accepted[i]) >= max_tokens[i]:
                        q_draft[i] = None
                        continue

                    greedy = torch.argmax(logits[row], dim=-1)  # [L-1]
                    new_seed = seq.token_ids[-1]
                    d2 = torch.empty((1, L), dtype=torch.long, device=self.device)
                    d2[0, 0] = new_seed

                    acc_len = 1 + len(committed)
                    if acc_len < L:
                        remaining = greedy[acc_len - 1 :] if acc_len > 1 else greedy[1:]
                        copy_len = min(int(remaining.numel()), L - 1)
                        if copy_len > 0:
                            d2[0, 1 : copy_len + 1] = remaining[:copy_len]
                    else:
                        d2[0, 1] = greedy[-1].item()
                        copy_len = 1

                    if copy_len < L - 1:
                        d2[0, copy_len + 1 :] = torch.randint(
                            0, self.vocab_size, (L - 1 - copy_len,), device=self.device
                        )
                    q_draft[i] = d2
                if profiler: profiler.stop("jacobi.verify_commit")

            self.stats["tokens_per_iteration"].append(tokens_this_iteration)

        total_tokens = sum(len(x) for x in accepted)
        self.stats["num_chunk_calls"] = int(self.stats["num_chunk_calls"]) + 1
        self.stats["num_jacobi_iterations"] = int(self.stats["num_jacobi_iterations"]) + num_iterations_this_call
        self.stats["tokens_accepted"] = int(self.stats["tokens_accepted"]) + total_tokens
        self.stats["tokens_per_call"].append(total_tokens)
        self.stats["iterations_per_call"].append(num_iterations_this_call)

        if not return_metrics:
            return accepted

        metrics_per_seq: List[Dict[str, float]] = []
        for i in range(B):
            tok = float(len(accepted[i]))
            it = float(iters[i])
            fw = float(forwards[i])
            metrics_per_seq.append(
                {
                    "total_tokens": tok,
                    "num_iters": it,
                    "num_forwards": fw,
                    "tokens_per_iter": (tok / it) if it > 0 else 0.0,
                    "tokens_per_forward": (tok / fw) if fw > 0 else 0.0,
                }
            )
        return accepted, metrics_per_seq
