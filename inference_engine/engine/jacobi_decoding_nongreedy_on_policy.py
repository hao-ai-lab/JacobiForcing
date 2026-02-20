from __future__ import annotations

"""
Non-greedy Jacobi-style decoding using speculative-decoding-style accept/reject
(delta proposal).

OUTPUT FORMAT (IMPORTANT):
- generate_rollout_records_batch(seqs) returns:
    List[ Dict[int, Dict[str, object]] ]

  For each sequence/prompt, you get one dict keyed by block index:
    {
      0: { record for itr_0 },
      1: { record for itr_1 },
      ...
    }

Each block record contains:
  - diffusion_itr_id: "itr_k"
  - prompt_ids: prefix BEFORE this block (trim-left-pad)
  - answer_trajectory_ids: trajectory for THIS block only
      list[list[int]]; each inner list is a fixed-length block vector (len = block_len)
      representing (accepted prefix + current drafted suffix). Reminder: this is BLOCK-LOCAL,
      NOT full prompt+completion.
  - teacher_output_ids: final FULL completion (prompt + completion), stop-truncated;
      we "max-fill" this to final completion for ALL blocks at the end.
  - tokens_per_iter / tokens_per_forward / num_iters / num_forwards:
      tracked cumulatively across the full rollout (up to this block).

Stop handling for Qwen2.5-Coder-Instruct:
- stop tokens default to {151645 (<|im_end|>), 151643 (<|endoftext|>)}.
- stop tokens in the PROMPT are ignored.
- we only stop when a stop token is generated at position >= completion_start_len.
"""

from typing import Callable, Dict, List, Optional, Sequence as PySeq, Tuple, Union
import os
import random

import torch
from torch import Tensor

from inference_engine.engine.sequence import Sequence
from inference_engine.engine.block_manager import BlockManager
from inference_engine.sampling_params import SamplingParams


# ===================
# Profiler helper
# ===================
def _get_profiler():
    if os.environ.get("PROFILE", "0") != "1":
        return None
    from inference_engine.engine.model_runner import get_profiler
    return get_profiler()


# -----------------------------------------------------------------------------
# Forward function types
# -----------------------------------------------------------------------------
LogitsForwardFn = Callable[[Sequence, Tensor], Tensor]
LogitsForwardFnBatch = Callable[[List[Sequence], Tensor], Tensor]


def _safe_getattr(obj, name: str, default):
    return getattr(obj, name, default)


def _infer_data_id(seq: Sequence, fallback: str) -> str:
    for attr in ("data_id", "request_id", "req_id", "uid", "id"):
        v = getattr(seq, attr, None)
        if v is not None:
            return str(v)
    return fallback


def _trim_left_padding(ids: List[int], pad_token_id: Optional[int]) -> List[int]:
    if not ids:
        return []
    if pad_token_id is None:
        return list(ids)
    pad = int(pad_token_id)
    for i, t in enumerate(ids):
        if int(t) != pad:
            return list(ids[i:])
    return []


@torch.inference_mode()
def _softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature is None or temperature <= 0:
        temperature = 1.0
    if float(temperature) != 1.0:
        logits = logits / float(temperature)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def _apply_top_k(probs: Tensor, top_k: Optional[int]) -> Tensor:
    if top_k is None or top_k <= 0 or top_k >= probs.size(-1):
        return probs
    v, idx = torch.topk(probs, k=int(top_k), dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, idx, v)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


@torch.inference_mode()
def _apply_top_p(probs: Tensor, top_p: Optional[float]) -> Tensor:
    if top_p is None:
        return probs
    tp = float(top_p)
    if tp <= 0.0 or tp >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= tp
    keep[..., 0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, filtered)
    return out


@torch.inference_mode()
def _build_target_probs(logits: Tensor, sp: Optional[SamplingParams]) -> Tensor:
    temperature = float(_safe_getattr(sp, "temperature", 1.0))
    top_k = _safe_getattr(sp, "top_k", None)
    top_p = _safe_getattr(sp, "top_p", None)

    probs = _softmax_with_temperature(logits, temperature)
    probs = _apply_top_k(probs, top_k)
    probs = _apply_top_p(probs, top_p)
    return probs


@torch.inference_mode()
def _sample_from_probs(probs: Tensor) -> int:
    # Validate probability distribution
    if torch.isnan(probs).any():
        raise ValueError(f"NaN detected in probability distribution")
    if torch.isinf(probs).any():
        raise ValueError(f"Inf detected in probability distribution")
    if (probs < 0).any():
        raise ValueError(f"Negative values detected in probability distribution")
    
    prob_sum = probs.sum().item()
    if prob_sum <= 0 or prob_sum > 1.01:  # Allow small numerical error
        raise ValueError(f"Invalid probability sum: {prob_sum}")
    
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def _sample_from_probs_not_equal(probs: Tensor, avoid_token: int, max_tries: int = 16) -> int:
    avoid = int(avoid_token)
    for _ in range(max_tries):
        y = _sample_from_probs(probs)
        if int(y) != avoid:
            return int(y)
    p2 = probs.clone()
    p2[avoid] = 0.0
    if float(p2.sum().item()) <= 0.0:
        return avoid
    return int(torch.argmax(p2).item())


def _first_stop_pos_after(ids: List[int], start_idx: int, stop_ids: PySeq[int]) -> Optional[int]:
    stop = set(int(x) for x in stop_ids)
    for i in range(max(0, int(start_idx)), len(ids)):
        if int(ids[i]) in stop:
            return i
    return None


def _truncate_after_stop(ids: List[int], start_idx: int, stop_ids: PySeq[int]) -> List[int]:
    p = _first_stop_pos_after(ids, start_idx, stop_ids)
    if p is None:
        return list(ids)
    return list(ids[: p + 1])


class JacobiDecoderNonGreedyOnPolicy:
    def __init__(
        self,
        block_manager: BlockManager,
        forward_step: Optional[LogitsForwardFn] = None,
        forward_step_batch: Optional[LogitsForwardFnBatch] = None,
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...], set]] = None,
        pad_token_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if forward_step is None and forward_step_batch is None:
            raise ValueError("Provide at least one of forward_step or forward_step_batch.")

        self.block_manager = block_manager
        self.forward_step = forward_step
        self.forward_step_batch = forward_step_batch

        # EOS token handling: should be provided from config
        # Qwen2.5 models: <|endoftext|> = 151643, <|im_end|> = 151645
        if eos_token_id is None:
            raise ValueError("eos_token_id must be provided from model config. Do not use hard-coded values.")
        elif isinstance(eos_token_id, (list, tuple, set)):
            self.stop_token_ids = tuple(int(x) for x in eos_token_id)
        else:
            self.stop_token_ids = (int(eos_token_id),)

        if pad_token_id is None:
            raise ValueError("pad_token_id must be provided from model config. Do not use hard-coded values.")
        self.pad_token_id = int(pad_token_id)
        
        if vocab_size is None:
            raise ValueError("vocab_size must be provided from model config. Do not use hard-coded values.")
        self.vocab_size = int(vocab_size)

        self.debug = os.environ.get("JACOBI_DEBUG", "0") == "1"

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    # ----------------- config helpers -----------------

    def _get_sampling_cfg(self, seq: Sequence) -> Tuple[int, int, int]:
        """
        Return (block_len, max_blocks, remaining_token_budget).

        NOTE: Here "jacobi_max_iterations" is treated as max number of BLOCKS.
        """
        sp: Optional[SamplingParams] = getattr(seq, "sampling_params", None)

        def g(name: str, default):
            return getattr(sp, name, default) if sp is not None else default

        block_len = int(g("jacobi_block_len", 64))
        max_blocks = int(g("jacobi_max_iterations", 128))
        max_tokens_total = int(g("max_tokens", 2048))
        remaining = max(0, max_tokens_total - int(getattr(seq, "num_completion_tokens", 0)))
        return block_len, max_blocks, remaining

    @torch.inference_mode()
    def _forward_single(self, seq: Sequence, draft: Tensor) -> Tensor:
        if self.forward_step is not None:
            return self.forward_step(seq, draft)
        assert self.forward_step_batch is not None
        return self.forward_step_batch([seq], draft)

    # ----------------- draft init -----------------

    def _init_block_draft_from_prompt(self, prompt_ids: List[int], block_len: int) -> List[int]:
        """
        Initialize draft vector by sampling from prompt tokens (with replacement),
        excluding pad tokens. This gives a "prompt-conditioned" initial guess.
        """
        if block_len <= 0:
            return []
        pad = int(self.pad_token_id)
        choices = [int(t) for t in prompt_ids if int(t) != pad]
        if not choices:
            return [random.randrange(self.vocab_size) for _ in range(block_len)]
        return [random.choice(choices) for _ in range(block_len)]

    # ----------------- verify (spec-dec style) -----------------

    @torch.inference_mode()
    def _verify_rejection_sampling(
        self,
        seq: Sequence,
        proposed: List[int],   # length = R
        logits: Tensor,        # [R, vocab]
    ) -> Tuple[List[int], bool]:
        """
        Sequential accept/reject:
          - accept proposed[t] with prob p(proposed[t]|ctx)
          - on first rejection: sample bonus != proposed[t] and STOP

        Returns:
          committed tokens (len>=1 if R>0)
          stop_hit: True if a stop token is committed
        """
        R = len(proposed)
        if R <= 0:
            return [], False

        sp: Optional[SamplingParams] = getattr(seq, "sampling_params", None)
        probs = _build_target_probs(logits, sp)

        committed: List[int] = []
        stop_hit = False

        for t in range(R):
            x = int(proposed[t])
            
            # Validate token index is within bounds
            vocab_dim = probs.size(-1)
            if x < 0 or x >= vocab_dim:
                raise ValueError(f"Token index {x} out of bounds for vocab size {vocab_dim}. "
                               f"This may indicate a mismatch between model vocab and tokenizer vocab.")
            
            p_x = float(probs[t, x].item())
            u = float(torch.rand((), device=probs.device).item())
            accept = (u < p_x)

            if self.debug:
                print(f"[RS] t={t} x={x} p={p_x:.6f} u={u:.6f} accept={accept}")

            if accept:
                committed.append(x)
            else:
                bonus = _sample_from_probs_not_equal(probs[t], avoid_token=x)
                committed.append(int(bonus))
                break

            if int(committed[-1]) in set(self.stop_token_ids):
                stop_hit = True
                break

        # If rejection branch returned a stop token, count it too
        if committed and int(committed[-1]) in set(self.stop_token_ids):
            stop_hit = True

        return committed, stop_hit

    # ----------------- run one block (ALWAYS records trajectory) -----------------

    @torch.inference_mode()
    def _run_one_block(
        self,
        seq: Sequence,
        block_len: int,
        token_budget_remaining: int,
        completion_start_len: int,
        profiler=None,
    ) -> Tuple[List[List[int]], int, int, bool]:
        """
        Run one block and record its trajectory.
        
        Trajectory vectors are always length = block_len. We only generate/verify
        up to gen_len = min(block_len, token_budget_remaining). Positions beyond
        gen_len are padded with pad_token_id. We truncate seq token_ids after stop
        only if stop token was committed and occurs after completion_start_len.
        """
        full_len = int(block_len)
        if full_len <= 0:
            return [], 0, 0, True
        if token_budget_remaining <= 0:
            return [], 0, 0, True

        gen_len = min(full_len, int(token_budget_remaining))
        if gen_len <= 0:
            return [], 0, 0, True

        # Initialize full-length block vector
        prompt_now = list(getattr(seq, "token_ids", []))
        init = self._init_block_draft_from_prompt(prompt_now, gen_len)
        pad = int(self.pad_token_id)

        block_tokens: List[int] = list(init) + [pad] * (full_len - gen_len)

        accepted = 0            # accepted count within [0, gen_len)
        stopped = False
        fwd_used = 0
        appended_total = 0

        trajectory: List[List[int]] = [list(block_tokens)]

        while accepted < gen_len and not stopped:
            remaining = gen_len - accepted
            if remaining <= 0:
                break

            seq_ids = list(getattr(seq, "token_ids", []))
            if not seq_ids:
                seq_ids = [pad]
                seq.token_ids = seq_ids  # type: ignore[attr-defined]

            seed = int(seq_ids[-1])
            proposed = [int(t) for t in block_tokens[accepted:gen_len]]

            draft = torch.empty((1, remaining + 1), dtype=torch.long, device=self.device)
            draft[0, 0] = seed
            draft[0, 1:] = torch.tensor(proposed, dtype=torch.long, device=self.device)
            seq.draft_tokens = draft[0].tolist()

            if profiler:
                profiler.start("jacobi.forward")
            logits = self._forward_single(seq, draft)  # [1, remaining, vocab]
            fwd_used += 1
            if profiler:
                profiler.stop("jacobi.forward")

            if logits.ndim != 3 or int(logits.size(0)) != 1 or int(logits.size(1)) != remaining:
                raise ValueError(f"forward must return logits [1, {remaining}, vocab], got {tuple(logits.shape)}")

            if profiler:
                profiler.start("jacobi.verify")
            committed, stop_hit_local = self._verify_rejection_sampling(seq=seq, proposed=proposed, logits=logits[0])
            if profiler:
                profiler.stop("jacobi.verify")

            if not committed:
                committed = [proposed[0]]
                stop_hit_local = (int(committed[0]) in set(self.stop_token_ids))

            # Commit committed tokens (these are ACCEPTED tokens)
            if profiler:
                profiler.start("jacobi.commit")
            seq.extend_tokens(committed)
            if self.block_manager is not None:
                self.block_manager.may_append_batch(seq, len(committed))
            if profiler:
                profiler.stop("jacobi.commit")

            appended_total += len(committed)

            num_to_trim = remaining - len(committed)
            if num_to_trim > 0 and self.block_manager is not None:
                if profiler:
                    profiler.start("jacobi.trim")
                self.block_manager.trim_kv_only_fast(seq, num_to_trim)
                if profiler:
                    profiler.stop("jacobi.trim")

            seq.clear_draft()

            if len(seq) != seq.num_cached_tokens:
                raise RuntimeError(
                    f"Invariant violated: len(token_ids)={len(seq)} != num_cached_tokens={seq.num_cached_tokens}"
                )

            prev = accepted
            accepted = min(gen_len, accepted + len(committed))
            block_tokens[prev:accepted] = committed[: (accepted - prev)]

            if stop_hit_local:
                full_ids = list(getattr(seq, "token_ids", []))
                truncated = _truncate_after_stop(
                    full_ids,
                    start_idx=completion_start_len,
                    stop_ids=self.stop_token_ids,
                )
                if len(truncated) != len(full_ids):
                    trim_n = len(full_ids) - len(truncated)
                    seq.token_ids = truncated  # type: ignore[attr-defined]
                    if trim_n > 0 and self.block_manager is not None:
                        self.block_manager.trim_kv_only_fast(seq, trim_n)

                stopped = True

                stop_set = set(int(x) for x in self.stop_token_ids)
                stop_local_pos = None
                for j in range(prev, accepted):
                    if int(block_tokens[j]) in stop_set:
                        stop_local_pos = j
                        break
                if stop_local_pos is not None:
                    for k in range(stop_local_pos + 1, full_len):
                        block_tokens[k] = pad
                    accepted = min(accepted, stop_local_pos + 1)

            if not stopped and accepted < gen_len:
                sp: Optional[SamplingParams] = getattr(seq, "sampling_params", None)
                probs_all = _build_target_probs(logits[0], sp)  # [remaining, vocab]
                local_start = len(committed)
                rem2 = gen_len - accepted

                new_drafts: List[int] = []
                for jj in range(rem2):
                    local_idx = local_start + jj
                    if local_idx >= int(probs_all.size(0)):
                        new_drafts.append(random.randrange(self.vocab_size))
                    else:
                        new_drafts.append(_sample_from_probs(probs_all[local_idx]))
                block_tokens[accepted:gen_len] = new_drafts

            if gen_len < full_len:
                for k in range(gen_len, full_len):
                    block_tokens[k] = pad

            trajectory.append(list(block_tokens))

        return trajectory, appended_total, fwd_used, stopped


    # ---------------------------------------------------------------------
    # Rollout records (record trajectory for EVERY block)
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def generate_rollout_records_batch(
        self,
        seqs: List[Sequence],
        n_token_seq_len: Optional[int] = None,
        return_metrics: bool = False,
    ) -> object:
        """
        Returns:
          List[ Dict[int, record_dict] ]  # one per sequence

        For EVERY block k, record_dict includes answer_trajectory_ids for that block.
        teacher_output_ids is max-filled to final completion for ALL blocks.
        """
        if not seqs:
            return ([], []) if return_metrics else []

        profiler = _get_profiler()
        B = len(seqs)

        completion_start_lens = [len(list(getattr(seqs[i], "token_ids", []))) for i in range(B)]
        cfgs = [self._get_sampling_cfg(seqs[i]) for i in range(B)]
        block_lens = [int(n_token_seq_len) if n_token_seq_len is not None else int(c[0]) for c in cfgs]
        max_blocks = [int(c[1]) for c in cfgs]
        budgets = [int(c[2]) for c in cfgs]

        stopped = [False] * B
        num_blocks_done = [0] * B
        num_forwards = [0] * B
        total_generated = [0] * B

        data_ids = [_infer_data_id(seqs[i], fallback=f"data_{i}") for i in range(B)]

        per_seq_out: List[Dict[int, Dict[str, object]]] = [dict() for _ in range(B)]

        # Run rollout, recording EVERY block
        while True:
            active = [
                i for i in range(B)
                if (not stopped[i]) and (num_blocks_done[i] < max_blocks[i]) and (budgets[i] > 0)
            ]
            if not active:
                break

            if profiler:
                profiler.add_iteration()

            for i in active:
                seq = seqs[i]
                k = int(num_blocks_done[i])
                block_len = int(block_lens[i])
                if block_len <= 0:
                    stopped[i] = True
                    continue

                prefix_before = list(getattr(seq, "token_ids", []))
                prompt_ids_trim = _trim_left_padding(prefix_before, self.pad_token_id)

                traj, appended_now, fwd_used, stop_hit = self._run_one_block(
                    seq=seq,
                    block_len=block_len,
                    token_budget_remaining=budgets[i],
                    completion_start_len=completion_start_lens[i],
                    profiler=profiler,
                )

                num_blocks_done[i] += 1
                num_forwards[i] += int(fwd_used)
                total_generated[i] += int(appended_now)
                budgets[i] = max(0, budgets[i] - int(appended_now))
                stopped[i] = bool(stop_hit)

                after_ids = list(getattr(seq, "token_ids", []))
                after_ids = _truncate_after_stop(after_ids, start_idx=completion_start_lens[i], stop_ids=self.stop_token_ids)
                teacher_ids_tmp = _trim_left_padding(after_ids, self.pad_token_id)

                it = max(1, int(num_blocks_done[i]))
                fw = max(1, int(num_forwards[i]))
                tok = float(total_generated[i])

                per_seq_out[i][k] = {
                    "diffusion_itr_id": f"itr_{k}",
                    "data_id": str(data_ids[i]),
                    "prompt_ids": prompt_ids_trim,
                    "answer_trajectory_ids": traj,
                    "teacher_output_ids": teacher_ids_tmp,
                    "tokens_per_iter": tok / float(it),
                    "tokens_per_forward": tok / float(fw),
                    "num_iters": int(num_blocks_done[i]),
                    "num_forwards": int(num_forwards[i]),
                }

        final_teacher_by_id: Dict[str, List[int]] = {}
        for i in range(B):
            full_ids = list(getattr(seqs[i], "token_ids", []))
            full_ids = _truncate_after_stop(full_ids, start_idx=completion_start_lens[i], stop_ids=self.stop_token_ids)
            final_teacher_by_id[str(data_ids[i])] = _trim_left_padding(full_ids, self.pad_token_id)

        for i in range(B):
            did = str(data_ids[i])
            final_teacher = final_teacher_by_id.get(did, [])
            for k in list(per_seq_out[i].keys()):
                per_seq_out[i][k]["teacher_output_ids"] = final_teacher

        if not return_metrics:
            return per_seq_out

        metrics_per_seq: List[Dict[str, float]] = []
        for i in range(B):
            it = float(max(1, int(num_blocks_done[i])))
            fw = float(max(1, int(num_forwards[i])))
            tok = float(total_generated[i])
            metrics_per_seq.append(
                {
                    "total_tokens": tok,
                    "num_iters": float(num_blocks_done[i]),
                    "num_forwards": float(num_forwards[i]),
                    "tokens_per_iter": tok / it,
                    "tokens_per_forward": tok / fw,
                }
            )
        return per_seq_out, metrics_per_seq

    @torch.inference_mode()
    def generate_rollout_records(
        self,
        seq: Sequence,
        n_token_seq_len: Optional[int] = None,
        return_metrics: bool = False,
    ) -> object:
        out = self.generate_rollout_records_batch([seq], n_token_seq_len=n_token_seq_len, return_metrics=return_metrics)
        if not return_metrics:
            return out[0] if out else {}
        records, metrics = out
        return (records[0] if records else {}), (metrics[0] if metrics else {})
