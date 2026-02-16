#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HF/Qwen2 jacobi_forward_nongreedy — aligned with reference "proper rejection sampling".

We interpret each iteration as speculative decoding with a delta proposal:
  - draft = [seed] + speculative tokens
  - verify speculative tokens sequentially with acceptance prob p_target(proposed)
  - on first rejection: sample bonus ~ p_target(· | token != proposed), COMMIT bonus immediately, stop iteration

KV correctness trick (no extra 1-token forward for bonus):
  - we NEVER keep KV for the last committed token. We pop it out of KV and return it as next seed.
  - this avoids the KV mismatch at the rejection position (bonus differs from proposed), because
    the bonus token's KV is computed at the start of the next iteration when it is re-forwarded as seed.

This matches the reference behavior at the token level (accept/reject + residual sampling) while keeping
KV consistent without an extra forward.
The correctness of the accept+residual rule is exactly the delta-proposal special case of speculative sampling. :contentReference[oaicite:1]{index=1}
"""

from typing import Optional, Tuple, List
import torch

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache


# ---------------------------
# DynamicCache helper
# ---------------------------
def delete_false_key_value(self, num_of_false_tokens) -> None:
    num = int(num_of_false_tokens)
    if num <= 0:
        return
    # Defensive: don't underflow
    cur = int(self.get_seq_length())
    num = min(num, cur)
    if num <= 0:
        return
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num, :]


DynamicCache.delete_false_key_value = delete_false_key_value


# ---------------------------
# Sampling utils
# ---------------------------
@torch.inference_mode()
def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or float(temperature) <= 0:
        temperature = 1.0
    if float(temperature) != 1.0:
        logits = logits / float(temperature)
    return torch.softmax(logits, dim=-1)


@torch.inference_mode()
def _apply_top_k(probs: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or int(top_k) <= 0 or int(top_k) >= probs.size(-1):
        return probs
    v, idx = torch.topk(probs, k=int(top_k), dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, idx, v)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def _apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
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
def _build_target_probs(
    logits: torch.Tensor,  # [..., vocab]
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    probs = _softmax_with_temperature(logits, temperature)
    probs = _apply_top_k(probs, top_k)
    probs = _apply_top_p(probs, top_p)
    return probs


@torch.inference_mode()
def _sample_from_probs(probs_1d: torch.Tensor) -> int:
    return int(torch.multinomial(probs_1d, num_samples=1).item())


@torch.inference_mode()
def _sample_from_probs_not_equal(probs_1d: torch.Tensor, avoid_token: int, max_tries: int = 16) -> int:
    for _ in range(max_tries):
        y = _sample_from_probs(probs_1d)
        if y != int(avoid_token):
            return y
    probs2 = probs_1d.clone()
    probs2[int(avoid_token)] = 0.0
    if probs2.sum().item() <= 0:
        return int(avoid_token)
    return int(torch.argmax(probs2).item())


# ---------------------------
# Main function
# ---------------------------
@torch.inference_mode()
def jacobi_forward_nongreedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,        # draft = [seed] + speculative
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,             # KV for prefix excluding seed (draft[0])
    use_cache: Optional[bool] = True,
    prefill_phase: Optional[bool] = False,               # unused in aligned inference
    n_token_seq_len: int = 64,                           # max tokens to return this call
    jacobi_max_iterations: int = 128,                    # max internal iters per call
    temperature: float = 1.0,
    top_p: Optional[float] = 0.3,
    top_k: Optional[int] = 50,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
):
    if input_ids is None:
        raise ValueError("jacobi_forward_nongreedy: input_ids must be provided.")
    if input_ids.size(0) != 1:
        raise ValueError(f"bsz=1 only, got {input_ids.size(0)}")

    if prefill_phase:
        raise ValueError(
            "prefill_phase=True is not used in the aligned setup. "
            "Do prompt prefill outside (cache prompt[:-1], seed=prompt[-1])."
        )

    if past_key_values is None:
        raise ValueError("past_key_values must be provided (KV for prefix excluding seed).")

    device = input_ids.device
    eos_id = eos_token_id
    eos_enabled = eos_id is not None

    def _mk1(tid: int) -> torch.LongTensor:
        return torch.tensor([[int(tid)]], device=device, dtype=input_ids.dtype)

    # Draft length L = [seed] + (L-1) speculative
    out = input_ids  # [1, L]
    L = int(out.shape[1])
    if L < 2:
        # nothing to verify
        return past_key_values, _mk1(int(out[0, 0].item())), out[:, :0], 0

    # We'll accumulate up to n_token_seq_len tokens to append this call
    accepted = torch.empty((1, n_token_seq_len), dtype=out.dtype, device=device)
    total_accepted = 0
    itr = 0

    # -----------------------
    # Forward helper: appends tok to past_key_values and returns logits for each position in tok
    # -----------------------
    def _forward_tokens(tok: torch.LongTensor) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(tok)
        attn = torch.ones_like(tok, device=device)

        past_seen = past_key_values.get_seq_length()
        cache_pos = torch.arange(past_seen, past_seen + tok.shape[1], device=inputs_embeds.device)
        pos_ids = cache_pos.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attn, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attn,
                "cache_position": cache_pos,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hs = inputs_embeds
        pos_emb = self.model.rotary_emb(hs, pos_ids)

        for layer in self.model.layers[: self.model.config.num_hidden_layers]:
            hs = layer(
                hs,
                attention_mask=causal_mask_mapping[layer.attention_type],
                position_ids=pos_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )[0]

        hs = self.model.norm(hs)
        return self.lm_head(hs).float()  # [1, len(tok), vocab]

    # -----------------------
    # Jacobi iterations
    # -----------------------
    while total_accepted < n_token_seq_len and itr < jacobi_max_iterations:
        itr += 1

        # forward current draft; this APPENDS L tokens into past_key_values
        logits_full = _forward_tokens(out)               # [1, L, vocab]
        logits_verify = logits_full[0, : L - 1, :]       # [L-1, vocab] predicts out[1:]

        probs_mat = _build_target_probs(logits_verify, float(temperature), top_k=top_k, top_p=top_p)

        # Verify speculative tokens sequentially
        committed: List[int] = []
        for t in range(L - 1):
            proposed = int(out[0, t + 1].item())
            p_x = float(probs_mat[t, proposed].item())
            u = float(torch.rand((), device=device).item())

            if u < p_x:
                committed.append(proposed)
                if eos_enabled and proposed == eos_id:
                    break
            else:
                bonus = _sample_from_probs_not_equal(probs_mat[t], avoid_token=proposed)
                committed.append(int(bonus))
                break

        # Truncate at EOS if it appears in committed
        if eos_enabled and eos_id in committed:
            committed = committed[: committed.index(eos_id) + 1]

        if not committed:
            # Defensive: if somehow nothing was committed, delete the whole draft we appended and stop.
            past_key_values.delete_false_key_value(L)
            break

        # Cap by remaining token budget for this call
        remain = n_token_seq_len - total_accepted
        take = committed[:remain]

        # Write accepted tokens
        k = len(take)
        accepted[:, total_accepted : total_accepted + k] = torch.tensor(
            take, dtype=accepted.dtype, device=device
        ).view(1, -1)
        total_accepted += k

        # Next seed is the LAST token we returned (already part of accepted)
        next_seed = int(take[-1])

        # KV trimming:
        # We want KV to end at token *before* next_seed, so next_seed will be re-forwarded next iter.
        # After forward we appended L tokens [seed] + speculative. Keep only:
        #   old seed + (take[:-1])  => length = len(take)
        # Therefore delete (L - len(take)) from the end.
        delete_from_end = L - len(take)
        if delete_from_end > 0:
            past_key_values.delete_false_key_value(delete_from_end)

        # Stop if EOS was committed
        if eos_enabled and next_seed == eos_id:
            return past_key_values, _mk1(next_seed), accepted[:, :total_accepted], itr

        if total_accepted >= n_token_seq_len:
            return past_key_values, _mk1(next_seed), accepted[:, :total_accepted], itr

        # Build next draft: [seed=next_seed] + greedy fill from current logits
        greedy_next = torch.argmax(logits_verify, dim=-1)  # [L-1]
        d = torch.empty((1, L), dtype=out.dtype, device=device)
        d[0, 0] = next_seed

        acc_len = 1 + len(take)   # seed + newly committed tokens (seed is at position acc_len-1 in old draft)

        if acc_len < L:
            # We want predictions starting from token after next_seed => index (acc_len-1)
            start = acc_len - 1
            remaining = greedy_next[start:] if start < greedy_next.numel() else greedy_next[-1:]
            copy_len = min(int(remaining.numel()), L - 1)
            if copy_len > 0:
                d[0, 1 : copy_len + 1] = remaining[:copy_len].to(out.dtype)
            if copy_len < (L - 1):
                d[0, copy_len + 1 :] = torch.randint(
                    0, int(probs_mat.size(-1)), (L - 1 - copy_len,), device=device, dtype=out.dtype
                )
        else:
            # Considered fully accepted; just continue with last greedy
            d[0, 1] = int(greedy_next[-1].item())
            if L > 2:
                d[0, 2:] = torch.randint(
                    0, int(probs_mat.size(-1)), (L - 2,), device=device, dtype=out.dtype
                )

        out = d

    # If we exit by max iters, return what we have (seed is last accepted token)
    if total_accepted > 0:
        last_seed = int(accepted[0, total_accepted - 1].item())
        return past_key_values, _mk1(last_seed), accepted[:, :total_accepted], itr

    # No progress
    seed0 = int(out[0, 0].item())
    return past_key_values, _mk1(seed0), accepted[:, :0], itr
