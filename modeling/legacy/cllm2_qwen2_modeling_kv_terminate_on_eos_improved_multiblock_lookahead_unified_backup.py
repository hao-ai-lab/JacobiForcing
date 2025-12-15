from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
from typing import Dict, Optional, Sequence, List, Tuple
from collections import deque
import itertools

# logits processors
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache


# --- utilities: cache trimming for rejected tails ---
def _delete_false_key_value(self: DynamicCache, num_of_false_tokens: int) -> None:
    if num_of_false_tokens <= 0 or not self.key_cache:
        return
    trim = -num_of_false_tokens
    for layer_idx in range(len(self.key_cache)):
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        # Slice once; avoid extra .contiguous() unless actually needed later.
        self.key_cache[layer_idx] = k[..., :trim, :]
        self.value_cache[layer_idx] = v[..., :trim, :]

DynamicCache.delete_false_key_value = _delete_false_key_value


# --- utilities: lookahead candidate building from n-gram pool ---
def _build_candidates(n_gram_pool: deque, next_token: torch.Tensor, out: torch.Tensor, nearest: bool=False):
    """
    n_gram_pool: deque of 1 x L_i tensors (draft tails you appended earlier)
    next_token:  1 x 1 tensor
    out:         1 x L_out (current draft: [next_token, greedy_tail...])
    Returns: list of 1D tensors length L_out (no batch dim)
    """
    candidates = []
    token_val = next_token.item()
    L_out = out.size(1)

    # iterate reversed but skip the very last pushed element
    for seq in itertools.islice(reversed(n_gram_pool), 1, None):
        seq_flat = seq[0]  # [L]
        # First match position (if any)
        matches = (seq_flat == token_val).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            pos = int(matches[0])
            new_cand = seq_flat[pos:].unsqueeze(0)  # [1, L_new]
            L_new = new_cand.size(1)

            if L_new > L_out:
                new_cand = new_cand[:, :L_out]
            elif L_new < L_out:
                # pad with the remainder from 'out' to preserve behavior
                pad = out[:, L_new:L_out]
                new_cand = torch.cat([new_cand, pad], dim=1)
            candidates.append(new_cand[0])

            if nearest:
                break
    return candidates


def _resize_dynamic_cache_batch(cache: DynamicCache, new_B: int) -> DynamicCache:
    """
    Make cache.key_cache/value_cache batch dimension == new_B.
    - Grow from 1 -> new_B via expand (fast).
    - Grow from k>1 -> new_B via repeat (tile rows).
    - Shrink from k>new_B -> new_B via slicing [:new_B].
    """
    if not cache.key_cache:
        return cache

    cur_B = cache.key_cache[0].size(0)
    if cur_B == new_B:
        return cache

    grow = new_B > cur_B
    tile_needed = grow and cur_B > 1

    if grow and cur_B == 1:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i]   = cache.key_cache[i].expand(new_B, -1, -1, -1)
            cache.value_cache[i] = cache.value_cache[i].expand(new_B, -1, -1, -1)
        return cache

    if tile_needed:
        reps = (new_B + cur_B - 1) // cur_B
        for i in range(len(cache.key_cache)):
            k = cache.key_cache[i].repeat(reps, 1, 1, 1)[:new_B]
            v = cache.value_cache[i].repeat(reps, 1, 1, 1)[:new_B]
            cache.key_cache[i], cache.value_cache[i] = k, v
        return cache

    # shrink
    for i in range(len(cache.key_cache)):
        cache.key_cache[i]   = cache.key_cache[i][:new_B]
        cache.value_cache[i] = cache.value_cache[i][:new_B]
    return cache


def _ensure_batch_like_ra(x: torch.Tensor, B_target: int, *, device, dtype) -> torch.Tensor:
    """
    Make 'x' 2D with batch size == B_target.
    Rules:
    - If x is empty (numel==0), return an empty [B_target, 0].
    - If x.shape[0] == 1 and B_target>1, expand along batch.
    - If x.shape[0] == B_target, return as-is.
    - Otherwise (e.g., x has batch!=1 and !=B_target), repeat rows to cover B_target then slice.
    """
    if x.numel() == 0:
        return torch.empty((B_target, 0), device=device, dtype=dtype)

    if x.dim() != 2:
        raise ValueError(f"_ensure_batch_like_ra expects [B, L], got {tuple(x.shape)}")

    B_cur, L = x.shape
    if B_cur == B_target:
        return x
    if B_cur == 1 and B_target > 1:
        return x.expand(B_target, -1)

    reps = (B_target + B_cur - 1) // B_cur
    x_tiled = x.repeat(reps, 1)  # [reps*B_cur, L]
    return x_tiled[:B_target, :]


@torch.inference_mode()
def jacobi_forward_greedy_multiblock(
    self,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len: int = 64,
    # multi-block controls
    K: int = 2,                 # max number of concurrent blocks (1 RA + K-1 pseudo)
    r: float = 0.8,             # spawn threshold as a fraction of n_token_seq_len
    # lookahead-related
    lookahead_start_ratio = 0.0,
    n_gram_pool_size = 8,
    # sampling knobs (kept for parity; we run greedy inside)
    temperature: float = 1.0,
    top_p: float = 0.85,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    lenience: float = 1.0,
    accept_threshold: float = 0.99,
    tokenizer = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    max_iteration_count: int = 128,
):
    """
    Prefill: identical to before.
    Generation: refactored to a single-`out` assembly loop (RA + pseudo blocks),
    with micro-optimizations that preserve behavior.
    """
    device = input_ids.device
    dtype  = input_ids.dtype

    # local bindings (avoid repeated attribute lookups in the loop)
    model = self.model
    layers = model.layers
    num_layers = model.config.num_hidden_layers
    has_sliding = getattr(model, "has_sliding_layers", False)
    embed_tokens = model.embed_tokens
    rotary_emb = model.rotary_emb
    norm = model.norm
    lm_head = self.lm_head
    cfg = self.config

    eos_enabled = eos_token_id is not None
    eos_id = eos_token_id

    def dbg_print(*_args, **_kwargs):
        # Flip this to print diagnostics without changing performance by default
        # print(*_args, **_kwargs)
        return

    # =========================
    # ===== PREFILL PHASE =====
    # =========================
    if prefill_phase:
        if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=device
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": cfg,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if has_sliding:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = rotary_emb(hidden_states, position_ids)
        for decoder_layer in layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states).float()

        prefill_drafted_n_gram = torch.argmax(logits[:, -n_token_seq_len-1:-1, :], dim=-1)
        first_correct_token = prefill_drafted_n_gram[0]

        if (past_key_values is not None) and (n_token_seq_len > 0):
            past_key_values.delete_false_key_value(n_token_seq_len)

        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0

    # ============================
    # ===== GENERATION PHASE =====
    # ============================
    assert past_key_values is not None, "past_key_values must be provided during generation."

    # --- n-gram pool to reuse rejected tails across iterations ---
    n_gram_pool = deque(maxlen=n_gram_pool_size)

    if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
        attention_mask = torch.ones_like(input_ids)

    # Block state
    out_acc: List[torch.Tensor] = [torch.empty((1, 0), device=device, dtype=dtype)]
    q_draft: List[torch.Tensor] = [input_ids.clone()]  # RA draft tail
    need_reverify: List[bool]   = [False]
    total_acc: List[int]        = [0]
    num_blocks = 1
    active_blocks = 1
    RA = 0

    last_next_token = out_acc[RA][:, :1]  # may be empty initially

    prompt_len = past_key_values.get_seq_length()
    spawn_threshold = math.ceil(r * n_token_seq_len)

    def committed_len(cur_RA: int) -> int:
        committed = prompt_len
        for b in range(num_blocks):
            if b != cur_RA and (not need_reverify[b]):
                committed += out_acc[b].shape[1]
        committed += out_acc[cur_RA].shape[1]
        return committed

    def _kv_trim_to(final_committed: int):
        kv_len = past_key_values.get_seq_length()
        td = kv_len - final_committed
        if td > 0:
            past_key_values.delete_false_key_value(td)

    def build_out_and_spans() -> Tuple[torch.Tensor, List[Tuple[int,int,int]]]:
        """
        Assemble 'out' using RA's batch size as the canonical B.
        out = [RA_draft] + Σ pseudo blocks: [acc_prefix] + [draft_tail]
        Spans: (block_id, start_idx, length), where logits slice is logits[:, start_idx-1 : start_idx-1+length, :].
        All non-RA pieces are broadcast to RA's batch size.
        """
        B_ra = q_draft[RA].size(0)

        pieces: List[torch.Tensor] = []
        spans: List[Tuple[int,int,int]] = []

        # Lookback intentionally omitted as in your version; keep cursor offset=1
        cursor = 1

        # 1) RA draft
        L_ra = int(q_draft[RA].size(1))
        if L_ra > 0:
            pieces.append(q_draft[RA])
            spans.append((RA, cursor, L_ra))
            cursor += L_ra

        # 2) Pseudo blocks
        for b in range(num_blocks):
            if b == RA or not need_reverify[b]:
                continue

            L_acc = int(out_acc[b].size(1))
            if L_acc > 0:
                acc_b = _ensure_batch_like_ra(out_acc[b], B_ra, device=device, dtype=dtype)
                pieces.append(acc_b)
                cursor += L_acc

            L_tail = int(q_draft[b].size(1))
            if L_tail > 0:
                draft_b = _ensure_batch_like_ra(q_draft[b], B_ra, device=device, dtype=dtype)
                pieces.append(draft_b)
                spans.append((b, cursor, L_tail))
                cursor += L_tail

        if not pieces:
            out = torch.empty((B_ra, 0), device=device, dtype=dtype)
        else:
            out = torch.cat(pieces, dim=-1)
        return out, spans

    iters = 0
    while iters < max_iteration_count:
        iters += 1

        out, spans = build_out_and_spans()
        if out.numel() == 0:
            break

        B_out = out.size(0)
        _resize_dynamic_cache_batch(past_key_values, B_out)

        # ========= single forward pass over `out` ========= #
        inputs_embeds = embed_tokens(out)
        out_attention_mask = torch.ones_like(out, device=device)

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=device
        )
        pos_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := out_attention_mask, dict):
            mask_kwargs = {
                "config": cfg,
                "input_embeds": inputs_embeds,
                "attention_mask": out_attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if has_sliding:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        pos_emb = rotary_emb(hidden_states, pos_ids)
        # Slice exactly num_hidden_layers for safety
        for decoder_layer in layers[:num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=pos_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=pos_emb,
            )[0]
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states).float()
        # ==================================================

        # PROCESS BLOCKS
        for (b, start, L) in spans:
            # logits for b's draft positions (use lookback-aligned window)
            block_logits = logits[:, start-1 : start-1+L, :]    # [B, L, vocab]
            greedy = torch.argmax(block_logits, dim=-1)         # [B, L]
            draft = q_draft[b]                                  # [B, L_eff]

            # Longest exact-match prefix length
            mismatch = (draft[:, 1:] != greedy[:, :-1])
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1

            if b == RA:
                best_idx = int(torch.argmax(accepted))
            else:
                best_idx = 0
            acc_len_raw = int(accepted[best_idx])

            # narrow to best batch row
            draft = draft[best_idx:best_idx+1, :]
            block_logits = block_logits[best_idx:best_idx+1, :, :]
            greedy = greedy[best_idx:best_idx+1, :]

            # shrink KV to that row only (matches original behavior)
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i]  = past_key_values.key_cache[i][best_idx:best_idx+1]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][best_idx:best_idx+1]

            L_eff = draft.shape[1]
            if L_eff == 0:
                continue

            acc_len = acc_len_raw
            # ----- EOS handling: cap acceptance at first EOS in the accepted region -----
            eos_reached = False
            if eos_enabled and b == RA and acc_len > 0:
                eos_mask = (draft[:, :acc_len] == eos_id)
                if eos_mask.any():
                    first_eos_rel = int(torch.nonzero(eos_mask, as_tuple=False)[0, 1])
                    acc_len = first_eos_rel + 1
                    eos_reached = True

            has_rejected = (acc_len < L_eff)

            # Accept the verified prefix (possibly EOS-capped)
            if acc_len > 0:
                out_acc[b] = torch.cat((out_acc[b], draft[:, :acc_len]), dim=-1)
                total_acc[b] += acc_len

            # If EOS was reached on the RA block, finalize immediately
            if eos_reached and b == RA:
                ret = torch.empty((1, 0), device=device, dtype=dtype)
                for bb in range(num_blocks):
                    if (bb != RA) and (not need_reverify[bb]) and out_acc[bb].numel() > 0:
                        ret = torch.cat((ret, out_acc[bb]), dim=-1)
                ret = torch.cat((ret, out_acc[RA]), dim=-1)

                final_committed_len = prompt_len + ret.shape[1]
                _kv_trim_to(final_committed_len)

                next_token = draft[:, acc_len-1:acc_len]
                return past_key_values, next_token, ret, iters

            # If not EOS, manage reject/accept tails
            if has_rejected:
                nxt_idx = max(acc_len - 1, 0)
                nxt = greedy[:, nxt_idx:nxt_idx+1]

                # Refresh tail with greedy continuation starting at mismatch
                q_draft[b] = torch.cat([nxt, greedy[:, acc_len:-1]], dim=-1)

                # update n-gram pool only for RA
                if b == RA:
                    n_gram_pool.append(greedy[:, acc_len:-1])

                # spawn extra candidates based on n_gram_pool
                if b == RA and (total_acc[b] / n_token_seq_len >= lookahead_start_ratio):
                    cands = _build_candidates(n_gram_pool, nxt, q_draft[b], nearest=False)
                    if len(cands) > 1:
                        cands_t = torch.stack(cands, dim=0)                 # [K, L]
                        q_draft[b] = torch.cat([q_draft[b], cands_t], dim=0)  # [1+K, L]
                        _resize_dynamic_cache_batch(past_key_values, q_draft[b].shape[0])
            else:
                # all-accept: tail empty; seed next step with last greedy
                q_draft[b] = torch.empty((1, 0), device=device, dtype=dtype)
                nxt = greedy[:, -1:]

            if b == RA:
                last_next_token = nxt

            # EOS on the next sampled token → return
            if eos_enabled and b == RA and last_next_token.item() == eos_id:
                out_acc[b] = torch.cat((out_acc[b], last_next_token), dim=-1)
                ret = torch.empty((1, 0), device=device, dtype=dtype)
                for bb in range(num_blocks):
                    if (bb != RA) and (not need_reverify[bb]) and out_acc[bb].numel() > 0:
                        ret = torch.cat((ret, out_acc[bb]), dim=-1)
                ret = torch.cat((ret, out_acc[RA]), dim=-1)

                final_committed_len = prompt_len + ret.shape[1]
                _kv_trim_to(final_committed_len)

                return past_key_values, last_next_token, ret, iters

        # maintain KV to exactly the committed length
        _kv_trim_to(committed_len(RA))

        # Possibly spawn a new pseudo block from RA's current draft if progressed enough
        newest_id = num_blocks - 1
        if total_acc[newest_id] >= spawn_threshold and active_blocks < K:
            if pad_token_id is None:
                raise ValueError("pad_token_id must be provided when spawning pseudo-active blocks.")

            dbg_print(f"======New block added (total={num_blocks+1}) at global_iter={iters}======")

            ra_tail = q_draft[RA]
            L_ra = int(ra_tail.shape[1])
            pad_len = max(0, n_token_seq_len - L_ra)
            if pad_len:
                pad_tail = torch.full(
                    (ra_tail.shape[0], pad_len), fill_value=pad_token_id, device=device, dtype=dtype
                )
                q_new = torch.cat([ra_tail.clone(), pad_tail], dim=-1)
            else:
                q_new = ra_tail.clone()

            q_draft.append(q_new)
            out_acc.append(torch.empty((1, 0), device=device, dtype=dtype))
            total_acc.append(0)
            need_reverify.append(True)
            num_blocks += 1
            active_blocks += 1

        # If RA finished, promote earliest pseudo block with progress to RA
        if total_acc[RA] >= n_token_seq_len:
            for b in range(num_blocks):
                dbg_print(f"checking if block: {b} is a good candidate for block switching...")
                if need_reverify[b] and total_acc[b] > 0:
                    dbg_print(f"============= SWITCHING REAL ACTIVE BLOCK TO {b} =============")

                    acc_pref = out_acc[b]   # [1, a]
                    tail    = q_draft[b]    # [1, t]
                    q_full  = torch.cat([acc_pref, tail], dim=-1) if acc_pref.numel() > 0 else tail.clone()
                    # Must be exact n_token_seq_len (assert preserved)
                    assert q_full.size(1) == n_token_seq_len, (
                        f"draft size mismatch: draft at {q_full.size(1)} vs. n_token_seq_len {n_token_seq_len}"
                    )

                    out_acc[b]   = torch.empty((1, 0), device=device, dtype=dtype)
                    total_acc[b] = 0
                    q_draft[b]   = torch.cat([last_next_token, q_full[:, 1:]], dim=-1)

                    need_reverify[b] = False
                    RA = b

                    _kv_trim_to(committed_len(RA))
                    active_blocks -= 1
                    break

                # no more blocks need re-verify: decrement active blocks count, put RA into sleep
                active_blocks -= 1

        # early stop if every block has accepted full length
        if all(total_acc[b] >= n_token_seq_len for b in range(num_blocks)):
            dbg_print("EARLY STOPPING SINCE ALL BLOCKS HAVE BEEN ACCEPTED TO FULL LENGTHS.")
            break

    # ============= Finalize return =============
    dbg_print("!!! MAX ITERATION COUNT REACHED, COLLECTING FINAL OUTPUT !!!")
    ret = torch.empty((1, 0), device=device, dtype=dtype)
    for b in range(num_blocks):
        if (b != RA) and (not need_reverify[b]) and out_acc[b].numel() > 0:
            ret = torch.cat((ret, out_acc[b]), dim=-1)
    if out_acc[RA].numel() > 0:
        ret = torch.cat((ret, out_acc[RA]), dim=-1)

    final_committed_len = prompt_len + ret.shape[1]
    _kv_trim_to(final_committed_len)

    next_token = last_next_token if last_next_token is not None else ret[:, -1:]
    return past_key_values, next_token, ret, iters
