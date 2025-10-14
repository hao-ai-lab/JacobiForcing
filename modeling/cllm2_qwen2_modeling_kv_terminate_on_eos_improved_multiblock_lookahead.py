from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
from typing import Dict, Optional, Sequence, List, Tuple
from collections import deque

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

from collections import deque
from typing import Tuple
import torch

# --- utilities: cache trimming for rejected tails ---
def _delete_false_key_value(self: DynamicCache, num_of_false_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_of_false_tokens <= 0:
        return
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx]  = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
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

    # iterate reversed (except very last element we just created)
    for seq in reversed(list(n_gram_pool)[:-1]):
        seq_flat = seq[0]  # [L]
        matches = (seq_flat == token_val).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            pos = matches[0].item()
            new_cand = seq_flat[pos:].unsqueeze(0)  # [1, L_new]
            L_new = new_cand.size(1)

            if L_new > L_out:
                new_cand = new_cand[:, :L_out]
            elif L_new < L_out:
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
    if len(cache.key_cache) == 0:
        return cache

    cur_B = cache.key_cache[0].size(0)
    if cur_B == new_B:
        return cache

    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        v = cache.value_cache[i]

        if new_B > cur_B:
            if cur_B == 1:
                # broadcast
                cache.key_cache[i]   = k.expand(new_B, -1, -1, -1).contiguous()
                cache.value_cache[i] = v.expand(new_B, -1, -1, -1).contiguous()
            else:
                # tile rows
                reps = (new_B + cur_B - 1) // cur_B
                k_rep = k.repeat(reps, 1, 1, 1)[:new_B]
                v_rep = v.repeat(reps, 1, 1, 1)[:new_B]
                cache.key_cache[i]   = k_rep.contiguous()
                cache.value_cache[i] = v_rep.contiguous()
        else:
            # shrink by slicing
            cache.key_cache[i]   = k[:new_B].contiguous()
            cache.value_cache[i] = v[:new_B].contiguous()
    return cache


def _expand_dynamic_cache_to_batch(cache: DynamicCache, new_B: int) -> DynamicCache:
    # Repeat KV along batch for speculative candidates
    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        v = cache.value_cache[i]
        cache.key_cache[i]  = k.expand(new_B, -1, -1, -1).contiguous()
        cache.value_cache[i] = v.expand(new_B, -1, -1, -1).contiguous()
    return cache


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
    """
    # =========================
    # ===== PREFILL PHASE =====
    # =========================
    if prefill_phase:
        if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()

        # Argmax over the draft window
        prefill_drafted_n_gram = torch.argmax(logits[:, -n_token_seq_len-1:-1, :], dim=-1)
        first_correct_token = prefill_drafted_n_gram[0]

        if (past_key_values is not None) and (n_token_seq_len > 0):
            past_key_values.delete_false_key_value(n_token_seq_len)

        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0

    # ============================
    # ===== GENERATION PHASE =====
    # ============================
    assert past_key_values is not None, "past_key_values must be provided during generation."
    device = input_ids.device
    eos_id = eos_token_id
    
    eos_enabled = eos_token_id is not None
    device = input_ids.device

    # --- n-gram pool to reuse rejected tails across iterations ---
    n_gram_pool = deque(maxlen=n_gram_pool_size)

    if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
        attention_mask = torch.ones_like(input_ids)

    # We assume `input_ids` encodes: [ first_correct_token | draft_(n-1) ]
    # Initialize blocks: one Real-Active (RA) and optionally spawn pseudo blocks later.
    # We'll keep per-block accepted prefix and current draft tail, but drive each step from a single `out`.

    # Block state
    #out_acc: List[torch.Tensor] = [input_ids[:, :1].clone()]
    out_acc: List[torch.Tensor] = [torch.empty(
                                        (1, 0), device=device, dtype=input_ids.dtype
                                    )]
    q_draft: List[torch.Tensor] = [input_ids.clone()]                       # RA draft tail
    need_reverify: List[bool]   = [False]                                    # RA is verified
    total_acc: List[int]        = [0]                                        # tokens accepted in each block
    num_blocks = 1
    active_blocks = 1
    RA = 0
    
    last_next_token = out_acc[RA][:, :1]

    prompt_len = past_key_values.get_seq_length()
    spawn_threshold = math.ceil(r * n_token_seq_len)

    def committed_len(cur_RA: int) -> int:
        # committed = prompt + all verified non-RA + current RA accepted
        committed = prompt_len
        for b in range(num_blocks):
            if b != cur_RA and (not need_reverify[b]):
                committed += out_acc[b].shape[1]
        committed += out_acc[cur_RA].shape[1]
        return committed

    def current_lookback_token() -> torch.Tensor:
        if out_acc[RA].numel() > 0:
            return out_acc[RA][:, -1:]
        elif RA == 0:
            # first iteration
            return q_draft[RA][:, :0]
        
        # Else if it's empty, search verified blocks from newest to oldest
        for b in reversed(range(num_blocks)):
            if b == RA:
                continue
            if (not need_reverify[b]) and out_acc[b].numel() > 0:
                return out_acc[b][:, -1:]
        raise RuntimeError("No committed token available for lookback.")

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
            return x.expand(B_target, -1).contiguous()

        # Fallback: tile rows to reach B_target, then slice
        reps = (B_target + B_cur - 1) // B_cur
        x_tiled = x.repeat(reps, 1)              # [reps*B_cur, L]
        return x_tiled[:B_target, :].contiguous()

    def _kv_batch_size(cache: DynamicCache) -> Optional[int]:
        return cache.key_cache[0].size(0) if len(cache.key_cache) > 0 else None

    def build_out_and_spans() -> Tuple[torch.Tensor, List[Tuple[int,int,int]]]:
        """
        Assemble 'out' using RA's batch size as the canonical B.
        out = [lookback] + [RA_draft] + Σ pseudo blocks: [acc_prefix] + [draft_tail]
        Spans: (block_id, start_idx, length), where logits slice is logits[:, start_idx-1 : start_idx-1+length, :].
        All non-RA pieces are **broadcast** to RA's batch size.
        """
        # Canonical batch size = RA's draft batch
        B_ra = q_draft[RA].size(0)
        device = input_ids.device
        dtype  = input_ids.dtype

        pieces: List[torch.Tensor] = []
        spans: List[Tuple[int,int,int]] = []

        # 1) Optional lookback (broadcast to RA batch if present)
        #lookback = current_lookback_token()  # [1,1] or [1,0]
        #if lookback.numel() > 0 and lookback.size(1) == 1:
        #    lookback = _ensure_batch_like_ra(lookback, B_ra, device=device, dtype=dtype)  # [B_ra,1]
        #    pieces.append(lookback)
        #    cursor = 1
        #else:
        #    cursor = 0
        cursor = 1
        
        # 2) RA draft (already at B_ra)
        L_ra = int(q_draft[RA].size(1))
        if L_ra > 0:
            pieces.append(q_draft[RA])                     # [B_ra, L_ra]
            spans.append((RA, cursor, L_ra))
            cursor += L_ra

        # 3) Pseudo blocks (broadcast to B_ra)
        for b in range(num_blocks):
            if b == RA or not need_reverify[b]:
                continue

            # accepted prefix (no logits)
            L_acc = int(out_acc[b].size(1))
            if L_acc > 0:
                acc_b = _ensure_batch_like_ra(out_acc[b], B_ra, device=device, dtype=dtype)  # [B_ra, L_acc]
                pieces.append(acc_b)
                cursor += L_acc

            # draft tail (collect logits)
            L_tail = int(q_draft[b].size(1))
            if L_tail > 0:
                draft_b = _ensure_batch_like_ra(q_draft[b], B_ra, device=device, dtype=dtype)  # [B_ra, L_tail]
                pieces.append(draft_b)
                spans.append((b, cursor, L_tail))
                cursor += L_tail

        # 4) Concatenate (or return empty [B_ra,0])
        if len(pieces) == 0:
            out = torch.empty((B_ra, 0), device=device, dtype=dtype)
        else:
            out = torch.cat(pieces, dim=-1)  # [B_ra, total_len]
        
        #print(f"out shape: {out.size()}")

        return out, spans


    iters = 0
    while iters < max_iteration_count:
        iters += 1

        out, spans = build_out_and_spans()
        if out.numel() == 0:
            break

        B_out = out.size(0)
        _resize_dynamic_cache_batch(past_key_values, B_out)
        
        #kvB = _kv_batch_size(past_key_values) 
        #if kvB is not None and kvB != B_out:
        #    _expand_dynamic_cache_to_batch(past_key_values, B_out)

        # ========= single forward pass over `out` ========= #
        inputs_embeds = self.model.embed_tokens(out)
        out_attention_mask = torch.ones_like(out, device=device)

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        pos_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := out_attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": out_attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        pos_emb = self.model.rotary_emb(hidden_states, pos_ids)
        for decoder_layer in self.model.layers[: self.model.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=pos_ids,
                past_key_value=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=pos_emb,
            )[0]
        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        # ==================================================

        # PROCESS BLOCKS: (pseudo) accept longest prefix that matches draft
        for (b, start, L) in spans:
            #if total_acc[b] >= n_token_seq_len:
            #    continue
            #print(f"b: {b}; start: {start}; L: {L}")
            
            # logits for b's draft positions (use lookback-aligned window)
            block_logits = logits[:, start-1 : start-1+L, :]    # [B, L, vocab]
            #print(f"block logits shape: {block_logits.size()}")
            
            greedy = torch.argmax(block_logits, dim=-1)         # [B, L]
            draft = q_draft[b]                                  # [B, L_eff]

            # Longest exact-match prefix length
            #print(f"draft shape: {draft.size()}")
            #print(f"greedy shape: {greedy.size()}")
            mismatch = (draft[:, 1:] != greedy[:, :-1])
            
            #if b == RA:
            
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
            if b == RA:
                # pick best candidate among rows (if B > 1)
                best_idx = int(torch.argmax(accepted).item())
            else:
                best_idx = 0
            acc_len_raw = int(accepted[best_idx])
            
            #print(f"selecting best idx: {best_idx}")
            # narrow to best batch row
            draft = draft[best_idx:best_idx+1, :].contiguous()
            #print(f"draft size: {draft.size()}")
            block_logits = block_logits[best_idx:best_idx+1, :, :].contiguous()
            greedy = greedy[best_idx:best_idx+1, :].contiguous()
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i]  = past_key_values.key_cache[i][best_idx:best_idx+1].contiguous()
                past_key_values.value_cache[i] = past_key_values.value_cache[i][best_idx:best_idx+1].contiguous()
                              
            #else:
            #    acc_len_raw = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0]) + 1
            
            L_eff = draft.shape[1]
            if L_eff == 0:
                continue
            
            acc_len = acc_len_raw
            # ----- EOS handling: cap acceptance at first EOS in the accepted region -----
            eos_reached = False
            if eos_enabled and b == RA and acc_len > 0:
                # Look only inside the currently accepted slice
                eos_mask = (draft[:, :acc_len] == eos_id)
                if eos_mask.any():
                    # position of the first EOS relative to the accepted prefix
                    first_eos_rel = int(torch.nonzero(eos_mask, as_tuple=False)[0, 1])
                    acc_len = first_eos_rel + 1
                    eos_reached = True

            has_rejected = (acc_len < L_eff)

            # Accept the prefix we verified (possibly EOS-capped)
            if acc_len > 0:
                out_acc[b] = torch.cat((out_acc[b], draft[:, :acc_len]), dim=-1)
                total_acc[b] += acc_len

            # If EOS was reached on the RA block, finalize immediately (ignore any greedy tail)
            if eos_reached and b == RA:
                # Build final return sequence in block order: verified non-RA + RA accepted
                ret = torch.empty((1, 0), device=device, dtype=input_ids.dtype)
                for bb in range(num_blocks):
                    if (bb != RA) and (not need_reverify[bb]) and out_acc[bb].numel() > 0:
                        ret = torch.cat((ret, out_acc[bb]), dim=-1)
                ret = torch.cat((ret, out_acc[RA]), dim=-1)

                # Tighten KV to committed length (prompt + ret)
                final_committed_len = prompt_len + ret.shape[1]
                td = past_key_values.get_seq_length() - final_committed_len
                if td > 0:
                    past_key_values.delete_false_key_value(td)

                # next token after EOS is conventionally EOS itself (or keep last_next_token fallback)
                next_token = draft[:, acc_len-1:acc_len]
                return past_key_values, next_token, ret, iters

            # If we didn't hit EOS, proceed with normal reject/accept tail management
            if has_rejected:
                # next token to seed the next step: first mismatch prediction.
                # guard acc_len==0 (use the very first greedy token)
                nxt_idx = max(acc_len - 1, 0)
                nxt = greedy[:, nxt_idx:nxt_idx+1]
            
                # RA refreshes its tail with the greedy continuation starting at the mismatch
                # (prepend nxt so the next step has a "lookback" token)
                q_draft[b] = torch.cat([nxt, greedy[:, acc_len:-1]], dim=-1)
                
                # n-gram pool only serves for the real active block
                if b == RA:
                    n_gram_pool.append(greedy[:, acc_len:-1])
                
                ### ADDED: Construct more drafts candidates based on n_gram_pool ###
                # spawn extra candidates after a fraction of block is accepted
                if b == RA and (total_acc[b] / n_token_seq_len >= lookahead_start_ratio):
                    cands = _build_candidates(n_gram_pool, nxt, q_draft[b], nearest=False)
                    if len(cands) > 1:
                        # stack: (K, L) -> batch with original at row 0
                        cands_t = torch.stack(cands, dim=0)                       # [K, L]
                        q_draft[b] = torch.cat([q_draft[b], cands_t], dim=0)      # [1+K, L]
                        # repeat KV across candidates (speculative batch)
                        
                        _resize_dynamic_cache_batch(past_key_values, q_draft[b].shape[0])
                        
                        #_expand_dynamic_cache_to_batch(past_key_values, q_draft[b].shape[0])
                ### ADDED: Construct more drafts candidates based on n_gram_pool ###

            else:
                # all-accept: tail is empty; seed next step with last greedy for continuity
                q_draft[b] = torch.empty((1, 0), device=device, dtype=input_ids.dtype)
                nxt = greedy[:, -1:]
            
            if b == RA:
                last_next_token = nxt
                
            # EOS on the next sampled token, return
            if eos_enabled and b == RA and last_next_token.item() == eos_id:
                out_acc[b] = torch.cat((out_acc[b], last_next_token), dim=-1)
                # accept EOS and stop
                ret = torch.empty((1, 0), device=device, dtype=input_ids.dtype)
                for bb in range(num_blocks):
                    if (bb != RA) and (not need_reverify[bb]) and out_acc[bb].numel() > 0:
                        ret = torch.cat((ret, out_acc[bb]), dim=-1)
                ret = torch.cat((ret, out_acc[RA]), dim=-1)
                
                # Tighten KV to committed length (prompt + ret)
                final_committed_len = prompt_len + ret.shape[1]
                td = past_key_values.get_seq_length() - final_committed_len
                if td > 0:
                    past_key_values.delete_false_key_value(td)

                return past_key_values, last_next_token, ret, iters
        
        # maintain KV to exactly the committed length (prompt + verified + RA accepted)
        cur_committed = committed_len(RA)
        kv_length = past_key_values.get_seq_length()
        #print(f"current commited length: {cur_committed}")
        #print(f"prompt length: {prompt_len}")
        #print(f"kv length: {kv_length}")
            
        td = kv_length - cur_committed
        #print(f"kv trimmed length: {td}")
        #print("----------")
        past_key_values.delete_false_key_value(td)
        
        # Possibly spawn a new pseudo block from RA's current draft if progressed enough
        newest_id = num_blocks - 1
        if total_acc[newest_id] >= spawn_threshold and active_blocks < K:
            if pad_token_id is None:
                raise ValueError("pad_token_id must be provided when spawning pseudo-active blocks.")

            print(f"======New block added (total={num_blocks+1}) at global_iter={iters}======")
            
            # new block draft = clone of current RA draft, padded to n_token_seq_len
            ra_tail = q_draft[RA]                      # [1, L_ra]
            L_ra = int(ra_tail.shape[1])
            pad_len = max(0, n_token_seq_len - L_ra)
            pad_tail = torch.full(
                (ra_tail.shape[0], pad_len),
                fill_value=pad_token_id,
                device=device,
                dtype=input_ids.dtype,
            )
            q_new = torch.cat([ra_tail.clone(), pad_tail], dim=-1)  # [1, n_token_seq_len]

            q_draft.append(q_new)
            out_acc.append(torch.empty((1, 0), device=device, dtype=input_ids.dtype))
            total_acc.append(0)
            need_reverify.append(True)
            num_blocks += 1
            active_blocks += 1

        # If RA finished, promote the earliest pseudo block with progress to RA
        if total_acc[RA] >= n_token_seq_len:
            for b in range(num_blocks):
                print(f"checking if block: {b} is a good candidate for block switching...")
                if need_reverify[b] and total_acc[b] > 0:
                    print(f"============= SWITCHING REAL ACTIVE BLOCK TO {b} =============")
                    
                    # rebuild a full-length draft for the new RA from (acc_prefix + tail)
                    acc_pref = out_acc[b]   # [1, a]
                    tail    = q_draft[b]    # [1, t]
                    q_full  = torch.cat([acc_pref, tail], dim=-1) if acc_pref.numel() > 0 else tail.clone()
                    # Ensure exact n_token_seq_len
                    assert q_full.size(1) == n_token_seq_len, f"draft size mismatch: draft at {q_full.size(1)} vs. n_token_seq_len {n_token_seq_len}"
                    
                    #q_full[:, 1] = last_next_token
                    
                    #if q_full.size(1) != n_token_seq_len:
                    #    # Trim or pad (should rarely happen; keep robust)
                    #    if q_full.size(1) > n_token_seq_len:
                    #        q_full = q_full[:, :n_token_seq_len]
                    #    else:
                    #        pad_len = n_token_seq_len - q_full.size(1)
                    #        q_full = torch.cat(
                    #            [q_full, torch.full((1, pad_len), pad_token_id, device=device, dtype=input_ids.dtype)],
                    #            dim=-1,
                    #        )

                    # Reset RA acceptance boundary for re-verification
                    
                    # TODO: verify this is correct — matching with additional token sampling
                    #out_acc[b] = last_next_token
                    #q_draft[b] = q_full[:, 1:].clone()
                    #total_acc[b] = 1
                    
                    #out_acc[b]   = last_next_token.clone() 
                    #q_draft[b]   = q_full[:, 1:].clone()
                    #total_acc[b] = 1
                    
                    out_acc[b]   = torch.empty((1, 0), device=device, dtype=input_ids.dtype)
                    total_acc[b] = 0
                    q_draft[b] = torch.cat([last_next_token, q_full[:, 1:]], dim=-1)
                    
                    need_reverify[b] = False
                    RA = b

                    # tighten KV again to new committed length
                    cur_committed = committed_len(RA)
                    kv_length = past_key_values.get_seq_length()
                    #print(f"kv length: {kv_length}")
                    td = kv_length - cur_committed
                    past_key_values.delete_false_key_value(td)
                    
                    #print(f"kv trimmed length: {td}")
                    
                    #print(f"==========================")
                    
                    active_blocks -= 1
                    break
            
                # if no more blocks need re-verify, decrement active blocks count, put the RA into sleep
                active_blocks -= 1

        # early stop if every block has accepted full length
        if all(total_acc[b] >= n_token_seq_len for b in range(num_blocks)):
            print("EARLY STOPPING SINCE ALL BLOCKS HAVE BEEN ACCEPTED TO FULL LENGTHS.")
            break

    # ============= Finalize return =============
    print("!!! MAX ITERATION COUNT REACHED, COLLECTING FINAL OUTPUT !!!")
    ret = torch.empty((1, 0), device=device, dtype=input_ids.dtype)
    for b in range(num_blocks):
        if (b != RA) and (not need_reverify[b]) and out_acc[b].numel() > 0:
            ret = torch.cat((ret, out_acc[b]), dim=-1)
    if out_acc[RA].numel() > 0:
        ret = torch.cat((ret, out_acc[RA]), dim=-1)

    # remove redundancy
    final_committed_len = prompt_len + ret.shape[1]
    td = past_key_values.get_seq_length() - final_committed_len
    if td > 0:
        past_key_values.delete_false_key_value(td)

    # If no explicit last_next_token was set, fallback to the final token of the committed output
    next_token = last_next_token if last_next_token is not None else ret[:, -1:]
    return past_key_values, next_token, ret, iters