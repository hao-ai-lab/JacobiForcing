from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import math
from typing import Optional, List, Tuple

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

def _safe_delete_false_key_value(self, num_of_false_tokens: int) -> None:
    """Trim the newest `num_of_false_tokens` KV entries from all layers (no-op if <=0)."""
    if not num_of_false_tokens or num_of_false_tokens <= 0:
        return
    for layer_idx in range(len(self.key_cache)):
        n = self.key_cache[layer_idx].size(-2)
        k = min(num_of_false_tokens, n)
        if k > 0:
            self.key_cache[layer_idx]  = self.key_cache[layer_idx][..., :-k, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-k, :]

DynamicCache.delete_false_key_value = _safe_delete_false_key_value

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
    # sampling knobs (kept for parity; we run greedy inside)
    temperature: float = 1.0,
    top_p: float = 0.9,
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
    eos_enabled = eos_id is not None

    if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
        attention_mask = torch.ones_like(input_ids)

    # We assume `input_ids` encodes: [ first_correct_token | draft_(n-1) ]
    # Initialize blocks: one Real-Active (RA) and optionally spawn pseudo blocks later.
    # We'll keep per-block accepted prefix and current draft tail, but drive each step from a single `out`.

    # Block state
    #out_acc: List[torch.Tensor] = [input_ids[:, :1].clone()]                # RA accepted prefix
    out_acc: List[torch.Tensor] = [torch.empty(
                                        (1, 0), device=device, dtype=input_ids.dtype
                                    )]
    #q_draft: List[torch.Tensor] = [input_ids[:, 1:].clone()]
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

    def build_out_and_spans() -> Tuple[torch.Tensor, List[Tuple[int,int,int]]]:
        """
        Assemble the single 'out' tensor:
          out = [global_lookback] + [RA_draft] + for each pseudo block: [acc_prefix] + [draft_tail]
        'spans' records, for each block, where its draft logits live in 'out' (start index and length).
        """
        pieces = []
        cursor = 1
        
        spans: List[Tuple[int,int,int]] = []

        # RA block: logits slice corresponds to full RA draft tail
        L_ra = int(q_draft[RA].shape[1])
        if L_ra > 0:
            pieces.append(q_draft[RA])
            spans.append((RA, cursor, L_ra))
            cursor += L_ra

        # Pseudo blocks: inject accepted prefix (no logits) then draft tail (collect logits)
        for b in range(num_blocks):
            if b == RA or not need_reverify[b]:
                continue
            L_acc = int(out_acc[b].shape[1])
            L_tail = int(q_draft[b].shape[1])

            if L_acc > 0:
                pieces.append(out_acc[b])
                cursor += L_acc
            if L_tail > 0:
                pieces.append(q_draft[b])
                spans.append((b, cursor, L_tail))
                cursor += L_tail

        out = torch.cat(pieces, dim=-1)
        #print(f"pieces size: {len(pieces)}")
        #print(f"first lookback token: {last_next_token.size()}")
        #print(f"RA draft shape: {q_draft[RA].size()}")
        #print(f"out shape: {out.size()}")

        return out, spans

    iters = 0
    while iters < max_iteration_count:
        iters += 1

        out, spans = build_out_and_spans()
        if out.numel() == 0:
            break

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
            # logits for b's draft positions (use lookback-aligned window)
            block_logits = logits[:, start-1 : start-1+L, :]   # [1, L, vocab]
            greedy = torch.argmax(block_logits, dim=-1)         # [1, L]
            draft = q_draft[b]                                  # [1, L_eff]

            L_eff = draft.shape[1]
            if L_eff == 0:
                continue

            # Longest exact-match prefix length
            mismatch = (draft[:, 1:] != greedy[:, :-1])
            acc_len_raw = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0]) + 1
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