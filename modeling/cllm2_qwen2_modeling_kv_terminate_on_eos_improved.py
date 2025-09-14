from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
from typing import Dict, Optional, Sequence, List, Tuple

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

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
            
DynamicCache.delete_false_key_value = delete_false_key_value

@torch.inference_mode()
def jacobi_forward_greedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    prefill_draft_token_ids: Optional[torch.LongTensor] = None,
    n_token_seq_len = 64,
    temperature = 1.0,
    top_p = 0.9, 
    top_k = None,
    repetition_penalty = None, 
    lenience = 1.,
    accept_threshold = 0.99,
    tokenizer = None,
    eos_token_id: Optional[int] = None,
    ):

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    # Resolve EOS id
    eos_id = eos_token_id

    eos_enabled = eos_id is not None
    if not eos_enabled:
        print("!!! WARNING: EOS handling disabled since eos_token_id is None !!!")

    # ---- LogitsProcessor: greedy only
    from transformers.generation.logits_process import LogitsProcessorList
    logits_processors = LogitsProcessorList()

    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        
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
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                # "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    
        hidden_states = inputs_embeds
    
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
    
        for decoder_layer in self.model.layers[: self.model.config.num_hidden_layers]:
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

        scores = logits_processors(input_ids, logits.squeeze(0)).unsqueeze(0) 
        #probs = torch.nn.functional.softmax(scores, dim=-1)
        first_correct_token = torch.argmax(scores[:, -(n_token_seq_len+1), :], dim=-1, keepdim=True)
        
        # ---- crop the provided draft AFTER forward pass ----
        # take the last n_token_seq_len tokens (the draft block) and drop the last one
        prefill_block = input_ids[:, -n_token_seq_len:]
        prefill_drafted_n_gram = prefill_block[:, :-1].contiguous()
        
        # crop KV back to prompt-only (remove appended draft)
        current_len = past_key_values.get_seq_length()
        desired_len = max(0, current_len - n_token_seq_len)
        to_delete   = current_len - desired_len
        if to_delete > 0:
            past_key_values.delete_false_key_value(to_delete)        
        
        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0

    else: # generation phase, input as random_initilized point ([first_corrected_token, tokens_from_prompt]) and output as fixed point

        assert past_key_values is not None
        
        batch, out, device = input_ids.shape[0], input_ids, input_ids.device
        accepted_n_gram = out  # assumes preallocated to n_token_seq_len

        total_accepted = 0
        itr = 0

        while total_accepted < n_token_seq_len:
            itr += 1
            inputs_embeds = self.model.embed_tokens(out)
            attention_mask = torch.ones_like(out, device=input_ids.device)

            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + out.shape[1], device=inputs_embeds.device
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
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                if self.model.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        
            hidden_states = inputs_embeds
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            for decoder_layer in self.model.layers[: self.model.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]
    
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states).float()
    
            # Apply logits processor, then softmax
            p_scores = logits_processors(out, logits.squeeze(0)).unsqueeze(0) 
            
            # TODO: make this generalizable to support top-k, now taking only p_scores for the sake of efficiency
            #p_prob = torch.nn.functional.softmax(p_scores, dim=-1)
            # Greedy tokens for each draft position (exclude the last slot which is prob_next)
            greedy_tokens = torch.argmax(p_scores[:, :-1, :], dim=-1)      # [1, L-1]
            
            # Compare draft vs greedy: accept the longest exact-match prefix
            mismatch = (out[:, 1:] != greedy_tokens)
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)+1
            L = out.shape[1]
            num_accepted_raw = int(accepted[0])

            # --- EOS handling within accepted prefix
            num_accepted = num_accepted_raw
            if eos_enabled:
                # if EOS appears in the accepted prefix, cap acceptance at first EOS
                eos_in_prefix = (out[0, :num_accepted_raw] == eos_id)
                if eos_in_prefix.any():
                    first_eos_idx = torch.nonzero(eos_in_prefix, as_tuple=False)[0].item()
                    num_accepted = first_eos_idx + 1

            # Write accepted portion (possibly capped at EOS)
            if num_accepted > 0:
                accepted_n_gram[:, total_accepted:total_accepted+num_accepted] = out[:, :num_accepted].clone()
            total_accepted += num_accepted

            # If EOS was inside the accepted prefix, finalize immediately
            if eos_enabled and (out[0, :num_accepted] == eos_id).any():
                # Trim KV cache to exactly the accepted sequence length
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    past_key_values.delete_false_key_value(to_delete)
                # Return truncated outputs up to EOS
                return past_key_values, torch.full((1,1), eos_id, device=device, dtype=out.dtype), accepted_n_gram[:, :total_accepted], itr

            has_rejected = (num_accepted_raw < L)  # note: use raw to preserve original mismatch logic

            if has_rejected:
                # Delete false keys&values for the rejected tail
                past_key_values.delete_false_key_value(out.shape[1]-num_accepted_raw)
                # Next token is the greedy token at the first mismatch position
                #next_token = torch.argmax(p_prob[:, num_accepted_raw-1, :], dim=-1, keepdim=True)
                
                # TODO: support p_prob to make more generalizable, while keep efficiency
                next_token = torch.argmax(p_scores[:, num_accepted_raw-1, :], dim=-1, keepdim=True)

                # --- EOS handling on the next sampled token (first mismatch)
                if eos_enabled and next_token.item() == eos_id:
                    # accept EOS and stop
                    accepted_n_gram[:, total_accepted:total_accepted+1] = next_token
                    total_accepted += 1
                    current_len = past_key_values.get_seq_length()
                    desired_len = total_accepted
                    to_delete = max(0, current_len - desired_len)
                    if to_delete > 0:
                        past_key_values.delete_false_key_value(to_delete)
                    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr

                # keep drafting from the mismatch token
                out = next_token

                # Rebuild draft tail greedily from the remaining positions in this pass (after the mismatch slot)
                # TODO: support p_prob to make more generalizable, while keep efficiency
                #q_probs_rem = p_prob[:, num_accepted_raw:-1, :]
                q_probs_rem = p_scores[:, num_accepted_raw:-1, :]
                if q_probs_rem.shape[1] > 0:
                    q_sampled = torch.argmax(q_probs_rem, dim=-1)  # [1, L']
                    out = torch.cat((out, q_sampled), dim=-1)
                    
                continue
    
            # If we didn't reject anything, append the next greedy token and finish this block
            # TODO: support p_prob to make more generalizable, while keep efficiency
            #next_token = torch.argmax(p_prob[:, -1, :], dim=-1, keepdim=True)
            next_token = torch.argmax(p_scores[:, -1, :], dim=-1, keepdim=True)

            # --- write the appended token to accepted_n_gram
            accepted_n_gram[:, total_accepted:total_accepted+1] = next_token

            # --- EOS handling on the appended next token
            if eos_enabled and next_token.item() == eos_id:
                total_accepted += 1
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    past_key_values.delete_false_key_value(to_delete)
                return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr

            total_accepted += 1  # normal non-EOS advance

        # Hit length limit without EOS
        return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr