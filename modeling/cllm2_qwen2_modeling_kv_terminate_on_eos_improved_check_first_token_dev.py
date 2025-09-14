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
    ) -> None:
   
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
    first_token_correct_flag: Optional[bool] = False,
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
    #logits_processors = LogitsProcessorList()

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

        #scores = logits_processors(input_ids, logits.squeeze(0)).unsqueeze(0) 
        #probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # ---- build prefill_drafted_n_gram from ARGMAX of model outputs over the draft block ----
        # take the last n_token_seq_len positions from the sequence and use their next-token predictions
        prefill_drafted_n_gram = torch.argmax(                                  
            logits[:, -n_token_seq_len-1:-1, :], dim=-1                                       
        )                   
        # shape: [1, n_token_seq_len] when input includes a full draft block
        # first_correct_token is mapped from the last token in the prompt
        first_correct_token = prefill_drafted_n_gram[0]
        
        # crop KV back to prompt(remove appended draft)
        if (past_key_values is not None) and (n_token_seq_len > 0):
            past_key_values.delete_false_key_value(n_token_seq_len)    
        
        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0, True

    else: # generation phase, input as random_initilized point ([first_corrected_token, tokens_from_prompt]) and output as fixed point

        assert past_key_values is not None
        
        batch = input_ids.shape[0]
        out = input_ids  # current draft block to evaluate/accept/reject
        device = input_ids.device

        # Preallocate the final accepted sequence buffer (separate from `out`)
        accepted_n_gram = torch.empty(
            (batch, n_token_seq_len), dtype=out.dtype, device=device
        )

        total_accepted = 0
        itr = 0

        # -------------------------------
        # If requested, pre-accept the first token
        # -------------------------------
        if first_token_correct_flag:
            accepted_n_gram[:, 0:1] = out[:, 0:1]
            total_accepted = 1

        while total_accepted < n_token_seq_len:
            itr += 1
            inputs_embeds = self.model.embed_tokens(out)
            attention_mask = torch.ones_like(out, device=input_ids.device)

            # Save the seq length BEFORE pushing this block
            seq_len_before_block = past_key_values.get_seq_length()

            cache_position = torch.arange(
                seq_len_before_block, seq_len_before_block + out.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)

            # MASK BUILDING
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
    
            # Apply logits processor (greedy)
            #p_scores = logits_processors(out, logits.squeeze(0)).unsqueeze(0)
            
            # Greedy tokens for each draft position (exclude the last slot which is prob_next)
            greedy_tokens = torch.argmax(logits[:, :-1, :], dim=-1)      # [bsz, L-1]
            # Compare draft vs greedy: accept the longest exact-match prefix
            mismatch = (out[:, 1:] != greedy_tokens)
            # first mismatch index
            accepted_prefix_len_raw = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0].item() + 1
            L = out.shape[1]

            # EOS handling within the accepted prefix
            accepted_prefix_len = accepted_prefix_len_raw
            if eos_enabled:
                eos_in_prefix = (out[0, :accepted_prefix_len_raw] == eos_id)
                if eos_in_prefix.any():
                    first_eos_idx = torch.nonzero(eos_in_prefix, as_tuple=False)[0].item()
                    accepted_prefix_len = first_eos_idx + 1

            # Determine how many tokens to write to the global buffer this pass.
            # If we pre-accepted the very first token, don't duplicate it: skip out[:, 0].
            already_preaccepted_first = 1 if (first_token_correct_flag and itr == 1) else 0
            num_to_write = max(0, accepted_prefix_len - already_preaccepted_first)
            src_start = already_preaccepted_first
            src_end = already_preaccepted_first + num_to_write

            if num_to_write > 0:
                accepted_n_gram[:, total_accepted:(total_accepted+num_to_write)] = out[:, src_start:src_end]
                total_accepted += num_to_write

            # If EOS was inside the accepted prefix, return
            if eos_enabled and (out[0, :accepted_prefix_len] == eos_id).any():
                
                current_len = past_key_values.get_seq_length()
                desired_len = seq_len_before_block + total_accepted
                to_delete = current_len - desired_len
                past_key_values.delete_false_key_value(to_delete)

                return past_key_values, torch.full((1,1), eos_id, device=device, dtype=out.dtype), accepted_n_gram[:, :total_accepted], itr, False

            has_rejected = (accepted_prefix_len_raw < L)  # use raw for original mismatch logic
            # BRANCH: WITH REJECTED TOKENS IN THE DRAFT
            if has_rejected:
                # Delete false keys&values for the rejected tail
                past_key_values.delete_false_key_value(out.shape[1] - accepted_prefix_len_raw)

                # Next token is the greedy token at the first mismatch position
                next_token = torch.argmax(logits[:, accepted_prefix_len_raw - 1, :], dim=-1, keepdim=True)

                # TERMINATION CONDITIONS:
                # (1) EOS handling on the next sampled token
                # (2) adding next_token suffices n_token_seq_len
                if eos_enabled and next_token.item() == eos_id:
                    # accept EOS and stop
                    accepted_n_gram[:, total_accepted: total_accepted + 1] = next_token
                    total_accepted += 1
                    
                    current_len = past_key_values.get_seq_length()
                    desired_len = seq_len_before_block + total_accepted
                    to_delete = current_len - desired_len
                    past_key_values.delete_false_key_value(to_delete)
                    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, False

                # keep drafting from the mismatch token
                out = next_token
                # Rebuild draft tail greedily from the remaining positions in this pass (after the mismatch slot)
                q_scores_rem = logits[:, accepted_prefix_len_raw:-1, :]
                if q_scores_rem.shape[1] > 0:
                    q_sampled = torch.argmax(q_scores_rem, dim=-1)  # [1, L']
                    out = torch.cat((out, q_sampled), dim=-1)
            
            # BRANCH: WITHOUT REJECTED TOKENS IN THE DRAFT
            else:
                # If we didn't reject anything, append the next greedy token and finish this block
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # write the appended token to accepted_n_gram
                accepted_n_gram[:, total_accepted: total_accepted + 1] = next_token
                total_accepted += 1

                # --- EOS handling on the appended next token
                if eos_enabled and next_token.item() == eos_id:  
                    
                    # Trim KV cache to exactly (previous length + total_accepted)
                    
                    current_len = past_key_values.get_seq_length()
                    desired_len = seq_len_before_block + total_accepted
                    to_delete = current_len - desired_len
                    past_key_values.delete_false_key_value(to_delete)
                    
                    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, True

                # THE WHILE LOOP SHOULD HAVE ENDED BY NOW
                
        # Hit length limit without EOS
        return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, False
