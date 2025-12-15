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

def make_left_pad_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Create an attention mask that only masks out the left-padded tokens,
    assuming left-padding was applied by the tokenizer.

    This function sets the attention mask to 0 for the leading (leftmost)
    consecutive pad_token_id tokens, and 1 elsewhere — including any pad_token_ids
    that may appear later during generation (which should not be masked).

    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len), containing token IDs.
        pad_token_id (int): The ID used for padding tokens.

    Returns:
        torch.Tensor: Attention mask of shape (batch_size, seq_len),
                      with 0s for left padding and 1s elsewhere.
    """
    # Identify padding positions
    is_pad = input_ids == pad_token_id  # [B, L]

    # Find the index of the first non-padding token for each sample
    first_non_pad_idx = (~is_pad).float().argmax(dim=1)  # [B]

    # Create position indices
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, L]

    # Mask positions before the first non-padding token
    attention_mask = (position_ids >= first_non_pad_idx.unsqueeze(1)).long()  # [B, L]
    return attention_mask

def compute_left_pad_lengths(batch_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    first_nonpad_idx = (batch_ids != pad_token_id).float().argmax(dim=1)
    return first_nonpad_idx

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

@classmethod
def empty_like(cls, other: "DynamicCache") -> "DynamicCache":
    
    new_cache = cls()
    # TODO: make cache size configurable
    shape = (other.key_cache[0].shape[0], other.key_cache[0].shape[1], other.key_cache[0].shape[2]+33, other.key_cache[0].shape[3])
    new_cache.key_cache = [
    torch.zeros(shape, device=other.key_cache[0].device, dtype=other.key_cache[0].dtype)
    for _ in range(len(other.key_cache))
]
    new_cache.value_cache = [
    torch.zeros(shape, device=other.key_cache[0].device, dtype=other.key_cache[0].dtype)
    for _ in range(len(other.key_cache))
]

    return new_cache

DynamicCache.empty_like = classmethod(empty_like)

def index_select_batch(self, indices: List[int]) -> "DynamicCache":

    new_cache = DynamicCache()
    new_cache.key_cache = []
    new_cache.value_cache = []
    idx_tensor = torch.tensor(indices, device=self.key_cache[0].device)
    for k in self.key_cache:
        new_cache.key_cache.append(k.index_select(0, idx_tensor))
    for v in self.value_cache:
        new_cache.value_cache.append(v.index_select(0, idx_tensor))
    return new_cache

DynamicCache.index_select_batch = index_select_batch

def merge(self, other: "DynamicCache", indices: list[int]):

    if len(indices) == 0:
        return

    idx_tensor = torch.tensor(indices, device=other.key_cache[0].device)
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx][idx_tensor] = other.key_cache[layer_idx].clone()
        self.value_cache[layer_idx][idx_tensor] = other.value_cache[layer_idx].clone()
        
DynamicCache.merge = merge

            
@torch.inference_mode()
def diffusion_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len = 64,
    temperature = 1.0,
    top_p = 0.9, 
    top_k = None,
    repetition_penalty = None, 
    lenience = 1.,
    accept_threshold = 0.99,
    tokenizer = None,
    ):

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    ### Initialize LogitsProcessor with GenerationConfig
    logits_processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        
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

        past_key_values.delete_false_key_value(1)
        
        return past_key_values

    else: # generation phase, input as random_initilized point and output as fixed point

        assert past_key_values is not None
        
        batch, prompt_lens, out, device = input_ids.shape[0], attention_mask.sum(dim=1), input_ids.clone(), input_ids.device
        pad_lens = compute_left_pad_lengths(input_ids, tokenizer.pad_token_id)
        
        ### Initialization draft distribution q(x) with 0-1 distribution from prompt
        q_sampled = torch.empty(batch, n_token_seq_len, dtype=torch.long, device=input_ids.device)
        for i in range(batch):
            choices = input_ids[i, :prompt_lens[i]+pad_lens[i]].tolist()
            q_sampled[i] = torch.tensor(random.choices(choices, k=n_token_seq_len),
                                    device=input_ids.device)
        out = torch.cat([out, q_sampled], dim=1)

        total_accepted = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        unfinished_mask = total_accepted < n_token_seq_len
        itr=0
        q_sampled_ids = {}
        delete_key_values_length = n_token_seq_len + 1
        past_key_values_fin = DynamicCache.empty_like(past_key_values)

        while True:

            ### 1) Group input by convergence & Use only unconverged samples
            idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
            out_unfin = out[idx_unfin]
            past_key_values_unfin = past_key_values.index_select_batch(idx_unfin.tolist())

            ### 2）Verify and speculate with larger network within a forward pass
            
            # batched kv-cache is obtained based on the delete_length
            inputs_embeds = self.model.embed_tokens(out_unfin[:, -delete_key_values_length:])
            out_attention_mask_unfin = make_left_pad_attention_mask(out_unfin, tokenizer.pad_token_id).to(input_ids.device)
            
            past_seen_tokens = past_key_values_unfin.get_seq_length()
            
            # batched kv-cache & input_embeds is obtained based on the delete_length
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + delete_key_values_length, device=inputs_embeds.device
            )
            # print(f'past_seen_tokens: {past_seen_tokens}')
            position_ids = cache_position.unsqueeze(0)
    
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": out_attention_mask_unfin,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values_unfin,
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
                    past_key_value=past_key_values_unfin,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]
        
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states).float()
            
            for n, idx in enumerate(idx_unfin):
                start = total_accepted[idx]
                p_logits = logits[n, start:, :]
                p_score = logits_processors(out[idx].unsqueeze(0), p_logits).unsqueeze(dim=0)
    
                p_prob = nn.functional.softmax(p_score, dim=-1)[:, :, :len(tokenizer)]
                p, prob_next = p_prob[:, :-1], p_prob[:, -1]       
    
                if itr == 0:
                    p = p.gather(-1, q_sampled[idx].unsqueeze(dim=0).unsqueeze(dim=-1))
                else:
                    p = p.gather(-1, q_sampled_ids.get(idx.item()).unsqueeze(dim=0).unsqueeze(dim=-1))
                    
                p = rearrange(p, '1 n 1 -> 1 n')
     
                accepted = find_first_true_index(p < accept_threshold)
                num_accepted = int(accepted[0])
                total_accepted[idx] += num_accepted
            
                cut = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx]
                mid_state_out = out[idx, :cut]
            
                # Additional sample if necessary
                sample_additional_token = False
                if num_accepted == 0:
                    next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1).squeeze(dim=0)
                    mid_state_out = torch.cat((mid_state_out, next_token), dim = -1)
                    total_accepted[idx] += 1
                    sample_additional_token = True
            
                has_rejected = (mid_state_out.shape[0] < out[idx].shape[0])
                if has_rejected:
                    ### update q(x) with self-speculated p(x) and sample new drafts tokens
                    if sample_additional_token:
                        q_probs = p_prob[:, num_accepted+1:-1, :]
                    else:
                        q_probs = p_prob[:, num_accepted:-1, :]
                    q_sampled_id = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1).squeeze(dim=0)
                    q_sampled_ids[idx.item()] = q_sampled_id
                    mid_state_out = torch.cat((mid_state_out, q_sampled_id), dim = -1)
                    out[idx] = mid_state_out
                    
                print(f'Iteration: {itr}')

            ### Delete False key&value if unfinished
            finished_n_poss = []
            orig_idxs = []
            for n_pos, orig_idx in enumerate(idx_unfin.tolist()):
                if total_accepted[orig_idx] == n_token_seq_len:
                    finished_n_poss.append(n_pos)
                    orig_idxs.append(orig_idx)
            if not len(finished_n_poss)==0:
                mid_point = past_key_values_unfin.index_select_batch(finished_n_poss)
                # print(past_key_values_fin.key_cache[0].shape)
                # print(mid_point.key_cache[0].shape)
                # print(past_key_values_unfin.key_cache[0] == mid_point.key_cache[0])
                past_key_values_fin.merge(mid_point, orig_idxs)

            unfinished_mask = total_accepted < n_token_seq_len      # [B] bool
            if not unfinished_mask.any():
                break 
                      
            itr+=1 

        past_key_values = past_key_values_fin
        past_key_values.delete_false_key_value(1)

        return out, past_key_values


@torch.inference_mode()
def get_diffusion_decoding_trajectory(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len = 64,
    temperature = 1.0,
    top_p = 0.9, 
    top_k = None,
    repetition_penalty = None, 
    lenience = 1.,
    accept_threshold = 0.99,
    tokenizer = None,
    ):

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    ### Initialize LogitsProcessor with GenerationConfig
    logits_processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        
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

        past_key_values.delete_false_key_value(1)
        
        return past_key_values

    else: # generation phase, input as random_initilized point and output as fixed point

        assert past_key_values is not None
        
        batch, prompt_lens, out, device = input_ids.shape[0], attention_mask.sum(dim=1), input_ids.clone(), input_ids.device
        pad_lens = compute_left_pad_lengths(input_ids, tokenizer.pad_token_id)
        
        ### Initialization draft distribution q(x) with 0-1 distribution from prompt
        trajectory = {i: [] for i in range(batch)}
        q_sampled = torch.empty(batch, n_token_seq_len, dtype=torch.long, device=input_ids.device)
        for i in range(batch):
            choices = input_ids[i, :prompt_lens[i]+pad_lens[i]].tolist()
            q_sampled[i] = torch.tensor(random.choices(choices, k=n_token_seq_len),
                                    device=input_ids.device)
        out = torch.cat([out, q_sampled], dim=1)
        for i in range(batch):
            trajectory[i].append(out[i].reshape(1, -1))

        total_accepted = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        unfinished_mask = total_accepted < n_token_seq_len
        itr=0
        q_sampled_ids = {}
        delete_key_values_length = n_token_seq_len + 1
        past_key_values_fin = DynamicCache.empty_like(past_key_values)

        while True:
            ### 1) Group input by convergence & Use only unconverged samples
            idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
            out_unfin = out[idx_unfin]
            past_key_values_unfin = past_key_values.index_select_batch(idx_unfin.tolist())

            ### 2）Verify and speculate with larger network within a forward pass
            
            # batched kv-cache is obtained based on the delete_length
            inputs_embeds = self.model.embed_tokens(out_unfin[:, -delete_key_values_length:])
            out_attention_mask_unfin = make_left_pad_attention_mask(out_unfin, tokenizer.pad_token_id).to(input_ids.device)
            
            past_seen_tokens = past_key_values_unfin.get_seq_length()
            
            # batched kv-cache & input_embeds is obtained based on the delete_length
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + delete_key_values_length, device=inputs_embeds.device
            )
            # print(f'past_seen_tokens: {past_seen_tokens}')
            position_ids = cache_position.unsqueeze(0)
    
            if not isinstance(causal_mask_mapping := attention_mask, dict):
                # Prepare mask arguments
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": out_attention_mask_unfin,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values_unfin,
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
                    past_key_value=past_key_values_unfin,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]
        
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states).float()
            
            for n, idx in enumerate(idx_unfin):
                start = total_accepted[idx]
                p_logits = logits[n, start:, :]
                p_score = logits_processors(out[idx].unsqueeze(0), p_logits).unsqueeze(dim=0)
    
                p_prob = nn.functional.softmax(p_score, dim=-1)[:, :, :len(tokenizer)]
                p, prob_next = p_prob[:, :-1], p_prob[:, -1]       
    
                if itr == 0:
                    p = p.gather(-1, q_sampled[idx].unsqueeze(dim=0).unsqueeze(dim=-1))
                else:
                    p = p.gather(-1, q_sampled_ids.get(idx.item()).unsqueeze(dim=0).unsqueeze(dim=-1))
                    
                p = rearrange(p, '1 n 1 -> 1 n')
     
                accepted = find_first_true_index(p < accept_threshold)
                num_accepted = int(accepted[0])
                total_accepted[idx] += num_accepted
            
                cut = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx]
                mid_state_out = out[idx, :cut]
            
                # Additional sample if necessary
                sample_additional_token = False
                if num_accepted == 0:
                    next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1).squeeze(dim=0)
                    mid_state_out = torch.cat((mid_state_out, next_token), dim = -1)
                    total_accepted[idx] += 1
                    sample_additional_token = True
            
                has_rejected = (mid_state_out.shape[0] < out[idx].shape[0])
                if has_rejected:
                    ### update q(x) with self-speculated p(x) and sample new drafts tokens
                    if sample_additional_token:
                        q_probs = p_prob[:, num_accepted+1:-1, :]
                    else:
                        q_probs = p_prob[:, num_accepted:-1, :]
                    q_sampled_id = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1).squeeze(dim=0)
                    q_sampled_ids[idx.item()] = q_sampled_id
                    mid_state_out = torch.cat((mid_state_out, q_sampled_id), dim = -1)
                    out[idx] = mid_state_out
                    
                trajectory[idx.item()].append(mid_state_out.unsqueeze(dim=0))

            ### Delete False key&value if unfinished
            finished_n_poss = []
            orig_idxs = []
            for n_pos, orig_idx in enumerate(idx_unfin.tolist()):
                if total_accepted[orig_idx] == n_token_seq_len:
                    finished_n_poss.append(n_pos)
                    orig_idxs.append(orig_idx)
                    
            if not len(finished_n_poss)==0:
                mid_point = past_key_values_unfin.index_select_batch(finished_n_poss)
                past_key_values_fin.merge(mid_point, orig_idxs)

            unfinished_mask = total_accepted < n_token_seq_len      # [B] bool
            if not unfinished_mask.any():
                break 
                      
            itr+=1 

        past_key_values = past_key_values_fin
        past_key_values.delete_false_key_value(1)

        return trajectory, past_key_values
