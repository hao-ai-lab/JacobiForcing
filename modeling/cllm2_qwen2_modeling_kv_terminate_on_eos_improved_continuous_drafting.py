from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
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
    # Trim KV tail by num_of_false_tokens positions
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

DynamicCache.delete_false_key_value = delete_false_key_value


def _pad_or_truncate_out(
    out: torch.LongTensor,
    target_len: int,
    input_ids: torch.LongTensor,
    eos_id: Optional[int],
) -> torch.LongTensor:
    """
    Keep out at exactly target_len. If it is shorter, append tokens randomly sampled
    from input_ids (avoid eos if possible). If longer, truncate right tail.
    """
    device = out.device
    cur_len = out.shape[1]
    if cur_len == target_len:
        return out
    if cur_len > target_len:
        return out[:, :target_len]

    # Need to append (target_len - cur_len) tokens sampled from input_ids
    need = target_len - cur_len

    pool = input_ids[0]
    if eos_id is not None:
        non_eos = pool[pool != eos_id]
        if non_eos.numel() > 0:
            pool = non_eos

    # Sample with replacement
    idx = torch.randint(low=0, high=pool.shape[0], size=(need,), device=device)
    filler = pool[idx].unsqueeze(0)  # [1, need]
    return torch.cat([out, filler], dim=-1)


@torch.inference_mode()
def jacobi_forward_greedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9, 
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None, 
    lenience: float = 1.,
    accept_threshold: float = 0.99,
    tokenizer = None,
    eos_token_id: Optional[int] = None,
    max_iteration_count: int = 64,
    ):
    """
      1) Keep `out` at fixed length n_token_seq_len every iteration by padding
         with random tokens from input_ids after any truncation.
      2) Append accepted tokens incrementally into `accepted_n_gram`.
      3) Run until EOS is reached (or iteration cap), not until length limit.
      4) Respect max_iteration_count (default 64).
    """

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    # Resolve EOS id
    eos_id = eos_token_id
    eos_enabled = eos_id is not None
    if not eos_enabled:
        print("!!! WARNING: EOS handling disabled since eos_token_id is None !!!")

    # ---- Prefill phase: build KV for the prompt and return first_correct_token and draft
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
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
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

        # Build prefill_drafted_n_gram from argmax of model outputs over the draft block
        prefill_drafted_n_gram = torch.argmax(
            logits[:, -n_token_seq_len-1:-1, :], dim=-1
        )  # [1, n_token_seq_len]
        first_correct_token = prefill_drafted_n_gram[0]

        # Crop KV back to prompt (remove appended draft)
        if (past_key_values is not None) and (n_token_seq_len > 0):
            past_key_values.delete_false_key_value(n_token_seq_len)

        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0

    # ---- Generation phase
    assert past_key_values is not None, "past_key_values must be provided in generation phase"

    device = input_ids.device
    out = input_ids  # shape [1, L0]
    # Ensure out is exactly n_token_seq_len long at start
    out = _pad_or_truncate_out(out, n_token_seq_len, input_ids, eos_id)

    # We'll accumulate accepted tokens incrementally
    accepted_chunks: List[torch.LongTensor] = []
    total_accepted = 0
    itr = 0
    eos_reached = False
    last_token = out[:, -1:]  # in case we need to return something

    while (not eos_reached) and (itr < max_iteration_count):
        itr += 1

        inputs_embeds = self.model.embed_tokens(out)
        attention_mask = torch.ones_like(out, device=device)

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

        # Greedy tokens for each draft position (exclude the last slot which predicts the next token)
        greedy_tokens = torch.argmax(logits[:, :-1, :], dim=-1)  # [1, L-1]
        mismatch = (out[:, 1:] != greedy_tokens)
        accepted_prefix_len_raw = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1).item() + 1)
        L = out.shape[1]  # should be == n_token_seq_len

        # EOS handling inside the accepted prefix
        num_accepted = accepted_prefix_len_raw
        if eos_enabled:
            eos_in_prefix = (out[0, :accepted_prefix_len_raw] == eos_id)
            if eos_in_prefix.any():
                first_eos_idx = torch.nonzero(eos_in_prefix, as_tuple=False)[0].item()
                num_accepted = first_eos_idx + 1

        # Commit accepted prefix tokens (possibly capped at EOS)
        if num_accepted > 0:
            accepted_seg = out[:, :num_accepted].clone()
            accepted_chunks.append(accepted_seg)
            total_accepted += num_accepted

        # If EOS occurred within the accepted prefix, finalize
        if eos_enabled and (out[0, :num_accepted] == eos_id).any():
            eos_reached = True
            # Trim KV cache to exactly accepted seq length
            current_len = past_key_values.get_seq_length()
            desired_len = total_accepted
            to_delete = max(0, current_len - desired_len)
            if to_delete > 0:
                past_key_values.delete_false_key_value(to_delete)
            last_token = torch.full((1, 1), eos_id, device=device, dtype=out.dtype)
            break

        has_rejected = (accepted_prefix_len_raw < L)

        if has_rejected:
            # Delete KV entries for rejected tail
            past_key_values.delete_false_key_value(L - accepted_prefix_len_raw)

            # Next token = greedy at first mismatch position
            next_token = torch.argmax(logits[:, accepted_prefix_len_raw - 1, :], dim=-1, keepdim=True)  # [1,1]
            last_token = next_token

            # If next token is EOS, accept it and stop
            if eos_enabled and next_token.item() == eos_id:
                accepted_chunks.append(next_token)
                total_accepted += 1
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    past_key_values.delete_false_key_value(to_delete)
                eos_reached = True
                break

            # Build new draft starting from the mismatch: [next_token, greedy remainder], then pad to fixed length
            q_probs_rem = logits[:, accepted_prefix_len_raw:-1, :]  # predictions for remaining draft slots
            if q_probs_rem.shape[1] > 0:
                q_sampled = torch.argmax(q_probs_rem, dim=-1)  # [1, L']
                new_out = torch.cat([next_token, q_sampled], dim=-1)  # length = (L - accepted_prefix_len_raw)
            else:
                new_out = next_token  # length = 1

            # Pad/truncate to fixed draft length
            out = _pad_or_truncate_out(new_out, n_token_seq_len, input_ids, eos_id)

        else:
            # No rejection: shift window and append the next greedy token
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [1,1]
            last_token = next_token

            # Commit exactly the newly generated next token
            accepted_chunks.append(next_token)
            total_accepted += 1

            # Stop if EOS
            if eos_enabled and next_token.item() == eos_id:
                # Trim KV to accepted length
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    past_key_values.delete_false_key_value(to_delete)
                eos_reached = True
                break

            # Maintain fixed-length draft by sliding window
            out = torch.cat([out[:, 1:], next_token], dim=-1)
            # (No need to pad; length stays L)

        # Enforce invariant: out has fixed length n_token_seq_len
        if out.shape[1] != n_token_seq_len:
            out = _pad_or_truncate_out(out, n_token_seq_len, input_ids, eos_id)

    # Prepare final accepted_n_gram tensor
    if len(accepted_chunks) == 0:
        accepted_n_gram = torch.empty((1, 0), dtype=input_ids.dtype, device=device)
    else:
        accepted_n_gram = torch.cat(accepted_chunks, dim=1)

    return past_key_values, last_token, accepted_n_gram, itr
