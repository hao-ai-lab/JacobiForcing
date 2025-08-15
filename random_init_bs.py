from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
import json

import pandas as pd

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

# logits processors
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

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

@torch.inference_mode()
def diffusion_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    n_token_seq_len,
    temperature = 0.9,
    top_p = 0.9, 
    top_k = 20,
    repetition_penalty = 1.05, 
    lenience = 1.,
    accept_threshold = 0.9,
    ):
    
    batch, prompt_lens, out, device = input_ids.shape[0], attention_mask.sum(dim=1), input_ids.clone(), input_ids.device
    pad_lens = compute_left_pad_lengths(input_ids, tokenizer.pad_token_id)
    
    ### Initialization draft distribution q(x) with 0-1 distribution from prompt
    q_sampled = torch.empty(batch, n_token_seq_len, dtype=torch.long, device=model.device)
    for i in range(batch):
        choices = input_ids[i, :prompt_lens[i]+pad_lens[i]].tolist()
        q_sampled[i] = torch.tensor(random.choices(choices, k=n_token_seq_len),
                                device=model.device)
    out = torch.cat([out, q_sampled], dim=1)

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
    
    ### Diffusion decoding
    total_accepted = torch.zeros(batch, dtype=torch.long, device=model.device)
    itr=0
    q_sampled_ids = {}
    while True:
        
        ### 1) Group input by convergence & Use only unconverged samples
        unfinished_mask = total_accepted < n_token_seq_len      # [B] bool
        if not unfinished_mask.any():
            break 

        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
        out_unfin = out[idx_unfin]                        # [B_un, L]

        ### 2）Verify and speculate with larger network within a forward pass
        out_attention_mask_unfin = make_left_pad_attention_mask(out_unfin, tokenizer.pad_token_id).to(model.device)
        logits = model(out_unfin, out_attention_mask_unfin).logits
        
        for n, idx in enumerate(idx_unfin):
            start = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx] - 1
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
            
        itr+=1
    
    return out

### Load dataset...
#data = []
#with open("data/raw_data/openthoughts2_1m.json", 'r') as f:
#    for idx, line in enumerate(f):
#        if idx>150:
#            break
#        data.append(json.loads(line))

data = []
df = pd.read_parquet("/checkpoint/lhu/data/OpenThoughts-114k/data/train-00001-of-00006.parquet")
data = df.head(100).to_dict(orient="records")

model_name = "/checkpoint/lhu/train_ckpts/cllm/trial-0-orderly-efficient-train-cllm-openthinker2-7B-ntok32-eos_tokens-without_think_format_split_ratio_40_size_2848_ntok_64_sampling_ratio_1_lookup_size_640_cllm_soft_loss_length_capped_16k_flexattn/hf_merged"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/checkpoint/lhu/models/OpenThinker2-7B")
tokenizer.padding_side = "left"
#TODOs: Check if this is okay
print(f'Changing padding_side to {tokenizer.padding_side}')
print('Padding token is the same as EOS token')

# prompts = [
#     data[0]['conversations'][0]["value"],
#     data[1]['conversations'][0]["value"],
#     data[2]['conversations'][0]["value"],
# ]
prompts = [
    data[1]['conversations'][0]["value"]
]

texts = []
for prompt in prompts:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    texts.append(text)

model_inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,        
    truncation=True,     
).to(model.device)

input_ids = model_inputs["input_ids"]         
attention_mask = model_inputs["attention_mask"] 
prompt_lengths = attention_mask.sum(dim=1)

### Decoding with Diffusion decoding
while True:
   
    eos_found = []
    for i in range(input_ids.size(0)):
        generated_part = input_ids[i, prompt_lengths[i]:]
        eos_found.append((generated_part == tokenizer.eos_token_id).any())
    
    eos_found = torch.stack(eos_found)
    if eos_found.all():
        break
    
    generated_ids = diffusion_decoding(
        model,
        tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_token_seq_len=64,
        temperature = 0.9,
        top_p = 0.9, 
        top_k = 20,
        repetition_penalty = 1.1, 
        lenience = 1.,
        accept_threshold = 0.99,
        )

    input_ids = generated_ids
    attention_mask = make_left_pad_attention_mask(input_ids, tokenizer.pad_token_id).to(model.device)
    generated_str = ''.join(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
    print(generated_str)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'---------Generated Answer----------')
print(response)