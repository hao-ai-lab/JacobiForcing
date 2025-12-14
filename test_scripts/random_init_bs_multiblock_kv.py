from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
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
from generate_trajectory.v2.cllm2_qwen2_modeling_new_multiblock import diffusion_forward

Qwen2ForCausalLM.diffusion_forward = diffusion_forward

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
    attention_mask = (position_ids >= first_non_pad_idx.unsqueeze(1)).long()    # [B, L]
    return attention_mask

def compute_left_pad_lengths(batch_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    first_nonpad_idx = (batch_ids != pad_token_id).float().argmax(dim=1)
    return first_nonpad_idx

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

### Load dataset...
#data = []
#with open("data/raw_data/openthoughts2_1m.json", 'r') as f:
#    for idx, line in enumerate(f):
#        if idx>150:
#            break
#        data.append(json.loads(line))

data = []
df = pd.read_parquet("/checkpoint/lhu/data/OpenThoughts-114k/data/train-00000-of-00006.parquet")
data = df.head(100).to_dict(orient="records")

model_name = "/checkpoint/lhu/models/OpenThinker2-7B"

model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/code/users/lhu/models/OpenThinker2-7B")
tokenizer.padding_side = "left"

multiblock_size = 4

#TODOs: Check if this is okay
print(f'Changing padding_side to {tokenizer.padding_side}')
print('Padding token is the same as EOS token.')

prompts = [
    data[0]['conversations'][0]["value"],
    data[1]['conversations'][0]["value"],
    data[2]['conversations'][0]["value"],
]
#prompts = [
#    data[0]['conversations'][0]["value"]
#]

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
iteration = 0
n_token_seq_len = 64
prefill_phase = True

import time

t0 = time.perf_counter()
while True:
   
    eos_found = []
    for i in range(input_ids.size(0)):
        generated_part = input_ids[i, prompt_lengths[i]:]
        eos_found.append((generated_part == tokenizer.eos_token_id).any())
    
    eos_found = torch.stack(eos_found)
    if eos_found.all():
        break

    if iteration * n_token_seq_len > 16384:
        print('Total length exceeds 16384.')
        break
    
    multiblock_draft_futures = None
    if prefill_phase:
        past_key_values = model.diffusion_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            prefill_phase=prefill_phase,
            n_token_seq_len=n_token_seq_len,
            temperature=1.0,
            top_p=0.9, 
            top_k=None,
            repetition_penalty=None,
            lenience=1.,
            accept_threshold=0.99,
            tokenizer=tokenizer,
            multiblock_size=multiblock_size
        )
        prefill_phase=False
        generated_ids=input_ids
        
    else:
        generated_ids, past_key_values, multiblock_draft_futures = model.diffusion_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            prefill_phase=prefill_phase,
            n_token_seq_len=n_token_seq_len,
            temperature=1.0,
            top_p=0.9,
            top_k=None,
            repetition_penalty=None,
            lenience=1.,
            accept_threshold=0.99,
            tokenizer=tokenizer,
            multiblock_size=multiblock_size,
            multiblock_draft_futures=multiblock_draft_futures,
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

t1 = time.perf_counter()
print(f"Start time: {t0:.6f}, End time: {t1:.6f}, Total elapsed: {t1 - t0:.3f} s")
print(generated_ids[0].shape[0])
