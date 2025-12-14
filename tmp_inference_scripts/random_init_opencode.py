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

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

#TODO: support bsz>1 for Diffusion Decoding
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
    accept_threshold = 0.8,
    ):

    batch, prompt_len, out, device = 1, int(torch.sum(attention_mask[0])), input_ids.clone(), input_ids.device
    seq_lens = torch.full((batch,), prompt_len, device = device, dtype = torch.long)

    ### Initialization draft distribution q(x) with 0-1 distribution from prompt
    q_sampled = []
    q_logits_all = []
    for _ in range(n_token_seq_len):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())]).to(dtype=torch.long, device=model.device).unsqueeze(dim=0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((batch, len(tokenizer)), float('-inf'), device=model.device)
        q_logits.scatter_(1, q_sample, 0.0) 
        q_sampled.append(q_sample)
        q_logits_all.append(q_logits)
    q_sampled = torch.cat(q_sampled, dim = 1)
    q_logits_all = torch.stack(q_logits_all, dim = -2)
    q_logits = q_logits_all

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
    total_accepted = 0
    itr=0
    while total_accepted < n_token_seq_len:

        itr+=1

        ### verify and speculate with larger network within a forward pass
        out_attention_mask = torch.full_like(out, 1).to(model.device)
        logits = model(out, out_attention_mask).logits
        p_logits = logits[:, prompt_len+total_accepted-1:, :]
        # only support bsz=1 now
        p_scores = logits_processors(out, p_logits.squeeze(dim=0)).unsqueeze(dim=0)
        q_scores = logits_processors(out, q_logits.squeeze(dim=0)).unsqueeze(dim=0)

        ### prob and prob of draft distribution (p(x) and q(x))
        p_prob = nn.functional.softmax(p_scores, dim=-1)[:, :, :len(tokenizer)]
        q_prob = nn.functional.softmax(q_scores, dim=-1)[:, :, :len(tokenizer)]

        p, prob_next = p_prob[:, :-1], p_prob[:, -1]

        p = p.gather(-1, q_sampled.unsqueeze(dim=-1))
        q = q_prob.gather(-1, q_sampled.unsqueeze(dim=-1)) * lenience
        
        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)
        threshold = torch.ones_like(q).float() * accept_threshold

        accepted = find_first_true_index(
                (r > (p / q)) | (p < threshold)
            )

        num_accepted = int(accepted[0])
        total_accepted += num_accepted
        out = out[:, :prompt_len+total_accepted]

        has_rejected = (num_accepted < q.shape[1])

        ### sample the additional token to better bound the worst case
        sample_additional_token = False
        if num_accepted == 0: 
            next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
            out = torch.cat((out, next_token), dim = -1)
            total_accepted += 1
            sample_additional_token = True
        elif has_rejected:
            adjusted_prob = F.relu(p_prob[:, num_accepted, :] - q_prob[:, num_accepted, :])
            adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
            prob_next = adjusted_prob
            # if all p_prob < q_prob, prob_next becomes nan, then we do not sample the additional token
            if torch.isnan(prob_next).any():
                pass
            else:
                next_token = torch.multinomial(prob_next, num_samples=1)
                out = torch.cat((out, next_token), dim = -1)
                total_accepted += 1                
                sample_additional_token = True

        if not has_rejected:
            next_token = torch.multinomial(prob_next, num_samples=1)
            out = torch.cat((out, next_token), dim = -1)
            total_accepted += 1
            return out, itr

        if has_rejected:
            ### update q(x) with self-speculated p(x) and sample new drafts tokens
            if sample_additional_token:
                q_logits = p_logits[:, num_accepted+1:-1, :]
                q_probs = p_prob[:, num_accepted+1:-1, :]
            else:
                q_logits = p_logits[:, num_accepted:-1, :]
                q_probs = p_prob[:, num_accepted:-1, :]
            q_sampled = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1)
            out = torch.cat((out, q_sampled), dim = -1)
            print(f'Itr: {itr}, Accepted tokens: {num_accepted}')
            
    
    return out, itr

### Load dataset...
# IN-DOMAIN
#with open("/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0003_avg255_min250_max260.json", 'r') as f:
#    data = json.load(f)

#prompt = data[8000]

data = []
df = pd.read_parquet("/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
data = df.head(100).to_dict(orient="records")
    
model_name = "/checkpoint/lhu/train_ckpts/cllm/logitsaligned-8-27-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_10k_samples_ar_10_lr5e-6/hf_merged_step_1250"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/checkpoint/lhu/models/Qwen2.5-Coder-7B-Instruct")


messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f'Prompt from user: {text}')
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_ids = model_inputs['input_ids']
attention_mask = torch.full_like(input_ids, 1).to(model.device)

### Decoding with Diffusion decoding
iters = []
total_new_tokens = 0
prev_len = input_ids.shape[1]

prompt_lengths = input_ids.shape[1]
generated_part = input_ids[0, prompt_lengths:]
while not ((generated_part == tokenizer.eos_token_id) | (generated_part == 151645)).any():

    generated_part = input_ids[0, prompt_lengths:]
    generated_ids, itr_count = diffusion_decoding(
        model,
        tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_token_seq_len=16,
        temperature = 0.9,
        top_p = 0.9, 
        top_k = 20,
        repetition_penalty = 1.2, 
        lenience = 1.,
        accept_threshold = 0.1,
        )
    
    iters.append(itr_count)
    added = generated_ids.shape[1] - prev_len
    if added > 0:
        total_new_tokens += added
    prev_len = generated_ids.shape[1]

    input_ids = generated_ids
    attention_mask = torch.full_like(input_ids, 1).to(model.device)
    generated_str = ''.join(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
    print(generated_str)

# report averages
if iters:
    avg_iter_per_call = sum(iters) / len(iters)
    avg_iter_per_token = (sum(iters) / total_new_tokens) if total_new_tokens > 0 else float('nan')
    print(f'Average iterations per call: {avg_iter_per_call:.3f}')
    print(f'Average iterations per generated token: {avg_iter_per_token:.3f}')
    print(f'Total iterations: {sum(iters)}, Calls: {len(iters)}, Tokens generated: {total_new_tokens}')

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(f'---------Generated Answer----------')
# print(response)