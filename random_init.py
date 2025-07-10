from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
import json

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

# sampling helpers
# def log(t, eps = 1e-20):
#     return torch.log(t.clamp(min = eps))

# def gumbel_sample(logits, temperature=1.0, dim=-1):
#     # Replace -inf with very negative number to avoid NaN
#     safe_logits = logits.clone()
#     safe_logits[torch.isinf(safe_logits)] = -1e9

#     noise = -torch.log(-torch.log(torch.rand_like(safe_logits)))
#     return ((safe_logits / max(temperature, 1e-10)) + noise).argmax(dim=dim)

# def top_k(logits, thres = 0.95):
#     k = math.ceil((1 - thres) * logits.shape[-1])
#     val, ind = torch.topk(logits, k)
#     probs = torch.full_like(logits, float('-inf'))
#     probs.scatter_(-1, ind, val)
#     return probs

# def safe_div(num, den, eps = 1e-10):
#     return num / max(den, eps)

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
    accept_threshold = 0.1,
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
            return out

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
            
        itr+=1
    
    return out

### Load dataset...
data = []
with open("data/raw_data/openthoughts2_1m.json", 'r') as f:
    for idx, line in enumerate(f):
        if idx>100:
            break
        data.append(json.loads(line))

model_name = "/data/phd/kousiqi/kousiqi/ckpts/checkpoint-4500/final_model_fp32.bin"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("/data/phd/kousiqi/kousiqi/ckpts/OpenThinker2-7B")

prompt = data[0]['conversations'][0]["value"]

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
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
while not (input_ids == tokenizer.eos_token_id).any():
    
    generated_ids = diffusion_decoding(
        model,
        tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_token_seq_len=64,
        temperature = 0.9,
        top_p = 0.9, 
        top_k = 20,
        repetition_penalty = 1.05, 
        lenience = 1.,
        accept_threshold = 0.1,
        )

    input_ids = generated_ids
    attention_mask = torch.full_like(input_ids, 1).to(model.device)
    generated_str = ''.join(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
    print(generated_str)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'---------Generated Answer----------')
print(response)
