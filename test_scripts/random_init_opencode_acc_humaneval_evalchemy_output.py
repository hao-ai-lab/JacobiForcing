from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
import json
from tqdm import tqdm
import time

import pandas as pd

import os

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

def _first_eos_abs_pos(ids_2d: torch.Tensor, start: int, eos_ids):
    """
    Return absolute index of the first EOS in ids_2d[0, start:], or None if not found.
    """
    assert ids_2d.dim() == 2 and ids_2d.size(0) == 1
    tail = ids_2d[0, start:]
    if tail.numel() == 0:
        return None
    mask = None
    for eid in eos_ids:
        if eid is None:
            continue
        m = (tail == eid)
        mask = m if mask is None else (mask | m)
    if mask is not None and mask.any():
        first_rel = int(torch.nonzero(mask, as_tuple=False)[0].item())
        return start + first_rel
    return None

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
    eos_ids=None,
):
    if eos_ids is None:
        eos_ids = []
    eos_set = set(e for e in eos_ids if e is not None)

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

        # if accepted prefix contains EOS, trim to EOS and stop
        eos_abs = _first_eos_abs_pos(out, prompt_len, eos_ids)
        if eos_abs is not None:
            out = out[:, :eos_abs + 1]
            return out, itr

        has_rejected = (num_accepted < q.shape[1])

        ### sample the additional token to better bound the worst case
        sample_additional_token = False
        if num_accepted == 0: 
            next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
            out = torch.cat((out, next_token), dim = -1)

            # return immediately upon sampling from EOS
            if int(next_token[0,0].item()) in eos_set:
                return out, itr

            total_accepted += 1
            sample_additional_token = True
        elif has_rejected:
            adjusted_prob = F.relu(p_prob[:, num_accepted, :] - q_prob[:, num_accepted, :])
            adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
            prob_next = adjusted_prob
            if torch.isnan(prob_next).any():
                pass
            else:
                next_token = torch.multinomial(prob_next, num_samples=1)
                out = torch.cat((out, next_token), dim = -1)
                # return immediately upon sampling from EOS
                if int(next_token[0,0].item()) in eos_set:
                    return out, itr
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
            
            #print(f'Itr: {itr}, Accepted tokens: {num_accepted}')
    
    #out = out[:, :prompt_len + total_accepted]
    return out, itr

### Load dataset...
# IN-DOMAIN
#with open("/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0003_avg255_min250_max260.json", 'r') as f:
#    data = json.load(f)

#prompt = data[8000]

# ---------------------------
# Load dataset (first 100)
# ---------------------------
df = pd.read_parquet("/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
#records = df.head(100).to_dict(orient="records")
records = df.to_dict(orient="records")

# ---------------------------
# Load model/tokenizer once
# ---------------------------
model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr1e-6/hf_merged_step_59258"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/checkpoint/lhu/models/Qwen2.5-Coder-7B-Instruct")
model.eval()

eos_id = tokenizer.eos_token_id
alt_eos_id = 151645  # keep your special EOS as a fallback
EOS_IDS = [eid for eid in (eos_id, alt_eos_id) if eid is not None]

# ---------------------------
# Generation/profiling config
# ---------------------------
n_token_seq_len = 16
temperature = 0.9
top_p = 0.9
top_k = 20
repetition_penalty = 1.2
lenience = 1.0
accept_threshold = 0.1

# Safety caps so a sample can't run forever.
max_new_tokens = 1024     # hard cap on total new tokens per prompt
max_calls = 10240         # hard cap on number of diffusion_decoding calls per prompt

# ---------------------------
# Iterate the dataset
# ---------------------------
all_rows = []
t0_overall = time.perf_counter()
all_generations = []

for idx, row in tqdm(enumerate(records)):
    task_id = row.get("task_id", f"idx_{idx}")
    #prompt = "You are given a partially completed Python function with the header and the doc string. Complete the following function according to given information:\n\n" + row["prompt"]
    prompt = "Respond only in code.\n" + row["prompt"]
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    input_ids = model_inputs["input_ids"]
    attention_mask = torch.full_like(input_ids, 1, device=model.device)

    # per-example stats
    iters = []
    total_new_tokens = 0
    calls = 0
    prev_len = input_ids.shape[1]
    prompt_len = prev_len
    stop_reason = None

    t_start = time.perf_counter()

    # run until EOS or caps
    while True:
        # Check EOS
        generated_part = input_ids[0, prompt_len:]
        hit_eos = False
        if eos_id is not None:
            hit_eos = (generated_part == eos_id).any().item()
        if not hit_eos:
            # allow alternate special EOS id
            hit_eos = (generated_part == alt_eos_id).any().item()

        if hit_eos:
            stop_reason = "eos"
            break
        if total_new_tokens >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if calls >= max_calls:
            stop_reason = "max_calls"
            break
        
        #print(f"\nInit new subsequence {calls}...\n")

        # One diffusion decoding call
        generated_ids, itr_count = diffusion_decoding(
            model,
            tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_token_seq_len=n_token_seq_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            lenience=lenience,
            accept_threshold=accept_threshold,
            eos_ids=EOS_IDS,
        )

        calls += 1
        iters.append(itr_count)

        prompt_len = model_inputs["input_ids"].shape[1]
        eos_abs = _first_eos_abs_pos(generated_ids, prompt_len, EOS_IDS)

        # account added tokens before trimming (so counters reflect total work done)
        added = generated_ids.shape[1] - prev_len
        if added > 0:
            total_new_tokens += added
        prev_len = generated_ids.shape[1]

        if eos_abs is not None:
            # keep EOS, drop everything after, then break
            new_len_effective = eos_abs + 1
            if generated_ids.shape[1] != new_len_effective:
                generated_ids = generated_ids[:, :new_len_effective]
            stop_reason = "eos"
            input_ids = generated_ids
            attention_mask = torch.full_like(input_ids, 1, device=model.device)
            break  # immediate exit on EOS

        input_ids = generated_ids
        attention_mask = torch.full_like(input_ids, 1, device=model.device)

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_str = ''.join(tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False))
    print(f'Generated answers: {generated_str}')
    all_generations.append(generated_str)

    # per-example finalize
    dt = time.perf_counter() - t_start
    total_iterations = sum(iters)
    avg_iter_per_call = (total_iterations / calls) if calls > 0 else float("nan")
    avg_iter_per_token = (total_iterations / total_new_tokens) if total_new_tokens > 0 else float("nan")
    toks_per_sec = (total_new_tokens / dt) if dt > 0 else float("nan")

    all_rows.append(
        {
            "index": idx,
            "task_id": task_id,
            "prompt_tokens": prompt_len,
            "new_tokens": total_new_tokens,
            "calls": calls,
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": dt,
            "toks_per_sec": toks_per_sec,
            "stop_reason": stop_reason,
        }
    )

    # light progress
    if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
        print(f"====[{idx+1}/{len(records)}] task_id={task_id} new_toks={total_new_tokens} "
              f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}====")
        
#### ADDED Lines ####
import json
import re

# Function to load the data from JSONL
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Function to save the data to JSONL
def save_jsonl(data, save_path):
    with open(save_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Function to extract Python code block from a string
def extract_python_code(text):
    match = re.search(r'```python([\s\S]*?)```', text)  # Regex to match the block
    if match:
        return match.group(1).strip()  # Return the code inside the block
    else:
        return text  # Return orginal one if no match is found

eval_dir = "/checkpoint/lhu/data/CLLM2_eval_generations/test_speed"
os.makedirs(eval_dir, exist_ok=True)

original_path = os.path.join(eval_dir, 'humaneval_python_example.jsonl')
original_generations = load_jsonl(original_path)

# Process each generation and update with processed generation
for i, original_generation in enumerate(original_generations):
    # Assuming `all_generations[i]` exists and has an 'extracted' key or method
    original_generation['output'] = all_generations[i]
    processed_generation = extract_python_code(all_generations[i])  # Apply the extract method
    print(f'Task id: {i}, Extracted answer: {processed_generation}')
    original_generation['generation'] = processed_generation

# Save processed generations
save_path = os.path.join(eval_dir, f'progressive_noise_shifted_mask_ntok_16_tmp_0p9_top_0p9_topk_20_threshold_0p1_code_only_prompt_humaneval_wo_kv_generation_{model_name.split("/")[-1]}.jsonl')
save_jsonl(original_generations, save_path)

print(f"\n=== All generation done (HumanEval). Results are saved to {save_path} ===")

#### ADDED Lines ####


# ---------------------------
# Aggregate + save
# ---------------------------
t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_profile_humaneval100.csv"
df_profile.to_csv(csv_path, index=False)

# Print quick summary (EOS-only)
def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
n_eos = len(df_eos)
n_total = len(df_profile)

print("\n=== Diffusion Decoding Profiling (HumanEval-100) — EOS-only ===")
print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.2f}s")
print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.2f}")
print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.2f}")
print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.2f}")
print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.2f}")
print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.2f}")

# Optional: also show overall stop-reason distribution for context
print("\nStop reasons (all examples):")
print(df_profile['stop_reason'].value_counts())

# Optional: save EOS-only rows too
df_eos.to_csv("diffusion_profile_humaneval100_eos.csv", index=False)
