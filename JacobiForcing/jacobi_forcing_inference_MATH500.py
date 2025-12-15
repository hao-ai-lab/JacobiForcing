from transformers import Qwen2ForCausalLM, Qwen3ForCausalLM, AutoTokenizer
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

import os

import pandas as pd

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy
#Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy
Qwen3ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy

# ---------------------------
# Load dataset (first 100)
# ---------------------------
import pandas as pd

df = pd.read_json("/home/lah003/data/MATH-500/test.jsonl", lines=True)
df_size = len(df)
print(f"Loaded MATH500 dataset with {df_size} samples")
records = df.to_dict(orient="records")

# ---------------------------
# Load model/tokenizer once
# ---------------------------
model_name = "/home/lah003/models/1022-lx-math-4b-math-n16w16"
model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained("/home/lah003/models/Qwen3-4B-Instruct-2507")
model.eval()


eos_id = tokenizer.eos_token_id
alt_eos_id = 151645  # keep your special EOS as a fallback

# ---------------------------
# Generation/profiling config
# ---------------------------
n_token_seq_len = 128

# Safety caps so a sample can't run forever.
max_new_tokens = 512     # hard cap on total new tokens per prompt
max_calls = 1024          # hard cap on number of diffusion_decoding calls per prompt

# ---------------------------
# Iterate the dataset
# ---------------------------
all_rows = []
t0_overall = time.perf_counter()
all_generations = []

total_gen_only_time = 0

for idx, row in tqdm(enumerate(records[:10])):
    task_id = row.get("task_id", f"idx_{idx}")
    # prompt = """Problem: {}\nMark your solution with \\boxed\nAnswer:""".strip().format(
    #         row["problem"].strip()
    #     )
    prompt = row["problem"]
    # messages = [{"role": "user", "content": prompt}]
    messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
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
    prefill_phase = True
    generated_ids = input_ids
    
    prefill_drafted_n_gram = None
    
    gen_only_time = 0

    t_start = time.time()
    # run until EOS or caps
    while True:
        # Check EOS
        generated_part = generated_ids[0, prompt_len:]
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

        ### One diffusion decoding call
        if prefill_phase:
            # pass in random-init draft
            q_sampled = []
            for _ in range(n_token_seq_len):
                q_sample = torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)  # shape [1, n_token_seq_len]
            
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids),dim=-1)
            
            # `jacobi_forward_greedy` will return iteration result from first iteration
            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
                )
            prefill_phase = False
            generated_ids = input_ids
            itr_count = 0
        else:
            # generation phase
            # ---- Initialize a draft tail (any tokens work; we'll fix on the first pass).
            # We keep your "random from prompt" init to avoid extra forward passes.
            if calls == 1:
                # First non-prefill call: reuse draft_tokens produced by prefill
                input_ids = prefill_drafted_n_gram
            else:
                q_sampled = []
                for _ in range(n_token_seq_len-1):
                    q_sample = torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                    q_sampled.append(q_sample)
                q_sampled = torch.cat(q_sampled, dim=1)  # shape [1, n_token_seq_len-1]
                input_ids = torch.cat((first_correct_token.view(1,-1), q_sampled),dim=-1)

            t_gen_start = time.perf_counter()
            past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
            )
            t_gen_time = time.perf_counter() - t_gen_start
            gen_only_time += t_gen_time
            
            generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

        calls += 1
        iters.append(itr_count)

        added = generated_ids.shape[1] - prev_len
        if added > 0:
            total_new_tokens += added
        prev_len = generated_ids.shape[1]
    
    # subtract prefill
    total_new_tokens -= 1
    # per-example finalize
    dt = time.time() - t_start
    total_iterations = sum(iters)
    avg_iter_per_call = (total_iterations / calls)
    avg_iter_per_token = (total_iterations / total_new_tokens)
    
    toks_per_sec = (total_new_tokens / gen_only_time)
    
    total_gen_only_time += gen_only_time
    
    prompt_len = model_inputs["input_ids"].shape[1]
    generated_str = ''.join(tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False))
    print(f'Generated answers: {generated_str}')
    all_generations.append(generated_str)

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
# ---------------------------
# Aggregate + save
# ---------------------------
t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_profile_math500.csv"
df_profile.to_csv(csv_path, index=False)

# Print quick summary (EOS-only)
def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
n_eos = len(df_eos)
n_total = len(df_profile)

print("\n=== Diffusion Decoding Profiling — EOS-only ===")
print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.4f}s")
print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.4f}")
print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

# Optional: also show overall stop-reason distribution for context
print("\nStop reasons (all examples):")
print(df_profile['stop_reason'].value_counts())

# Optional: save EOS-only rows too
df_eos.to_csv("diffusion_profile_greedy_math500_eos.csv", index=False)
