from transformers import Qwen2ForCausalLM, AutoTokenizer
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

import re

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified import jacobi_forward_greedy_multiblock
Qwen2ForCausalLM.jacobi_forward_greedy_multiblock = jacobi_forward_greedy_multiblock

# ---------------------------
# Load dataset (first 100)
# ---------------------------
df = pd.read_parquet("/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
df_size = len(df)
print(f"Loaded HumanEval dataset with {df_size} samples")
records = df.to_dict(orient="records")

# ---------------------------
# Load model/tokenizer once
# ---------------------------
#model_name = "/home/lah003/models/shiftedattn-9-3-coder-7B-ntok16_soft_ce_oci_datav1_59k_stp_ar_10_cyclic_prog_noise_all_lr1e-6"
#model_name = "/home/lah003/models/yc-blk32-10k"

#model_name = "/home/lah003/models/progressive_noise_cllm2_mask_1m_steps"
#model_name = "/home/lah003/models/0915_w16_blk32_cllm_progressive_21k"

#model_name = "/home/lah003/models/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6/ckpt-212000"
model_name = "/raid/lah003/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6/ckpt-344092"

#model_name = "/home/lah003/models/shiftedattn-10-23-7b-qwen2p5-coder-n16w16-distilln32w16-ar-1-cyclic-noise-all-1e-6/ckpt_218000"
#model_name = "/home/lah003/models/shiftedattn-10-23-7b-qwen2p5-coder-n16w16-distilln32w16-ar-1-cyclic-noise-all-1e-6/ckpt_344092"

#model_name ="/raid/lah003/shiftedattn-11-21-7b-qwen2p5-coder-n16w16-distilln32w16-ar-1-cyclic-noise-all-1e-6/checkpoint-150000"
#model_name = "/raid/lah003/shiftedattn-11-21-7b-qwen2p5-coder-n16w16-distilln64w32-ar-1-cyclic-noise-all-5e-7/checkpoint-150000"

model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/home/lah003/models/Qwen2.5-Coder-7B-Instruct")
model.eval()


eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

print(f"eos id: {eos_id}")
print(f"pad id: {pad_id}")

#alt_eos_id = 151645  # keep your special EOS as a fallback

# ---------------------------
# Generation/profiling config
# ---------------------------
n_token_seq_len = 64

# Safety caps so a sample can't run forever.
max_new_tokens = 1024     # hard cap on total new tokens per prompt
max_calls = 1024          # hard cap on number of diffusion_decoding calls per prompt

# ---------------------------
# Iterate the dataset
# ---------------------------
all_rows = []
t0_overall = time.perf_counter()
all_generations = []

total_gen_only_time = 0

for idx, row in tqdm(enumerate(records)):
    task_id = row.get("task_id", f"idx_{idx}")
#    prompt = "You are given a partially completed Python function with the header and the doc string. Complete the following function according to given information:\n\n" + row["prompt"]

#    prompt = """
#Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
#```python
#{}
#```
#""".strip().format(
#            row["prompt"].strip()
#    )

# currently best performing scheme (best acc + best speed) for distilled ckpt:
#    prompt = """Please continue to complete the function. You are not allowed to modify the given code and please do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
#{}
#""".strip().format(
#            row["prompt"].strip()
#    )

#    prompt = """You are given an incomplete Python code. You are not allowed to modify the given code and please do the completion only. Please return all completed function in a codeblock.
#
#**Function Header and Doc String:**
#```
#{}
#```
#""".strip().format(
#            row["prompt"].strip()
#    )

# currently best performing scheme (best acc + best speed) for trained from original with distilled data ckpt:
    prompt = """Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```
{}
```
""".strip().format(
            row["prompt"].strip()
    )

#    pattern = r'"""(.*?)"""'
#    docstring = re.search(pattern, row["prompt"], re.DOTALL)
#    if not docstring:
#        print(f"WARNING!!! NO DOCSTRING FOUND FOR ENTRY: {idx}")
    
#    prompt = """
#Complete the Python function starting with the following header with docstring. Do not add explaination.
#{}
#""".strip().format(
#            row["prompt"].strip()
#    )

#    prompt = "Respond only in code.\n" + row["prompt"]

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
        hit_eos = (generated_part == eos_id).any().item()
        
        if hit_eos:
            print("HITTING EOS, TERMINATING GENERATION...")
            stop_reason = "eos"
            break
        if total_new_tokens >= max_new_tokens:
            print("EXCEEDING MAX NEW TOKENS COUNT, TERMINATING GENERATION...")
            stop_reason = "max_new_tokens"
            break
        if calls >= max_calls:
            print("EXCEEDING MAX NEW CALLS COUNT, TERMINATING GENERATION...")
            stop_reason = "max_calls"
            break
        
        print(f"\nInit new subsequence {calls}...\n")

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
            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy_multiblock(
                input_ids=prefill_input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                )
            prefill_phase = False
            generated_ids = input_ids
            itr_count = 0
            
            #generated_str = ''.join(tokenizer.batch_decode(prefill_drafted_n_gram, skip_special_tokens=False))
            #print(f'Prefill drafted ngram: {generated_str}')
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
            past_key_values, first_correct_token, accepted_n_gram, itr_count = model.jacobi_forward_greedy_multiblock(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=prefill_phase,
                n_token_seq_len=n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
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

eval_dir = "/home/lah003/data/CLLM2_eval_generations/multiblock_testing_prompt"
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
#save_path = os.path.join(eval_dir, f'sep_n16w16_distilln32_400kdata_multiblock_lookahead_k2_r0p8_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
#save_path = os.path.join(eval_dir, f'oct_n16w16_distilln32w32_344kstps_multiblock_lookahead_k2_r0p8_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'oct_n16w16_distilln32w16_ckpt344092_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'oct_n16w16_distilln64w32_ckpt229000_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
#save_path = os.path.join(eval_dir, f'oct_n16w16_distilln32w32_ckpt344092_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'oct_original_ckpt_distilln32w16_ckpt212000_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
#save_path = os.path.join(eval_dir, f'oct_original_ckpt_distilln32w16_ckpt344092_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_oct_n16w16_distilln32w16_ckpt218000_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_oct_n16w16_distilln32w16_ckpt344092_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_sep_n16w16_distilln32w16_ckpt20k_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_oct_original_ckpt_distilln32w16_ckpt344092_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')


#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_oct_original_ckpt_distilln32w16_ckpt344092_multiblock_lookahead_k2_r0p8_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')
#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_nov_25_ckpt_distilln32w16_ckpt150000_lr_1e-6_multiblock_lookahead_k2_r0p8_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'mod_longer_fastest_prompt_nov_25_ckpt_distilln64w32_ckpt150000_lr_5e-7_multiblock_lookahead_k2_r0p8_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')

#save_path = os.path.join(eval_dir, f'base_path_oct_n16w16_multiblock_lookahead_k2_r0p85_ntok64_lkahead0p0_ngp4_greedy_code_only_prompt_humaneval_w_kv_generation_{model_name.split("/")[-1]}.jsonl')



save_jsonl(original_generations, save_path)

print(f"\n=== All generation done (HumanEval). Results are saved to {save_path} ===")

#### ADDED Lines ####

# ---------------------------
# Aggregate + save
# ---------------------------
t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_profile_humaneval.csv"
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
df_eos.to_csv("diffusion_profile_greedy_humaneval_eos.csv", index=False)
