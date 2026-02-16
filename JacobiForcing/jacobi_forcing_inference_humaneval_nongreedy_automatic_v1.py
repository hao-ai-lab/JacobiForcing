#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch
import random
import pandas as pd
from tqdm import tqdm
import time
import os
import json
import re
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_nongreedy import jacobi_forward_nongreedy
Qwen2ForCausalLM.jacobi_forward_nongreedy = jacobi_forward_nongreedy


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f]


def save_jsonl(data, save_path):
    with open(save_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def extract_python_code(text):
    """Extract the LAST Python code block from text.
    
    For automatic mode, STAGE 1 might contain example code blocks,
    and STAGE 2/3 contains the actual solution - we want the last one.
    """
    matches = re.findall(r"```python([\s\S]*?)```", text)
    if matches:
        return matches[-1].strip()  # Return LAST match
    return text


def create_automatic_prompt(user_input, tokenizer, enable_verbosity_selection=False):
    """
    Create automatic two-stage/three-stage prompt (LLM decides task representation and verbosity).
    
    This follows the automatic mode from the examples folder.
    """
    if enable_verbosity_selection:
        system_instruction = """This is a 3-stage generation task. You MUST follow this exact structure:

STAGE 1 (Outside code block): Output task representation
STAGE 2 (Outside code block): Output verbosity level  
STAGE 3 (Inside code block): Generate Python solution

IMPORTANT: Stages 1 and 2 must be OUTSIDE the ```python code block."""
        
        stage1 = """TASK_ANALYSIS: {"strategy": "iterative|recursive|dynamic_programming|data_structure|mathematical", "decomposition": "single_function|helper_functions|class_based"}"""
        stage2 = """VERBOSITY: 1-5 (1=minimal, 3=standard docs, 5=comprehensive)"""
        
        enhanced_input = f"""{system_instruction}

STRICTLY FOLLOW THE FOLLOWING OUTPUT STRUCTURE:

STAGE 1: {stage1}
STAGE 2: {stage2}
STAGE 3:
```python
...YOUR CODE HERE...
```

Problem:
{user_input}

Now generate your solution following the 3 stages."""
    else:
        system_instruction = """This is a 2-stage generation task. You MUST follow this exact structure:

STAGE 1 (Outside code block): Output task representation
STAGE 2 (Inside code block): Generate Python solution

IMPORTANT: Stage 1 must be OUTSIDE the ```python code block."""
        
        stage1 = """TASK_ANALYSIS: {"strategy": "iterative|recursive|dynamic_programming|data_structure|mathematical", "decomposition": "single_function|helper_functions|class_based"}"""
        
        enhanced_input = f"""{system_instruction}

STRICTLY FOLLOW THE FOLLOWING OUTPUT STRUCTURE:

STAGE 1: {stage1}
STAGE 2:
```python
...YOUR CODE HERE...
```

Problem:
{user_input}

Now generate your solution following the 2 stages."""
    
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": enhanced_input}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    return chat_prompt


# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_parquet("/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
records = df.to_dict(orient="records")
print(f"Loaded HumanEval dataset with {len(records)} samples")

# ---------------------------
# Load model/tokenizer once
# ---------------------------
model_name = "/home/lah003/workspace/inference_engines/Decode-Learning/tmp/e2e_test_20260119_074618/checkpoints/iteration_0/hf_checkpoint"

#model_name = "/raid/lah003/ckpts/JacobiForcing_Coder_7B_v1"

model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model.eval()

eos_id = tokenizer.eos_token_id
alt_eos_id = 151645  # optional secondary EOS, you can keep it

# ---------------------------
# Generation/profiling config
# ---------------------------
n_token_seq_len = 64
draft_len = n_token_seq_len + 1          # [seed] + 64 speculative
jacobi_max_iterations = 128              # per call (chunk) cap inside jacobi_forward_nongreedy

max_new_tokens = 1024
max_calls = 1024

# Automatic mode configuration
enable_verbosity_selection = False  # Set to True to enable 3-stage mode

all_rows = []
all_generations = []
total_gen_only_time = 0.0
t0_overall = time.perf_counter()

for idx, row in tqdm(list(enumerate(records)), total=len(records)):
    task_id = row.get("task_id", f"idx_{idx}")

    prompt = """Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```
{}
```
""".strip().format(row["prompt"].strip())

    # Use automatic prompt generation instead of imperative
    text = create_automatic_prompt(prompt, tokenizer, enable_verbosity_selection=enable_verbosity_selection)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    prompt_ids = model_inputs["input_ids"]  # [1, S]
    prompt_len = int(prompt_ids.shape[1])

    # If prompt is too short, skip safely
    if prompt_len < 2:
        all_generations.append("")
        all_rows.append(
            dict(
                index=idx,
                task_id=task_id,
                prompt_tokens=prompt_len,
                new_tokens=0,
                calls=0,
                total_iterations=0,
                avg_iter_per_call=0.0,
                avg_iter_per_token=0.0,
                time_sec=0.0,
                toks_per_sec=0.0,
                stop_reason="prompt_too_short",
            )
        )
        continue

    # ---------------------------
    # Prefill: cache prompt[:-1], seed = prompt[-1]
    # ---------------------------
    with torch.inference_mode():
        seed = int(prompt_ids[0, -1].item())
        prefill_out = model(input_ids=prompt_ids[:, :-1], use_cache=True)
        past_key_values = prefill_out.past_key_values

    generated_ids = prompt_ids.clone()  # full prompt visible to user
    calls = 0
    iters = []
    total_new = 0
    stop_reason = None
    gen_only_time = 0.0
    t_start = time.time()

    while True:
        # Stop checks (EOS in generated part)
        generated_part = generated_ids[0, prompt_len:]
        hit_eos = False
        if eos_id is not None:
            hit_eos = (generated_part == eos_id).any().item()
        if (not hit_eos) and (alt_eos_id is not None):
            hit_eos = (generated_part == alt_eos_id).any().item()

        if hit_eos:
            stop_reason = "eos"
            break
        if total_new >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if calls >= max_calls:
            stop_reason = "max_calls"
            break

        # Build speculative draft = [seed] + random tokens
        # (You can keep your "base_pool from previous tokens" heuristic)
        base_pool = [t for t in generated_ids[0].tolist() if t not in (eos_id, alt_eos_id)]
        if not base_pool:
            base_pool = [0]

        spec = [random.choice(base_pool) for _ in range(draft_len - 1)]
        draft = torch.tensor([[seed] + spec], dtype=torch.long, device=model.device)

        t_gen_start = time.perf_counter()
        past_key_values, seed_tensor, accepted_tokens, itr_count = model.jacobi_forward_nongreedy(
            input_ids=draft,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            prefill_phase=False,
            n_token_seq_len=n_token_seq_len,           # max tokens to return this call
            jacobi_max_iterations=jacobi_max_iterations,
            temperature=0.5,
            top_p=None,
            top_k=500,
            tokenizer=tokenizer,
            eos_token_id=eos_id,
        )
        gen_only_time += (time.perf_counter() - t_gen_start)

        # accepted_tokens already includes any rejection "bonus" (committed immediately)
        if accepted_tokens.numel() > 0:
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=-1)
            total_new += int(accepted_tokens.shape[1])

            # next seed is the returned seed (== last committed token)
            seed = int(seed_tensor.item())
        else:
            # Extremely defensive: if nothing committed, force a stop to avoid infinite loop
            stop_reason = "no_progress"
            break

        calls += 1
        iters.append(int(itr_count))

    dt = time.time() - t_start
    total_iterations = sum(iters) if iters else 0
    toks_per_sec = (total_new / gen_only_time) if gen_only_time > 0 else 0.0
    total_gen_only_time += gen_only_time

    generated_str = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False)
    print(f"\n[{idx}] task_id={task_id}\nGenerated answers:\n{generated_str}\n")
    all_generations.append(generated_str)

    avg_iter_per_call = (total_iterations / calls) if calls > 0 else 0.0
    avg_iter_per_token = (total_iterations / total_new) if total_new > 0 else 0.0

    all_rows.append(
        {
            "index": idx,
            "task_id": task_id,
            "prompt_tokens": prompt_len,
            "new_tokens": total_new,
            "calls": calls,
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": dt,
            "toks_per_sec": toks_per_sec,
            "stop_reason": stop_reason,
        }
    )

    if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
        print(
            f"====[{idx+1}/{len(records)}] task_id={task_id} new_toks={total_new} "
            f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}===="
        )

# ---------------------------
# Save generations in JSONL (your existing postprocess)
# ---------------------------
eval_dir = "/home/lah003/data/CLLM2_eval_generations/rl_results"
os.makedirs(eval_dir, exist_ok=True)

original_path = os.path.join(eval_dir, "humaneval_python_example.jsonl")
original_generations = load_jsonl(original_path)

# Update outputs (assumes same length/order)
for i, item in enumerate(original_generations):
    if i >= len(all_generations):
        break
    item["output"] = all_generations[i]
    item["generation"] = extract_python_code(all_generations[i])
    print(f"Task idx: {i}, Extracted answer:\n{item['generation']}\n")

save_path = os.path.join(
    eval_dir, f"automatic_prompt_nongreedy_jacobi_ntok64_humaneval_{model_name.split('/')[-1]}.jsonl"
)
save_jsonl(original_generations, save_path)
print(f"\n=== All generation done (HumanEval). Results are saved to {save_path} ===")

# ---------------------------
# Aggregate + save profiling
# ---------------------------
t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_profile_humaneval_automatic.csv"
df_profile.to_csv(csv_path, index=False)

def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()

print("\n=== Jacobi (RS) Profiling — EOS-only (AUTOMATIC MODE) ===")
print(f"Examples (eos): {len(df_eos)} / {len(df_profile)}   Total wall time: {t_overall:.4f}s")
print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.4f}")
print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

print("\nStop reasons (all examples):")
print(df_profile["stop_reason"].value_counts())

df_eos.to_csv("diffusion_profile_rs_humaneval_eos_automatic.csv", index=False)

