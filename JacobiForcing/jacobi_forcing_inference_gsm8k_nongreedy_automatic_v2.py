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
from pathlib import Path
import sys

from datasets import load_dataset

# ---------------------------
# Local path import (match reference script)
# ---------------------------
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_nongreedy import (
    jacobi_forward_nongreedy,
)

Qwen2ForCausalLM.jacobi_forward_nongreedy = jacobi_forward_nongreedy


# ---------------------------
# IO helpers
# ---------------------------
def load_jsonl(file_path: str):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f]


def save_jsonl(data, save_path: str):
    with open(save_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


import re

def extract_boxed_answer(text: str) -> str | None:
    """
    Extract content inside the LAST \\boxed{...} in `text`.
    Handles nested braces by brace counting.
    """
    key = r"\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i = start + len(key)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if out else None


def extract_gsm8k_gold(answer_field: str) -> str | None:
    """
    GSM8K gold `answer` usually ends with: '#### <final_number>'.
    """
    if "####" not in answer_field:
        return None
    return answer_field.split("####")[-1].strip()


def normalize_final(s: str) -> str:
    """
    Normalize predicted / gold answers for robust matching.
    GSM8K final answers should be integers, but we normalize gently.
    """
    s = s.strip()
    # remove common latex wrappers
    s = s.replace("$", "").replace("\\,", "").replace(",", "")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # if it's something like "42." or "42\n", strip trailing punctuation
    s = re.sub(r"[^\w\/\.\-]+$", "", s).strip()

    # prefer extracting the last signed integer if present
    ints = re.findall(r"[-+]?\d+", s)
    if ints:
        return ints[-1].lstrip("+")  # GSM8K expects integer final
    return s


def is_correct_gsm8k(pred_text: str, gold_answer_field: str) -> tuple[bool, str | None, str | None]:
    """
    Returns: (correct, pred_final, gold_final)
    pred_final: from \\boxed{...} if available, else fallback to '####' style, else last integer.
    gold_final: from gold '####'
    """
    gold_raw = extract_gsm8k_gold(gold_answer_field)
    if gold_raw is None:
        return (False, None, None)

    pred_raw = extract_boxed_answer(pred_text)
    if pred_raw is None:
        # fallback: some pipelines enforce #### in model outputs
        if "####" in pred_text:
            pred_raw = pred_text.split("####")[-1].strip()
        else:
            pred_raw = pred_text

    pred_norm = normalize_final(pred_raw)
    gold_norm = normalize_final(gold_raw)
    return (pred_norm == gold_norm, pred_norm, gold_norm)


# ---------------------------
# Prompt (AUTOMATIC PROMPT: 2-stage)
# ---------------------------
def create_automatic_math_prompt(problem_text: str, tokenizer):
    stage1 = """PROBLEM_ANALYSIS: {
    "approach": "algebraic|geometric|analytical|computational|proof-based", 
    "problem_type": "numeric|symbolic|proof|optimization|application",
    "reasoning_chain_steps_length": "short|medium|long"
}"""

    enhanced_input = f"""You will be provided with a math problem, generate both STAGE 1 and STAGE 2 with the following output format:

STAGE 1: {stage1}
STAGE 2: Solve the following math problem step by step, with final answer in \\boxed{{...}}.

{problem_text}
"""

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": enhanced_input}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat_prompt


# ---------------------------
# Load dataset (LOCAL PARQUET via HF datasets)
# ---------------------------
gsm8k_parquet = "/raid/lah003/data/gsm8k/main/test-00000-of-00001.parquet"

ds = load_dataset(
    "parquet",
    data_files={"test": gsm8k_parquet},
    split="test",
)

# GSM8K fields are expected to be: question, answer
required_cols = {"question", "answer"}
missing = required_cols.difference(set(ds.column_names))
if missing:
    raise ValueError(f"Missing required columns {missing}. Found columns: {ds.column_names}")

print(f"Loaded GSM8K test shard with {len(ds)} samples from {gsm8k_parquet}")


# ---------------------------
# Load model/tokenizer
# ---------------------------
#model_name = "/raid/lah003/ckpts/JacobiForcing_Math_7B_v1"
#model_name = "/raid/lah003/ckpts/rl/MATH_dapo_e2e_test_math_20260203_022450_middle50_filtering_reward_scale_seqgroup_lr5e-6_reward_multiplicative_automatic_text_representation_2_classes_cot_p12r12_gsm8k_using_math_model_GOOD_CKPT/checkpoints/iteration_47/hf_checkpoint"
model_name = "/raid/lah003/ckpts/rl/e2e_test_math_20260203_180015/checkpoints/iteration_99/hf_checkpoint"

tokenizer_name = "Qwen/Qwen2.5-Math-7B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model.eval()

eos_id = tokenizer.eos_token_id
alt_eos_id = None  # set to 151645 if you use a secondary EOS
print(f"eos id: {eos_id}")


# ---------------------------
# Generation/profiling config (match reference)
# ---------------------------
n_token_seq_len = 64
draft_len = n_token_seq_len + 1
jacobi_max_iterations = 128

max_new_tokens = 1024
max_calls = 1024

all_rows = []
all_generations = []
total_gen_only_time = 0.0
t0_overall = time.perf_counter()


# ---------------------------
# Run GSM8K
# ---------------------------
PROMPT_TMPL = "Math problem to solve:\n{problem}"
NUM_EXAMPLES = min(2000, len(ds))
ds_run = ds.select(range(NUM_EXAMPLES))

for idx, row in tqdm(list(enumerate(ds_run)), total=NUM_EXAMPLES):
    task_id = f"gsm8k_test_{idx}"

    problem_text = PROMPT_TMPL.format(problem=row["question"])
    text = create_automatic_math_prompt(problem_text, tokenizer)

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

    generated_ids = prompt_ids.clone()
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
            n_token_seq_len=n_token_seq_len,
            jacobi_max_iterations=jacobi_max_iterations,
            temperature=0.5,
            top_p=None,
            top_k=500,
            tokenizer=tokenizer,
            eos_token_id=eos_id,
        )
        gen_only_time += (time.perf_counter() - t_gen_start)

        if accepted_tokens.numel() > 0:
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=-1)
            total_new += int(accepted_tokens.shape[1])
            seed = int(seed_tensor.item())
        else:
            stop_reason = "no_progress"
            break

        calls += 1
        iters.append(int(itr_count))

    dt = time.time() - t_start
    total_iterations = sum(iters) if iters else 0
    toks_per_sec = (total_new / gen_only_time) if gen_only_time > 0 else 0.0
    total_gen_only_time += gen_only_time

    generated_str = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False)
    print(f"\n[{idx}] task_id={task_id}\nGenerated:\n{generated_str}\n")
    all_generations.append(generated_str)

    avg_iter_per_call = (total_iterations / calls) if calls > 0 else 0.0
    avg_iter_per_token = (total_iterations / total_new) if total_new > 0 else 0.0

    correct, pred_final, gold_final = is_correct_gsm8k(generated_str, row["answer"])

    all_rows.append(
        dict(
            index=idx,
            task_id=task_id,
            prompt_tokens=prompt_len,
            new_tokens=total_new,
            calls=calls,
            total_iterations=total_iterations,
            avg_iter_per_call=avg_iter_per_call,
            avg_iter_per_token=avg_iter_per_token,
            time_sec=dt,
            toks_per_sec=toks_per_sec,
            stop_reason=stop_reason,
            pred_final=pred_final,
            gold_final=gold_final,
            correct=int(correct),
        )
    )

    if (idx + 1) % 5 == 0 or (idx + 1) == NUM_EXAMPLES:
        print(
            f"====[{idx+1}/{NUM_EXAMPLES}] task_id={task_id} new_toks={total_new} "
            f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}===="
        )

# ---------------------------
# Save generations + profiling
# ---------------------------
eval_dir = path_root / "CLLM2_eval_generations" / "gsm8k_results"
eval_dir.mkdir(parents=True, exist_ok=True)

out_jsonl = eval_dir / f"gsm8k_automatic_prompt_nongreedy_jacobi_ntok{n_token_seq_len}_{Path(model_name).name}.jsonl"
payload = []
for i in range(NUM_EXAMPLES):
    row = ds_run[i]
    payload.append(
        dict(
            index=i,
            task_id=f"gsm8k_test_{i}",
            question=row.get("question", ""),
            gt_answer=row.get("answer", ""),
            output=all_generations[i] if i < len(all_generations) else "",
        )
    )
save_jsonl(payload, str(out_jsonl))
print(f"\n=== GSM8K generations saved to {out_jsonl} ===")

t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
acc = float(df_profile["correct"].mean()) if len(df_profile) else 0.0
print(f"\n=== GSM8K Accuracy (by extracted final answer) ===\nAccuracy: {acc:.4f} ({df_profile['correct'].sum()}/{len(df_profile)})")

csv_path = eval_dir / "diffusion_profile_gsm8k_automatic_nongreedy.csv"
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

df_eos.to_csv(eval_dir / "diffusion_profile_gsm8k_eos_automatic_nongreedy.csv", index=False)
print(f"\n=== Profiling CSV saved to {csv_path} ===")
