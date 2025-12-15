import os
import re
import time
import json
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_parquet", type=str,
                   default="/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    p.add_argument("--model_name", type=str,
                   default="/home/lah003/models/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--tokenizer_name", type=str,
                   default="/home/lah003/models/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--eval_dir", type=str,
                   default="/home/lah003/data/CLLM2_eval_generations/baselines")
    p.add_argument("--original_jsonl", type=str,
                   default="/home/lah003/data/CLLM2_eval_generations/humaneval_python_example.jsonl")

    # Gen settings
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")

    # Misc
    p.add_argument("--attention_impl", type=str, default="flash_attention_2")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device_map", type=str, default="cuda")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]


def save_jsonl(data, save_path):
    with open(save_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_python_code(text: str) -> str:
    """Extract the first ```python ... ``` code block; fallback to raw text if none."""
    match = re.search(r'```python([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    return text


def main():
    args = parse_args()
    set_seed(args.seed)

    # Dtype mapping
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    df = pd.read_parquet(args.dataset_parquet)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    records = df.to_dict(orient="records")
    print(f"Loaded HumanEval dataset with {len(records)} samples")


    print("Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        attn_implementation=args.attention_impl,
    )
    model.eval()

    # Handle pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645

    os.makedirs(args.eval_dir, exist_ok=True)
    all_rows = []
    all_generations = []

    overall_gen_time = 0.0
    overall_total_tokens = 0

    t0_overall = time.perf_counter()

    with torch.inference_mode():
        for idx, row in tqdm(list(enumerate(records)), total=len(records)):
            task_id = row.get("task_id", f"idx_{idx}")
            prompt = (
                "Please continue to complete the function. You are not allowed to modify the given code and do the completion only. "
                "Please return all completed function in a codeblock. Here is the given code to do completion:\n"
                "```python\n"
                f"{row['prompt'].strip()}\n"
                "```"
            )

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            input_ids = model_inputs["input_ids"]
            prompt_len = input_ids.shape[1]
            
            # ==============================
            # === Generation-only timing ===
            t0 = time.perf_counter()
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=[eos_id, alt_eos_id] if alt_eos_id is not None else eos_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            gen_time = time.perf_counter() - t0
            # ==============================

            # Stats
            new_tokens = int(output_ids.shape[1] - prompt_len)
            total_tokens = new_tokens
            toks_per_sec = (total_tokens / gen_time)

            # Determine stop reason
            generated_part = output_ids[0, prompt_len:]
            hit_eos = False
            if eos_id is not None:
                hit_eos = (generated_part == eos_id).any().item()
            if not hit_eos and alt_eos_id is not None:
                hit_eos = (generated_part == alt_eos_id).any().item()
            stop_reason = "eos" if hit_eos else ("max_new_tokens" if new_tokens >= args.max_new_tokens else "unknown")

            # Decode only the newly generated portion
            gen_str = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=False)
            print(f"Generated answer:\n{gen_str}\n")
            all_generations.append(gen_str)

            all_rows.append({
                "index": idx,
                "task_id": task_id,
                "prompt_tokens": int(prompt_len),
                "new_tokens": int(new_tokens),
                "total_tokens": int(total_tokens),
                "gen_time_sec": float(gen_time),
                "toks_per_sec": float(toks_per_sec),
                "stop_reason": stop_reason,
            })

            overall_gen_time += gen_time
            overall_total_tokens += total_tokens

            if (idx + 1) % 5 == 0 or (idx + 1) == len(records):
                print(f"====[{idx+1}/{len(records)}] task_id={task_id} "
                      f"new_toks={new_tokens} gen_time={gen_time:.2f}s toks/sec={toks_per_sec:.2f} "
                      f"reason={stop_reason}====")
            
            break

    t_overall = time.perf_counter() - t0_overall

    # ---------------------------
    # Save generations as JSONL
    # ---------------------------
    original_generations = load_jsonl(args.original_jsonl)
    if len(original_generations) != len(all_generations):
        print(f"[WARN] original_jsonl has {len(original_generations)} entries, but we produced {len(all_generations)}.")

    for i, original in enumerate(original_generations[:len(all_generations)]):
        original['output'] = all_generations[i]
        code_only = extract_python_code(all_generations[i])
        print(f"Task id: {i}, Extracted answer:\n{code_only}\n")
        original['generation'] = code_only

    ar_save_path = os.path.join(
        args.eval_dir,
        f"ar_code_only_prompt_humaneval_generation_{Path(args.model_name).name}.jsonl"
    )
    save_jsonl(original_generations[:len(all_generations)], ar_save_path)
    print(f"\n=== All AR generations done (HumanEval). Results are saved to {ar_save_path} ===")

    df_profile = pd.DataFrame(all_rows)

    def _safe_mean(series):
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

    df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
    n_eos = len(df_eos)
    n_total = len(df_profile)

    print("\n=== AR Generation Profiling (HumanEval) ===")
    print(f"Examples (eos): {n_eos} / {n_total}   Total gen time: {overall_gen_time:.2f}s (overall wall: {t_overall:.2f}s)")
    print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.2f}")
    print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.2f}")

if __name__ == "__main__":
    main()
