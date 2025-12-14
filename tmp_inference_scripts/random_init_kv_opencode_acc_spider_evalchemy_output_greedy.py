# spider_generate_qwen2.py
import os
import json
import sqlite3
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import time
import random
import pandas as pd

from transformers import Qwen2ForCausalLM, AutoTokenizer

# === import jacobi_forward_greedy ===
from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy
Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy


# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spider_json", default="/home/lah003/data/spider/spider_data/dev.json",
                   help="Path to Spider dev/test JSON (e.g., dev.json).")
    p.add_argument("--db_root", default="/home/lah003/data/spider/spider_data/test_database",
                   help="Root to Spider DBs (each DB under <db_root>/<db_id>/<db_id>.sqlite).")
    p.add_argument("--model_name", default="/home/lah003/models/0911_blcksz32_w32_steps58k",
                   help="Path/name of your Qwen2 model.")
    p.add_argument("--tokenizer_name", default="/home/lah003/models/Qwen2.5-Coder-7B-Instruct",
                   help="Path/name of tokenizer (Qwen2.5-Coder-7B-Instruct recommended).")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--out_suffix", default=None,
                   help="Optional suffix for output filename. Defaults to model dir name.")
    return p.parse_args()


# ---------------------------
# DB schema helpers
# ---------------------------
def read_db_schema(db_path: str):
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table";')
    tables = [t[0] for t in cursor.fetchall()]

    schema = {}
    for tname in tables:
        cursor_t = con.execute(f"SELECT * FROM \"{tname}\"")
        col_names = [desc[0] for desc in cursor_t.description]
        cursor_t.close()
        schema[tname] = col_names

    cursor.close()
    con.close()
    return schema

def schema_to_text(schema: dict):
    parts = []
    for t, cols in schema.items():
        parts.append(f"table named {t} with columns {cols}")
    return "The SQL database has " + ", ".join(parts) + ", "


# ---------------------------
# Prompt & generation
# ---------------------------
def build_prompt(question: str, schema_text: str):
    prefix = ("Could you translate the following question into SQL. "
              "Please only generate SQL, don't include explanation in the answer. ")
    return prefix + schema_text + "Question: " + question


def main():
    args = get_args()

    # Load model/tokenizer once
    print("Loading model/tokenizer...")
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model.eval()

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645  # optional fallback EOS

    # Read Spider data
    print(f"Loading Spider JSON: {args.spider_json}")
    with open(args.spider_json) as f:
        data = json.load(f)

    # Output file (one JSON string per line)
    base = Path(args.spider_json).with_suffix("").name
    tag = args.out_suffix or Path(args.model_name).name
    out_path = f"/home/lah003/data/CLLM2_eval_generations/spider/{base}_with_answer_{tag}.json"
    print(f"Writing answers to: {out_path}")

    answers = []
    all_rows = []
    all_generations = []

    # ---------------------------
    # Diffusion/Jacobi config
    # ---------------------------
    n_token_seq_len = 32
    max_new_tokens = args.max_new_tokens
    max_calls = 32

    t0_overall = time.perf_counter()
    total_gen_only_time = 0.0

    for idx, d in tqdm(enumerate(data), total=len(data), desc="Generating"):
        db_id = d["db_id"]
        question = d["question"]

        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DB not found: {db_path}")

        schema = read_db_schema(db_path)
        schema_text = schema_to_text(schema)
        prompt = build_prompt(question, schema_text)

        # Encode prompt
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = torch.full_like(input_ids, 1, device=model.device)

        # Per-example stats
        iters = []
        total_new = 0
        calls = 0
        prev_len = input_ids.shape[1]
        prompt_len = prev_len
        stop_reason = None
        prefill_phase = True
        generated_ids = input_ids
        prefill_drafted_n_gram = None
        gen_only_time = 0.0

        #print("\nPROMPT:\n", prompt, "\n")

        t_start = time.time()
        while True:
            # EOS checks
            generated_part = generated_ids[0, prompt_len:]
            hit_eos = False
            if eos_id is not None:
                hit_eos = (generated_part == eos_id).any().item()
            if not hit_eos:
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

            if prefill_phase:
                # random-init draft for prefill
                q_sampled = []
                for _ in range(n_token_seq_len):
                    q = torch.tensor([random.choice(generated_ids[0].tolist())],
                                     dtype=torch.long, device=model.device).unsqueeze(0)
                    q_sampled.append(q)
                prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
                prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

                past_key_values, first_correct_token, prefill_drafted_n_gram, itr_count = model.jacobi_forward_greedy(
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
            else:
                if calls == 1:
                    # reuse prefill-produced draft
                    input_ids = prefill_drafted_n_gram
                else:
                    q_sampled = []
                    for _ in range(n_token_seq_len - 1):
                        q = torch.tensor([random.choice(generated_ids[0].tolist())],
                                         dtype=torch.long, device=model.device).unsqueeze(0)
                        q_sampled.append(q)
                    q_sampled = torch.cat(q_sampled, dim=1)
                    input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)

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
                total_new += added
            prev_len = generated_ids.shape[1]

        # subtract prefill token
        total_new -= 1

        # per-example finalize
        dt = time.time() - t_start
        total_iterations = sum(iters)
        avg_iter_per_call = (total_iterations / calls) if calls > 0 else 0.0
        avg_iter_per_token = (total_iterations / max(total_new, 1))
        toks_per_sec = (total_new / gen_only_time) if gen_only_time > 0 else 0.0

        total_gen_only_time += gen_only_time

        generated_str = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()
        print(f"Generated SQL:\n{generated_str}\n")

        answers.append(generated_str)
        all_generations.append(generated_str)

        all_rows.append({
            "index": idx,
            "db_id": db_id,
            "prompt_tokens": prompt_len,
            "new_tokens": total_new,
            "calls": calls,
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": dt,
            "toks_per_sec": toks_per_sec,
            "stop_reason": stop_reason,
        })

        # light progress
        if (idx + 1) % 5 == 0 or (idx + 1) == len(data):
            print(f"====[{idx+1}/{len(data)}] db_id={db_id} new_toks={total_new} "
                  f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}====")

    # ---------------------------
    # Save generations (JSONL-style, one JSON string per line)
    # ---------------------------
    with open(out_path, "w") as f:
        for a in answers:
            json.dump(a, f)
            f.write("\n")

    print(f"\n=== All generation done (Spider). Results saved to {out_path} ===")

    # ---------------------------
    # Aggregate + save (Spider)
    # ---------------------------
    t_overall = time.perf_counter() - t0_overall
    df_profile = pd.DataFrame(all_rows)
    csv_path = f"diffusion_profile_spider_{tag}.csv"
    df_profile.to_csv(csv_path, index=False)
    print(f"Profiling saved to {csv_path}. Total wall time: {t_overall:.4f}s")

    # ---- EOS-only summary (matches HumanEval script) ----
    def _safe_mean(series):
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if s.size and not pd.isna(s).all() else float('nan')

    df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
    n_eos = len(df_eos)
    n_total = len(df_profile)

    print("\n=== Diffusion Decoding Profiling — EOS-only (Spider) ===")
    print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.4f}s")
    print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
    print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
    print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
    print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.4f}")
    print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

    # Optional: overall stop-reason distro
    print("\nStop reasons (all examples):")
    try:
        print(df_profile['stop_reason'].value_counts())
    except Exception:
        pass

    # Optional: save EOS-only rows too
    df_eos_csv = f"diffusion_profile_greedy_spider_eos_{tag}.csv"
    df_eos.to_csv(df_eos_csv, index=False)
    print(f"Saved EOS-only rows to {df_eos_csv}")


if __name__ == "__main__":
    main()
