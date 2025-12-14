# bird_generate_qwen2.py
import os
import re
import json
import time
import random
import sqlite3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2ForCausalLM, AutoTokenizer

# ---- jacobi_forward_greedy hook (keep this exactly as in your setup)
from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy
Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy


# ---------------------------
# CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="BIRD generation & execution-match eval with Jacobi forward greedy.")
    # model/tokenizer
    p.add_argument("--model_name", default="/home/lah003/models/shiftedattn-9-3-coder-7B-ntok16_soft_ce_oci_datav1_59k_stp_ar_10_cyclic_prog_noise_all_lr1e-6")
    p.add_argument("--tokenizer_name", default="/home/lah003/models/Qwen2.5-Coder-7B-Instruct")

    # jacobi/diffusion config
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--n_token_seq_len", type=int, default=128)
    p.add_argument("--max_calls", type=int, default=32)

    # BIRD dataset + DB root
    p.add_argument("--dataset_path", type=str, default="/home/lah003/data/BIRD-SQL-data-train/data",
                   help="Local JSONL/JSON path (each line is a JSON obj) or HF repo name.")
    p.add_argument("--split", type=str, default="dev", help="HF split (e.g., dev/test). Ignored for JSON/JSONL.")
    p.add_argument("--db_root", type=str, default="/home/lah003/data/train_databases",
                   help="Directory where per-db sqlite files are created.")

    # run/output
    p.add_argument("--limit", type=int, default=100, help="Max examples to run.")
    p.add_argument("--out_suffix", default=None, help="Suffix for output file names (defaults to model dir name).")

    # DB behavior
    p.add_argument("--reset_db_each_time", action="store_true",
                   help="If set, delete and recreate the DB file for each sample.")
    return p.parse_args()


# ---------------------------
# Dataset load (no ragen)
# ---------------------------
import re
import glob

def load_bird_dataset(dataset_path: str, split: str):
    """
    Loads BIRD samples robustly.

    Priority:
      1) Local parquet directory or single parquet file via HF datasets 'parquet' builder
      2) Local .json/.jsonl (handles BOM, JSON array, JSONL)
      3) HF hub dataset name via datasets.load_dataset(dataset_path, split=split)

    Returns: List[dict] with keys: question, SQL, db_id, schema (mapped when needed).
    """
    dataset_path = dataset_path.strip()
    is_local = os.path.exists(dataset_path)

    # --- (A) Local parquet directory OR single parquet file ---
    def _standardize(rec):
        # Map common alt names -> expected keys
        q = rec.get("question") or rec.get("nl_question") or rec.get("text") or ""
        sql = rec.get("SQL") or rec.get("sql") or rec.get("gold_sql") or ""
        dbid = rec.get("db_id") or rec.get("database_id") or rec.get("db") or "unknown"
        schema = rec.get("schema") or rec.get("create_table_sql") or rec.get("schema_sql") or ""
        return {"question": q, "SQL": sql, "db_id": dbid, "schema": schema}

    if is_local and os.path.isdir(dataset_path):
        parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
        if parquet_files:
            ds = load_dataset("parquet", data_files={"train": parquet_files})["train"]
            return [_standardize(x) for x in ds]

    if is_local and dataset_path.lower().endswith(".parquet"):
        ds = load_dataset("parquet", data_files={"train": dataset_path})["train"]
        return [_standardize(x) for x in ds]

    # --- (B) Local JSON/JSONL fallback (your original robust loader) ---
    if is_local and re.search(r"\.jsonl?$", dataset_path, flags=re.IGNORECASE):
        with open(dataset_path, "r", encoding="utf-8-sig") as f:
            content = f.read().lstrip("\ufeff").strip()

        # Try full-file JSON first
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return [_standardize(x) for x in obj]
            if isinstance(obj, dict):
                for key in ("data", "samples", "items", "examples", split):
                    v = obj.get(key)
                    if isinstance(v, list):
                        return [_standardize(x) for x in v]
        except Exception:
            pass  # Not full-file JSON — treat as JSONL

        records = []
        for line_num, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if line.endswith(","):
                line = line[:-1]
            line = line.lstrip("\ufeff")
            try:
                records.append(_standardize(json.loads(line)))
            except Exception as e:
                snippet = raw_line[:120].replace("\n", "\\n")
                raise ValueError(
                    f"Failed to parse JSON at {dataset_path}:{line_num}.\n"
                    f"Snippet: {snippet}\nError: {e}"
                ) from e
        return records

    # --- (C) HF hub dataset name ---
    ds = load_dataset(dataset_path, split=split)
    return [_standardize(x) for x in ds]

# ---------------------------
# Minimal SQL helpers (BIRD-style)
# ---------------------------
CODE_FENCE_RE = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

def extract_sql(text: str) -> str:
    """Prefer ```sql fenced blocks```. If none, try generic fences, else return raw."""
    if not text:
        return ""
    m = CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # generic fences w/o language
    if "```" in text:
        inner = text.split("```", 1)[-1]
        inner = inner.rsplit("```", 1)[0] if "```" in inner else inner
        return inner.strip()
    # strip possible leading 'sql' token
    text = re.sub(r"^\s*sql\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def normalize_sql(sql: str) -> str:
    sql = (sql or "").strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip()

def _db_file(db_root: str, db_id: str) -> str:
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")

def clean_schema(schema_sql: str) -> str:
    # Drop any statements that touch sqlite_sequence
    lines = []
    for stmt in schema_sql.splitlines():
        if "sqlite_sequence" in stmt.lower():
            continue
        lines.append(stmt)
    return "\n".join(lines)

def _ensure_db(db_root: str, db_id: str, schema_sql: str, reset: bool = False) -> str:
    db_dir = os.path.join(db_root, db_id)
    os.makedirs(db_dir, exist_ok=True)
    db_file = _db_file(db_root, db_id)
    if reset and os.path.exists(db_file):
        os.remove(db_file)
    init_new = not os.path.exists(db_file)
    with sqlite3.connect(db_file) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        if init_new:
            if schema_sql:
                safe_sql = clean_schema(schema_sql)
                conn.executescript(safe_sql)
            else:
                raise RuntimeError("Missing schema; cannot initialize database.")
    return db_file

def _sort_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    return sorted(rows, key=lambda row: [x if x is not None else float("-inf") for x in row])

def execute_sql_on_db(db_file: str, sql: str) -> Tuple[bool, Union[List[Tuple[Any, ...]], str]]:
    try:
        with sqlite3.connect(db_file) as conn:
            conn.execute("PRAGMA foreign_keys = OFF;")
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
        return True, _sort_rows(rows)
    except Exception as e:
        return False, str(e)


# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()

    # model/tokenizer
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

    # dataset
    data = load_bird_dataset(args.dataset_path, args.split)
    total_examples = len(data) if args.limit is None else min(args.limit, len(data))
    print(f"Loaded BIRD dataset with {len(data)} samples. Running {total_examples} examples.")

    # outputs
    tag = args.out_suffix or Path(args.model_name).name
    out_jsonl = f"bird_with_answer_{tag}.jsonl"
    csv_path = f"diffusion_profile_bird_{tag}.csv"
    eos_csv_path = f"diffusion_profile_greedy_bird_eos_{tag}.csv"

    # jacobi config
    n_token_seq_len = args.n_token_seq_len
    max_new_tokens = args.max_new_tokens
    max_calls = args.max_calls

    # accumulators
    all_rows = []
    all_generations = []
    t0_overall = time.perf_counter()

    # iterate
    for idx in tqdm(range(total_examples), desc="Generating"):
        sample = data[idx]
        db_id = sample["db_id"]
        question = sample["question"]
        gold_sql = sample["SQL"]
        schema_sql = sample["schema"]

        # prepare DB
        db_file = _ensure_db(args.db_root, db_id, schema_sql, reset=args.reset_db_each_time)

        # prompt
        render_prompt = f"[DB schema:\n{schema_sql}] {question}"
        directive = "Return only the SQL in a ```sql``` code block. Do not include any other text."
        prompt = f"{render_prompt}\n\n{directive}"

        # encode prompt
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = torch.full_like(input_ids, 1, device=model.device)

        # per-example stats
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

        generated_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True).strip()
        generated_sql = normalize_sql(extract_sql(generated_text))

        # evaluate by execution-result match
        gold_ok, gold_res_or_err = execute_sql_on_db(db_file, normalize_sql(gold_sql))
        sub_ok, sub_res_or_err = execute_sql_on_db(db_file, generated_sql)

        if gold_ok and sub_ok:
            success = (gold_res_or_err == sub_res_or_err)
            sql_error_msg = ""
        else:
            success = False
            # prefer submission error
            sql_error_msg = sub_res_or_err if not sub_ok else gold_res_or_err

        reward = 1.0 if success else 0.0

        print(f"Generated SQL:\n```sql\n{generated_sql}\n```\n")
        if not success and sql_error_msg:
            print(f"[SQL error] {sql_error_msg}")
        
        #print(f"[RESULT] success={success} reward={reward}\n")

        all_generations.append({
            "index": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "generated_sql": generated_sql,
            "stop_reason": stop_reason,
            "success": success,
            "reward": reward,
        })

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
            "success": success,
            "reward": reward,
        })

        # light progress
        if (idx + 1) % 5 == 0 or (idx + 1) == total_examples:
            print(f"====[{idx+1}/{total_examples}] db_id={db_id} new_toks={total_new} "
                  f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} "
                  f"reason={stop_reason} success={success}====")

    # ---------------------------
    # Save generations (JSONL)
    # ---------------------------
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in all_generations:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n=== All generation done (BIRD). Results saved to {out_jsonl} ===")

    # ---------------------------
    # Aggregate + save (profiling)
    # ---------------------------
    t_overall = time.perf_counter() - t0_overall
    df_profile = pd.DataFrame(all_rows)
    df_profile.to_csv(csv_path, index=False)
    print(f"Profiling saved to {csv_path}. Total wall time: {t_overall:.4f}s")

    # ---- EOS-only summary (same style as your HumanEval script) ----
    def _safe_mean(series):
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if s.size and not pd.isna(s).all() else float('nan')

    df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
    n_eos = len(df_eos)
    n_total = len(df_profile)

    print("\n=== Diffusion Decoding Profiling — EOS-only (BIRD) ===")
    print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.4f}s")
    print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.4f}")
    print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.4f}")
    print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.4f}")
    print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.4f}")
    print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.4f}")

    # Success summary
    sr_overall = df_profile["success"].mean() if len(df_profile) else float("nan")
    sr_eos = df_eos["success"].mean() if len(df_eos) else float("nan")
    print(f"\nSuccess rate (all): {sr_overall:.4f}")
    print(f"Success rate (eos-only): {sr_eos:.4f}")

    # Stop reasons
    print("\nStop reasons (all examples):")
    try:
        print(df_profile['stop_reason'].value_counts())
    except Exception:
        pass

    # Save EOS-only rows
    df_eos.to_csv(eos_csv_path, index=False)
    print(f"Saved EOS-only rows to {eos_csv_path}")


if __name__ == "__main__":
    main()
