#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import random
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# --------------------------------
# Local imports path (unchanged)
# --------------------------------
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

# NOTE: unchanged algorithm; counts *one* iteration per model() forward,
# even with multiple parallel blocks. Returns (out, iter_count).
@torch.inference_mode()
def diffusion_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    n_token_seq_len,
    temperature=0.9,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.05,
    lenience=1.0,
    accept_threshold=0.8,
    confidence_threshold=0.4,
):
    batch, prompt_len, out, device = 1, int(torch.sum(attention_mask[0])), input_ids.clone(), input_ids.device

    # --- init draft distribution q(x) from prompt tokens (0-1 distribution) ---
    q_sampled, q_logits_all = [], []
    for _ in range(n_token_seq_len):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())]).to(dtype=torch.long, device=model.device).unsqueeze(0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((1, len(tokenizer)), float("-inf"), device=model.device)
        q_logits.scatter_(1, q_sample, 0.0)
        q_sampled.append(q_sample); q_logits_all.append(q_logits)
    q_sampled = torch.cat(q_sampled, dim=1)
    q_logits_all = torch.stack(q_logits_all, dim=-2)

    logits_processors = LogitsProcessorList()
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # --- multi-block state ---
    total_accepted_all_blocks = [0]
    q_logits_all_blocks = [q_logits_all]
    q_sampled_all_blocks = [q_sampled]
    out_accepted_all_blocks = [torch.empty((1, 0), device=model.device)]
    iteration_all_blocks = [0]  # kept for per-block debug only
    num_blocks = 1
    confidence_of_first_token = []

    # Global iteration counter: increments ONCE per model forward pass
    iter_count = 0

    while any(t < n_token_seq_len for t in total_accepted_all_blocks):
        # One verify/speculate pass = one global iteration
        out_attention_mask = torch.full_like(out, 1).to(model.device)
        logits = model(out, out_attention_mask).logits
        iter_count += 1

        for block_id in range(num_blocks):
            if total_accepted_all_blocks[block_id] < n_token_seq_len:
                block_position = block_id * n_token_seq_len
                p_logits_per_block = logits[:, prompt_len + total_accepted_all_blocks[block_id] - 1 + block_position :
                                                prompt_len + (block_position + n_token_seq_len), :]
                q_logits_per_block = q_logits_all_blocks[block_id]

                p_scores = logits_processors(out, p_logits_per_block.squeeze(0)).unsqueeze(0)
                q_scores = logits_processors(out, q_logits_per_block.squeeze(0)).unsqueeze(0)

                p_prob = nn.functional.softmax(p_scores, dim=-1)[:, :, :len(tokenizer)]
                q_prob = nn.functional.softmax(q_scores, dim=-1)[:, :, :len(tokenizer)]

                p, prob_next = p_prob[:, :-1], p_prob[:, -1]
                p = p.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1))
                q = q_prob.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1)) * lenience

                p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
                r = torch.zeros_like(q).float().uniform_(0, 1)
                threshold = torch.ones_like(q).float() * accept_threshold

                accepted = find_first_true_index((r > (p / q)) | (p < threshold))
                num_accepted = int(accepted[0])
                total_accepted_all_blocks[block_id] += num_accepted
                out_accepted_all_blocks[block_id] = out[:, prompt_len + block_position :
                                                          prompt_len + block_position + total_accepted_all_blocks[block_id]]

                has_rejected = (num_accepted < q.shape[1])

                # extra token for worst-case bound
                sample_additional_token = False
                if num_accepted == 0:
                    next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
                    out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                    total_accepted_all_blocks[block_id] += 1
                    sample_additional_token = True

                # LOG: show global iteration (not per-block sum)
                print(f'[global_iter={iter_count}] block={block_id} '
                      f'accepted_total={total_accepted_all_blocks[block_id]}')

                if not has_rejected and all(t >= n_token_seq_len for t in total_accepted_all_blocks):
                    next_token = torch.multinomial(prob_next, num_samples=1)
                    out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                    total_accepted_all_blocks[block_id] += 1
                    out = out[:, :prompt_len]
                    for b in range(num_blocks):
                        out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
                    return out, iter_count  # finished; global_iter already counted

                if has_rejected:
                    if sample_additional_token:
                        q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted + 1 : -1, :]
                        q_probs = p_prob[:, num_accepted + 1 : -1, :]
                    else:
                        q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted : -1, :]
                        q_probs = p_prob[:, num_accepted : -1, :]
                    q_sampled_all_blocks[block_id] = torch.multinomial(q_probs.squeeze(0), num_samples=1).reshape(1, -1)

                iteration_all_blocks[block_id] += 1  # debug-only

        # Block spawning heuristic
        first_token_confidence = torch.max(prob_next[0]).item()
        confidence_of_first_token.append(first_token_confidence)

        min_iteration_thresholding_count = 3
        accumulative_thresholding_count = 2
        if len(confidence_of_first_token) > min_iteration_thresholding_count and all(
            c > confidence_threshold for c in confidence_of_first_token[-accumulative_thresholding_count:]
        ):
            num_blocks += 1
            confidence_of_first_token = []
            print(f'----------- New block added (total={num_blocks}) at global_iter={iter_count} -----------')

            q_sampled_new_block, q_logits_new_block_all = [], []
            total_accepted_all_blocks.append(1)  # first token forced top-1

            for step in range(n_token_seq_len):
                if step == 0:
                    top_token = torch.argmax(prob_next[0]).unsqueeze(0).to(dtype=torch.long, device=model.device)
                    out = torch.cat((out, top_token.unsqueeze(0)), dim=1)
                else:
                    q_sample_new_block = torch.tensor([random.choice(input_ids[0].tolist())],
                                                      dtype=torch.long, device=model.device).unsqueeze(0)
                    out = torch.cat((out, q_sample_new_block), dim=1)
                    q_logits_new_block = torch.full((1, len(tokenizer)), float('-inf'), device=model.device)
                    q_logits_new_block.scatter_(1, q_sample_new_block, 0.0)
                    q_sampled_new_block.append(q_sample_new_block)
                    q_logits_new_block_all.append(q_logits_new_block)

            q_sampled_new_block = torch.cat(q_sampled_new_block, dim=1)
            q_logits_new_block_all = torch.stack(q_logits_new_block_all, dim=-2)
            q_logits_all_blocks.append(q_logits_new_block_all)
            q_sampled_all_blocks.append(q_sampled_new_block)
            out_accepted_all_blocks.append(top_token.unsqueeze(0))
            iteration_all_blocks.append(0)

        # Rebuild out: [prompt | accepted/draft for each block...]
        out = out[:, :prompt_len]
        for b in range(num_blocks):
            if out_accepted_all_blocks[b].numel() > 0:
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            if total_accepted_all_blocks[b] < n_token_seq_len:
                out = torch.cat((out, q_sampled_all_blocks[b]), dim=-1)

    return out, iter_count  # fallback


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

def main():
    parser = argparse.ArgumentParser(description="Profile diffusion decoding on first N HumanEval entries.")
    parser.add_argument("--n", type=int, default=5, help="number of HumanEval entries to profile from the top of the parquet")
    args = parser.parse_args()
    n = args.n

    # =========================
    # Config (paths + knobs)
    # =========================
    parquet_path = "/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"

    # Model/tokenizer
    model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr2e-6/hf_merged_step_59258"
    tokenizer_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr2e-6/hf_merged_step_59258"

    # Decoding parameters
    n_token_seq_len = 16
    temperature = 0.9
    top_p = 0.9
    top_k = 20
    repetition_penalty = 1.2
    lenience = 1.0
    accept_threshold = 0.1
    confidence_threshold = 0.9

    # Safety caps
    max_new_tokens = 1024
    max_calls = 1024

    # =========================
    # Load model/tokenizer
    # =========================
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645  # optional alt EOS

    # =========================
    # Load HumanEval (first n)
    # =========================
    df = pd.read_parquet(parquet_path)
    if n <= 0:
        print("n must be positive.")
        sys.exit(1)
    n = min(n, len(df))
    records = df.iloc[:n].to_dict(orient="records")

    # =========================
    # Iterate & profile
    # =========================
    all_rows = []
    t0_overall = time.perf_counter()

    for idx, row in enumerate(records):
        task_id = row.get("task_id", f"idx_{idx}")

        print(f"========== STARTING DECODING FOR NEW RECORD: {task_id} ==========")
        prompt = "Respond only in code.\n" + row["prompt"]

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        input_ids = model_inputs["input_ids"]
        attention_mask = torch.full_like(input_ids, 1).to(model.device)

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
            generated_part = input_ids[0, prompt_len:]
            hit_eos = False
            if eos_id is not None:
                hit_eos = (generated_part == eos_id).any().item()
            if not hit_eos:
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

            # one diffusion step
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
                confidence_threshold=confidence_threshold,
            )

            iters.append(itr_count)
            calls += 1

            added = generated_ids.shape[1] - prev_len
            if added > 0:
                total_new_tokens += added
            prev_len = generated_ids.shape[1]

            input_ids = generated_ids
            attention_mask = torch.full_like(input_ids, 1).to(model.device)

            # (optional) verbose stream:
            generated_str = ''.join(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
            print(generated_str)

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
            print(
                f"====[{idx+1}/{len(records)}] task_id={task_id} new_toks={total_new_tokens} "
                f"calls={calls} avg_iter/call={avg_iter_per_call:.2f} reason={stop_reason}===="
            )

    # =========================
    # Aggregate + save
    # =========================
    t_overall = time.perf_counter() - t0_overall
    df_profile = pd.DataFrame(all_rows)

    csv_path = f"diffusion_profile_humaneval_{n}.csv"
    df_profile.to_csv(csv_path, index=False)

    # Print quick summary (EOS-only)
    df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
    n_eos = len(df_eos)
    n_total = len(df_profile)

    print("\n=== Diffusion Decoding Profiling (HumanEval) — EOS-only ===")
    print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.2f}s")
    print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.2f}")
    print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.2f}")
    print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.2f}")
    print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.2f}")
    print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.2f}")

    print("\nStop reasons (all examples):")
    print(df_profile["stop_reason"].value_counts())

    eos_csv_path = f"diffusion_profile_humaneval_{n}_eos.csv"
    df_eos.to_csv(eos_csv_path, index=False)
    print(f"\nSaved:\n  {csv_path}\n  {eos_csv_path}")

if __name__ == "__main__":
    main()
