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

def _next_multiple_of(x, base=16):
    return ((x + base - 1) // base) * base

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

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
    window_size=4,  # NEW: window around n_token_seq_len-1 for split checks
):
    """
    Diffusion-like decoding with dynamic variable-size blocks.

    Changes:
    - Start with a single block of size 2*n_token_seq_len.
    - Before a block converges, monitor (n_token_seq_len-1) ± window_size.
      If the new-block condition is satisfied, split at earliest accepted X in that window,
      shrink current block to size X (exclude X), then extend the new (right) block so that the
      total drafted length reaches the next multiple of 16.
    - Repeat this scheme for the (current last) block.
    """
    batch = 1
    prompt_len = int(torch.sum(attention_mask[0]))
    device = model.device
    out = input_ids.clone()

    # -----------------------------
    # Init draft to length 2*N
    # -----------------------------
    initial_block_size = 2 * n_token_seq_len
    vocab_size = len(tokenizer)

    # Append 2*N draft tokens (random from prompt tokens for parity with original)
    q_sampled_init, q_logits_init = [], []
    for _ in range(initial_block_size):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())],
                                dtype=torch.long, device=device).unsqueeze(0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((1, vocab_size), float("-inf"), device=device)
        q_logits.scatter_(1, q_sample, 0.0)
        q_sampled_init.append(q_sample)
        q_logits_init.append(q_logits)
    q_sampled_init = torch.cat(q_sampled_init, dim=1)            # [1, 2N]
    q_logits_init = torch.stack(q_logits_init, dim=-2)           # [1, 2N, V]

    # -----------------------------
    # Logits processors
    # -----------------------------
    logits_processors = LogitsProcessorList()
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # -----------------------------
    # Variable-size block state
    # -----------------------------
    block_sizes = [initial_block_size]                    # variable lengths per block
    q_logits_all_blocks = [q_logits_init]                # list of [1, L_b, V]
    q_sampled_all_blocks = [q_sampled_init]              # list of [1, L_b]
    total_accepted_all_blocks = [0]                      # per block accepted count
    out_accepted_all_blocks = [torch.empty((1, 0), device=device)]  # per block accepted ids
    iteration_all_blocks = [0]                           # debug
    num_blocks = 1

    def block_starts():
        starts = []
        s = 0
        for L in block_sizes:
            starts.append(s)
            s += L
        return starts

    confidence_of_first_token = []  # track prob of the "next token" at tail
    iter_count = 0                   # global forward-pass counter

    # -----------------------------
    # Main loop
    # -----------------------------
    def all_done():
        return all(total_accepted_all_blocks[b] >= block_sizes[b] for b in range(num_blocks))

    while not all_done():
        # One verify/speculate pass = one global iteration
        out_attention_mask = torch.full_like(out, 1).to(device)
        logits = model(out, out_attention_mask).logits
        iter_count += 1

        starts = block_starts()

        # Track the "prob_next" for the last (current) block for confidence signal
        last_prob_next = None

        for block_id in range(num_blocks):
            Lb = block_sizes[block_id]
            Tb = total_accepted_all_blocks[block_id]
            if Tb >= Lb:
                continue  # this block is already converged

            b_start = starts[block_id]
            # Slice logits for this block: [prompt + accepted-1 ... prompt + b_start + Lb]
            p_logits_per_block = logits[:, prompt_len + b_start + Tb - 1 : prompt_len + b_start + Lb, :]
            q_logits_per_block = q_logits_all_blocks[block_id]

            # Apply processors
            p_scores = logits_processors(out, p_logits_per_block.squeeze(0)).unsqueeze(0)
            q_scores = logits_processors(out, q_logits_per_block.squeeze(0)).unsqueeze(0)

            # Softmax
            p_prob = torch.softmax(p_scores, dim=-1)[:, :, :vocab_size]
            q_prob = torch.softmax(q_scores, dim=-1)[:, :, :vocab_size]

            # Compare to draft
            p, prob_next = p_prob[:, :-1], p_prob[:, -1]  # prob_next used for spawn signal
            p = p.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1))
            q = q_prob.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1)) * lenience
            p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

            r = torch.zeros_like(q).float().uniform_(0, 1)
            threshold = torch.ones_like(q).float() * accept_threshold

            accepted = find_first_true_index((r > (p / q)) | (p < threshold))
            num_accepted = int(accepted[0])

            # Apply acceptances
            total_accepted_all_blocks[block_id] += num_accepted
            out_accepted_all_blocks[block_id] = out[:, prompt_len + b_start :
                                                       prompt_len + b_start + total_accepted_all_blocks[block_id]]

            has_rejected = (num_accepted < q.shape[1])

            # Worst-case extra token when zero accepted
            sample_additional_token = False
            if num_accepted == 0:
                next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
                out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                total_accepted_all_blocks[block_id] += 1
                sample_additional_token = True

            print(f'[global_iter={iter_count}] block={block_id} '
                  f'accepted_total={total_accepted_all_blocks[block_id]} / {Lb}')

            # If this was the last block we touched, remember prob_next for spawn/split signal
            last_prob_next = prob_next

            # Refresh the block's draft tail if there was a rejection
            if has_rejected:
                if sample_additional_token:
                    q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted + 1 : -1, :]
                    q_probs = p_prob[:, num_accepted + 1 : -1, :]
                else:
                    q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted : -1, :]
                    q_probs = p_prob[:, num_accepted : -1, :]
                q_sampled_all_blocks[block_id] = torch.multinomial(q_probs.squeeze(0), num_samples=1).reshape(1, -1)

            iteration_all_blocks[block_id] += 1

        # If *all* blocks are fully accepted, emit one more token and finish
        if all_done():
            # Use last_prob_next from the last visited block
            if last_prob_next is not None:
                next_token = torch.multinomial(last_prob_next, num_samples=1)
                # Append to the final accepted of the last block
                out_accepted_all_blocks[-1] = torch.cat((out_accepted_all_blocks[-1], next_token), dim=-1)
            # Rebuild and return
            out = out[:, :prompt_len]
            for b in range(num_blocks):
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            return out, iter_count

        # -----------------------------
        # Splitting/extension heuristic
        # -----------------------------
        # Track confidence signal (max prob of next token at tail)
        if last_prob_next is not None:
            first_token_confidence = torch.max(last_prob_next[0]).item()
            confidence_of_first_token.append(first_token_confidence)

        min_iteration_thresholding_count = 3
        accumulative_thresholding_count = 2
        spawn_gate = (
            len(confidence_of_first_token) > min_iteration_thresholding_count and
            all(c > confidence_threshold for c in confidence_of_first_token[-accumulative_thresholding_count:])
        )

        # Consider only the current *last* (rightmost, not-yet-converged) block for splitting
        # (This aligns with "track all blocks and repeat" as we advance rightward.)
        last_incomplete = None
        for b in reversed(range(num_blocks)):
            if total_accepted_all_blocks[b] < block_sizes[b]:
                last_incomplete = b
                break

        if spawn_gate and last_incomplete is not None:
            b = last_incomplete
            Lb = block_sizes[b]
            Tb = total_accepted_all_blocks[b]

            # Local window around n_token_seq_len-1 within this block
            center = n_token_seq_len - 1
            w_lo = max(0, center - window_size)
            w_hi = min(Lb - 1, center + window_size)

            # Split only if we've accepted past the earliest index in the window
            if Tb > w_lo:
                # Choose the earliest accepted position within the window
                X_local = w_lo  # "first one"
                X_local = min(X_local, Tb - 1)  # must be < Tb to ensure it's accepted

                # Shrink current block to size X_local (exclude X)
                left_size = X_local  # tokens [0 .. X_local-1]
                right_size = Lb - left_size  # includes original draft from X_local onward

                if left_size <= 0 or right_size <= 0:
                    # Degenerate; skip this iteration's split
                    pass
                else:
                    print(f'----------- Split block {b} at local X={X_local} (Lb={Lb}, Tb={Tb}) '
                          f'at global_iter={iter_count} -----------')

                    # Split q_* and accepted for the current block
                    q_logits_left  = q_logits_all_blocks[b][:, :left_size, :]
                    q_logits_right = q_logits_all_blocks[b][:, left_size:, :]

                    q_samp_left  = q_sampled_all_blocks[b][:, :left_size]
                    q_samp_right = q_sampled_all_blocks[b][:, left_size:]

                    # Accepted redistribution:
                    accept_left = min(total_accepted_all_blocks[b], left_size)
                    accept_right = max(0, total_accepted_all_blocks[b] - left_size)

                    # Slice accepted token ids from 'out' (relative to prompt)
                    starts = block_starts()
                    b_start = starts[b]
                    out_acc_full = out[:, prompt_len + b_start : prompt_len + b_start + total_accepted_all_blocks[b]]
                    out_acc_left = out_acc_full[:, :accept_left]
                    out_acc_right = out_acc_full[:, accept_left:accept_left + accept_right]

                    # Replace current block with left; insert new block (right) after it
                    block_sizes[b] = left_size
                    q_logits_all_blocks[b] = q_logits_left
                    q_sampled_all_blocks[b] = q_samp_left
                    total_accepted_all_blocks[b] = accept_left
                    out_accepted_all_blocks[b] = out_acc_left

                    # Insert right block
                    block_sizes.insert(b + 1, right_size)
                    q_logits_all_blocks.insert(b + 1, q_logits_right)
                    q_sampled_all_blocks.insert(b + 1, q_samp_right)
                    total_accepted_all_blocks.insert(b + 1, accept_right)
                    out_accepted_all_blocks.insert(b + 1, out_acc_right)
                    iteration_all_blocks.insert(b + 1, 0)
                    num_blocks += 1

                    # Now extend the *new right block* so that total drafted length is next multiple of 16
                    total_len_now = sum(block_sizes)
                    target_total = _next_multiple_of(total_len_now, 16)
                    delta_extend = target_total - total_len_now
                    if delta_extend > 0:
                        # Append delta draft tokens to the rightmost block (b+1)
                        extend_tokens, extend_logits = [], []
                        for _ in range(delta_extend):
                            t = torch.tensor([random.choice(input_ids[0].tolist())],
                                             dtype=torch.long, device=device).unsqueeze(0)
                            # Grow global out (we will rebuild anyway below, but keep parity with original behavior)
                            out = torch.cat((out, t), dim=1)
                            lg = torch.full((1, vocab_size), float("-inf"), device=device)
                            lg.scatter_(1, t, 0.0)
                            extend_tokens.append(t)
                            extend_logits.append(lg)
                        if extend_tokens:
                            ext_samp = torch.cat(extend_tokens, dim=1)          # [1, delta]
                            ext_log  = torch.stack(extend_logits, dim=-2)       # [1, delta, V]
                            q_sampled_all_blocks[b + 1] = torch.cat((q_sampled_all_blocks[b + 1], ext_samp), dim=1)
                            q_logits_all_blocks[b + 1]  = torch.cat((q_logits_all_blocks[b + 1],  ext_log), dim=1)
                            block_sizes[b + 1] += delta_extend

                    # Reset the gate tracker after a split
                    confidence_of_first_token = []

        # -----------------------------
        # Rebuild out = prompt | (accepted + draft) for each block
        # -----------------------------
        out = out[:, :prompt_len]
        starts = block_starts()
        for b in range(num_blocks):
            # accepted
            if out_accepted_all_blocks[b].numel() > 0:
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            # draft tail for unfinished block
            if total_accepted_all_blocks[b] < block_sizes[b]:
                out = torch.cat((out, q_sampled_all_blocks[b]), dim=-1)

    return out, iter_count  # safety fallback (should have returned earlier)


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
    model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr1e-6/hf_merged_step_59258"
    tokenizer_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr1e-6/hf_merged_step_59258"

    # Decoding parameters
    n_token_seq_len = 16
    temperature = 0.9
    top_p = 0.9
    top_k = 20
    repetition_penalty = 1.2
    lenience = 1.0
    accept_threshold = 0.1
    confidence_threshold = 0.5

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
