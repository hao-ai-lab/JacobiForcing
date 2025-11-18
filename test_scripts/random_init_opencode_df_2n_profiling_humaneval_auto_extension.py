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

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

def _next_multiple_of(x, base=16):
    return ((x + base - 1) // base) * base

def _next_multiple_of_strict(x, base=16):
    return ((x + base) // base) * base

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
    window_size=4,
    lookahead_size=16,
    min_iteration_thresholding_count=3,
    accumulative_thresholding_count=2,
):
    """
    Diffusion-like decoding with dynamic variable-size blocks.

    - Start with one block of size 2*n_token_seq_len.
    - Monitor region: (current_converged_index Tb + lookahead_size) ± window_size,
      but never beyond 2*n_token_seq_len inside the block.
    - Spawn only if concurrency is guaranteed: split_at > Tb and split_at < Lb.
    - After spawning, extend ONLY the new right block so that
        (left_size + new_right_size) == NEXT multiple of 16 (strictly larger than the current sum),
      initializing the extra tokens by random-from-context ids.
    """
    batch = 1
    prompt_len = int(torch.sum(attention_mask[0]))
    device = model.device
    out = input_ids.clone()

    # Use the model vocab size consistently for logits
    model_vocab_size = int(getattr(model.config, "vocab_size"))

    # Init draft to length 2 * N
    initial_block_size = 2 * n_token_seq_len

    # Append 2 * N draft tokens (random from prompt tokens)
    q_sampled_init, q_logits_init = [], []
    for _ in range(initial_block_size):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())],
                                dtype=torch.long, device=device).unsqueeze(0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((1, model_vocab_size), float("-inf"), device=device)
        q_logits.scatter_(1, q_sample.clamp_max(model_vocab_size - 1), 0.0)
        q_sampled_init.append(q_sample)
        q_logits_init.append(q_logits)
    q_sampled_init = torch.cat(q_sampled_init, dim=1)            # [1, 2N]
    q_logits_init = torch.stack(q_logits_init, dim=-2)           # [1, 2N, V]

    # Logits processors
    logits_processors = LogitsProcessorList()
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # Variable-size block state
    block_sizes = [initial_block_size]                              # variable lengths per block
    q_logits_all_blocks = [q_logits_init]                           # list of [1, L_b, V]
    q_sampled_all_blocks = [q_sampled_init]                         # list of [1, L_b]
    total_accepted_all_blocks = [0]                                 # per block accepted count
    out_accepted_all_blocks = [torch.empty((1, 0), device=device)]  # per block accepted ids
    iteration_all_blocks = [0]                                      # debug
    num_blocks = 1

    def block_starts():
        starts = []
        s = 0
        for L in block_sizes:
            starts.append(s)
            s += L
        return starts

    iter_count = 0
    
    # Track per-iteration window gate results for the current last-incomplete block
    window_gate_history = []
    history_for_block = None

    def all_done():
        nb = len(block_sizes)
        return all(total_accepted_all_blocks[b] >= block_sizes[b] for b in range(nb))

    while not all_done():
        out_attention_mask = torch.full_like(out, 1).to(device)
        logits = model(out, out_attention_mask).logits
        iter_count += 1

        starts = block_starts()

        # Cache per-block top-1 confidences (by position) for this iteration
        block_top1_confs = {}
        last_prob_next = None

        # verify & speculate over each block
        for block_id in range(len(block_sizes)):
            Lb = block_sizes[block_id]
            Tb = total_accepted_all_blocks[block_id]
            if Tb >= Lb:
                continue

            b_start = starts[block_id]
            p_logits_per_block = logits[:, prompt_len + b_start + Tb - 1 : prompt_len + b_start + Lb, :]
            q_logits_per_block = q_logits_all_blocks[block_id]

            p_scores = logits_processors(out, p_logits_per_block.squeeze(0)).unsqueeze(0)
            q_scores = logits_processors(out, q_logits_per_block.squeeze(0)).unsqueeze(0)

            p_prob = torch.softmax(p_scores, dim=-1)
            q_prob = torch.softmax(q_scores, dim=-1)

            last_prob_next = p_prob[:, -1]

            # apply two-condition acceptance rule
            p, prob_next = p_prob[:, :-1], p_prob[:, -1]
            p = p.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1))
            q = q_prob.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(-1)) * lenience
            p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

            r = torch.zeros_like(q).float().uniform_(0, 1)
            threshold = torch.ones_like(q).float() * accept_threshold
            accepted = find_first_true_index((r > (p / q)) | (p < threshold))
            num_accepted = int(accepted[0])

            total_accepted_all_blocks[block_id] += num_accepted
            out_accepted_all_blocks[block_id] = out[:, prompt_len + b_start :
                                                       prompt_len + b_start + total_accepted_all_blocks[block_id]]
            has_rejected = (num_accepted < q.shape[1])

            sample_additional_token = False
            if num_accepted == 0:
                next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
                out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                total_accepted_all_blocks[block_id] += 1
                sample_additional_token = True

            print(f'[global_iter={iter_count}] block={block_id} '
                  f'accepted_total={total_accepted_all_blocks[block_id]} / {Lb}')

            if has_rejected:
                if sample_additional_token:
                    q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted + 1 : -1, :]
                else:
                    q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted : -1, :]
                q_probs = p_prob[:, (num_accepted + 1 if sample_additional_token else num_accepted) : -1, :]
                q_sampled_all_blocks[block_id] = torch.multinomial(q_probs.squeeze(0), num_samples=1).reshape(1, -1)

            iteration_all_blocks[block_id] += 1

            # Cache top-1 confidence per position for this block
            top1_conf_vec = p_prob.max(dim=-1).values.squeeze(0)  # [Lb - Tb + 1]
            block_top1_confs[block_id] = (top1_conf_vec, Tb, Lb)

        # ----- window-based split gating with history -----
        last_incomplete = None
        for b in reversed(range(len(block_sizes))):
            if total_accepted_all_blocks[b] < block_sizes[b]:
                last_incomplete = b
                break

        if last_incomplete is not None:
            # Reset history when the target block changes (e.g., after a split)
            if history_for_block != last_incomplete:
                window_gate_history = []
                history_for_block = last_incomplete

            b = last_incomplete
            Lb = block_sizes[b]
            Tb = total_accepted_all_blocks[b]

            monitor_cap = min(Lb - 1, 2 * n_token_seq_len - 1)
            center = min(Tb + lookahead_size, monitor_cap)

            w_lo = max(Tb + 1, center - window_size)
            w_hi = min(monitor_cap - 1, center + window_size)

            passed_this_iter = False
            split_at_candidate = None

            if (w_lo <= w_hi) and (b in block_top1_confs):
                top1_conf_vec, Tb_cache, Lb_cache = block_top1_confs[b]
                cand_t = torch.arange(w_lo, w_hi + 1, device=device)
                conf_idx = cand_t - Tb + 1  # index 0 in vec corresponds to block pos (Tb - 1)
                valid = (conf_idx >= 0) & (conf_idx < top1_conf_vec.shape[0])

                if valid.any():
                    cand_t = cand_t[valid]
                    conf_idx = conf_idx[valid]
                    cand_conf = top1_conf_vec[conf_idx]
                    mask = cand_conf > confidence_threshold
                    passed_this_iter = bool(mask.any().item())
                    if passed_this_iter:
                        split_at_candidate = int(cand_t[mask][0].item())

            # Record this iteration’s pass & fail
            window_gate_history.append(passed_this_iter)

            # Apply iter counter & consecutive confidence counter
            spawn_gate = (
                len(window_gate_history) >= min_iteration_thresholding_count and
                all(window_gate_history[-accumulative_thresholding_count:])
            )

            if spawn_gate and passed_this_iter and split_at_candidate is not None:
                split_at = split_at_candidate
                if Tb < split_at < Lb:
                    left_size  = split_at
                    right_size = Lb - left_size

                    left_has_draft  = (left_size - Tb) >= 1
                    right_has_draft = right_size >= 1
                    if left_has_draft and right_has_draft:
                        print(f'----------- Split block {b} at split_at={split_at} (Lb={Lb}, Tb={Tb}) '
                              f'for concurrency at global_iter={iter_count} -----------')

                        q_logits_curr = q_logits_all_blocks[b]
                        q_samp_curr   = q_sampled_all_blocks[b]
                        M = q_logits_curr.shape[1]

                        split_q = max(0, min(left_size - Tb, M))

                        q_logits_left  = q_logits_curr[:, :split_q, :]
                        q_logits_right = q_logits_curr[:, split_q:, :]
                        q_samp_left    = q_samp_curr[:, :split_q]
                        q_samp_right   = q_samp_curr[:, split_q:]

                        starts = block_starts()
                        b_start = starts[b]
                        out_acc_full = out[:, prompt_len + b_start : prompt_len + b_start + total_accepted_all_blocks[b]]
                        out_acc_left = out_acc_full[:, :min(total_accepted_all_blocks[b], left_size)]
                        out_acc_right = out_acc_full[:, min(total_accepted_all_blocks[b], left_size):]

                        # Replace current block with left; insert right
                        block_sizes[b] = left_size
                        q_logits_all_blocks[b] = q_logits_left
                        q_sampled_all_blocks[b] = q_samp_left
                        total_accepted_all_blocks[b] = out_acc_left.shape[1]
                        out_accepted_all_blocks[b] = out_acc_left

                        block_sizes.insert(b + 1, right_size)
                        q_logits_all_blocks.insert(b + 1, q_logits_right)
                        q_sampled_all_blocks.insert(b + 1, q_samp_right)
                        total_accepted_all_blocks.insert(b + 1, out_acc_right.shape[1])
                        out_accepted_all_blocks.insert(b + 1, out_acc_right)
                        iteration_all_blocks.insert(b + 1, 0)
                        # no num_blocks++; use len(block_sizes)

                        # Extend only the new right block to next strict multiple of 16
                        right_idx = b + 1
                        current_sum = block_sizes[b] + block_sizes[right_idx]
                        target_sum  = _next_multiple_of_strict(current_sum, 16)
                        delta_extend = target_sum - current_sum

                        if delta_extend > 0:
                            V_right = int(q_logits_all_blocks[right_idx].shape[-1])
                            extend_tokens, extend_logits = [], []
                            for _ in range(delta_extend):
                                t = torch.tensor([random.choice(input_ids[0].tolist())],
                                                 dtype=torch.long, device=device).unsqueeze(0)
                                out = torch.cat((out, t), dim=1)
                                lg = torch.full((1, V_right), float("-inf"), device=device)
                                lg.scatter_(1, t.clamp_max(V_right - 1), 0.0)
                                extend_tokens.append(t)
                                extend_logits.append(lg)
                            if extend_tokens:
                                ext_samp = torch.cat(extend_tokens, dim=1)
                                ext_log  = torch.stack(extend_logits, dim=-2)
                                q_sampled_all_blocks[right_idx] = torch.cat((q_sampled_all_blocks[right_idx], ext_samp), dim=1)
                                q_logits_all_blocks[right_idx]  = torch.cat((q_logits_all_blocks[right_idx],  ext_log), dim=1)
                                block_sizes[right_idx] += delta_extend

                        # Reset gate history after a split
                        window_gate_history = []
                        history_for_block = right_idx  # track the new last-incomplete going forward

        if all_done():
            if last_prob_next is not None:
                next_token = torch.multinomial(last_prob_next, num_samples=1)
                out_accepted_all_blocks[-1] = torch.cat((out_accepted_all_blocks[-1], next_token), dim=-1)
            out = out[:, :prompt_len]
            for b in range(len(block_sizes)):
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            return out, iter_count

        out = out[:, :prompt_len]
        for b in range(len(block_sizes)):
            if out_accepted_all_blocks[b].numel() > 0:
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            if total_accepted_all_blocks[b] < block_sizes[b]:
                out = torch.cat((out, q_sampled_all_blocks[b]), dim=-1)

    return out, iter_count


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

def main():
    parser = argparse.ArgumentParser(description="Profile diffusion decoding on first N HumanEval entries.")
    parser.add_argument("--n", type=int, default=5, help="number of HumanEval entries to profile from the top of the parquet")
    args = parser.parse_args()
    n = args.n

    parquet_path = "/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"

    # Model/tokenizer
    model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr1e-6/hf_merged_step_59258"
    tokenizer_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr1e-6/hf_merged_step_59258"

    # Decoding parameters
    n_token_seq_len = 64
    temperature = 0.9
    top_p = 0.9
    top_k = 20
    repetition_penalty = 1.2
    lenience = 1.0
    accept_threshold = 0.1
    confidence_threshold = 0.8

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

    # Iterate over all records & profile
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

            # TODO: OTPIONAL, make a debug flag for this 
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
