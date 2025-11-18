#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import json
import time
import pandas as pd

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

# logits processors
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

# NOTE: keeps your multi-block diffusion decoding, just as in your original
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
    seq_lens = torch.full((batch,), prompt_len, device=device, dtype=torch.long)

    # --- init draft distribution q(x) from prompt tokens (0-1 distribution) ---
    q_sampled = []
    q_logits_all = []
    for _ in range(n_token_seq_len):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())]).to(dtype=torch.long, device=model.device).unsqueeze(dim=0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((batch, len(tokenizer)), float('-inf'), device=model.device)
        q_logits.scatter_(1, q_sample, 0.0)
        q_sampled.append(q_sample)
        q_logits_all.append(q_logits)
    q_sampled = torch.cat(q_sampled, dim=1)
    q_logits_all = torch.stack(q_logits_all, dim=-2)
    q_logits = q_logits_all

    # --- logits processors ---
    logits_processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # --- multi-block state ---
    total_accepted_all_blocks = [0]
    q_logits_all_blocks = [q_logits]
    q_sampled_all_blocks = [q_sampled]
    out_accepted_all_blocks = [torch.empty((batch, 0), device=model.device)]
    iteration_all_blocks = [0]
    num_blocks = 1
    confidence_of_first_token = []

    while any(t < n_token_seq_len for t in total_accepted_all_blocks):

        # verify & speculate
        out_attention_mask = torch.full_like(out, 1).to(model.device)
        logits = model(out, out_attention_mask).logits

        for block_id in range(num_blocks):
            if total_accepted_all_blocks[block_id] < n_token_seq_len:
                block_position = block_id * n_token_seq_len
                p_logits_per_block = logits[:, prompt_len + total_accepted_all_blocks[block_id] - 1 + block_position : prompt_len + (block_position + n_token_seq_len), :]
                q_logits_per_block = q_logits_all_blocks[block_id]

                # bsz=1
                p_scores = logits_processors(out, p_logits_per_block.squeeze(dim=0)).unsqueeze(dim=0)
                q_scores = logits_processors(out, q_logits_per_block.squeeze(dim=0)).unsqueeze(dim=0)

                # probs
                p_prob = nn.functional.softmax(p_scores, dim=-1)[:, :, :len(tokenizer)]
                q_prob = nn.functional.softmax(q_scores, dim=-1)[:, :, :len(tokenizer)]

                p, prob_next = p_prob[:, :-1], p_prob[:, -1]
                p = p.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(dim=-1))
                q = q_prob.gather(-1, q_sampled_all_blocks[block_id].unsqueeze(dim=-1)) * lenience

                p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
                r = torch.zeros_like(q).float().uniform_(0, 1)
                threshold = torch.ones_like(q).float() * accept_threshold

                accepted = find_first_true_index((r > (p / q)) | (p < threshold))
                num_accepted = int(accepted[0])
                total_accepted_all_blocks[block_id] += num_accepted
                out_accepted_all_blocks[block_id] = out[:, prompt_len + block_position : prompt_len + block_position + total_accepted_all_blocks[block_id]]

                has_rejected = (num_accepted < q.shape[1])

                # sample the additional token to better bound worst case
                sample_additional_token = False
                if num_accepted == 0:
                    next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
                    out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                    total_accepted_all_blocks[block_id] += 1
                    sample_additional_token = True

                if not has_rejected and all(t >= n_token_seq_len for t in total_accepted_all_blocks):
                    next_token = torch.multinomial(prob_next, num_samples=1)
                    out_accepted_all_blocks[block_id] = torch.cat((out_accepted_all_blocks[block_id], next_token), dim=-1)
                    total_accepted_all_blocks[block_id] += 1
                    out = out[:, :prompt_len]
                    for b in range(num_blocks):
                        out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
                    return out  # finished

                if has_rejected:
                    # update q(x) with self-speculated p(x)
                    if sample_additional_token:
                        q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted + 1 : -1, :]
                        q_probs = p_prob[:, num_accepted + 1 : -1, :]
                    else:
                        q_logits_all_blocks[block_id] = p_logits_per_block[:, num_accepted : -1, :]
                        q_probs = p_prob[:, num_accepted : -1, :]
                    q_sampled_all_blocks[block_id] = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1)

                iteration_all_blocks[block_id] += 1
                print(f'Block id: {block_id}, Iteration step: {iteration_all_blocks[block_id]}, '
                      f'Accepted tokens: {total_accepted_all_blocks[block_id]}')

        # --- block spawning heuristic based on first-token confidence ---
        first_token_confidence = torch.max(prob_next[0]).item()
        confidence_of_first_token.append(first_token_confidence)

        if len(confidence_of_first_token) > 3 and all(c > confidence_threshold for c in confidence_of_first_token[-2:]):
            num_blocks += 1
            confidence_of_first_token = []
            print(f'-----------New block added. Currently {num_blocks} blocks in decoding-----------')

            q_sampled_new_block = []
            q_logits_new_block_all = []
            total_accepted_all_blocks.append(1)  # first token will be forced top-1

            for step in range(n_token_seq_len):
                if step == 0:
                    top_token = torch.argmax(prob_next[0]).unsqueeze(0).to(dtype=torch.long, device=model.device)
                    out = torch.cat((out, top_token.unsqueeze(0)), dim=1)
                else:
                    q_sample_new_block = torch.tensor(
                        [random.choice(input_ids[0].tolist())],
                        dtype=torch.long, device=model.device
                    ).unsqueeze(0)
                    out = torch.cat((out, q_sample_new_block), dim=1)
                    q_logits_new_block = torch.full((batch, len(tokenizer)), float('-inf'), device=model.device)
                    q_logits_new_block.scatter_(1, q_sample_new_block, 0.0)
                    q_sampled_new_block.append(q_sample_new_block)
                    q_logits_new_block_all.append(q_logits_new_block)

            q_sampled_new_block = torch.cat(q_sampled_new_block, dim=1)
            q_logits_new_block_all = torch.stack(q_logits_new_block_all, dim=-2)
            q_logits_all_blocks.append(q_logits_new_block_all)
            q_sampled_all_blocks.append(q_sampled_new_block)
            out_accepted_all_blocks.append(top_token.unsqueeze(0))
            iteration_all_blocks.append(0)

        # rebuild `out` packing: [prompt | accepted(block0) | draft(block0) || accepted(block1) | draft(block1) || ...]
        out = out[:, :prompt_len]
        for b in range(num_blocks):
            if out_accepted_all_blocks[b].numel() > 0:
                out = torch.cat((out, out_accepted_all_blocks[b]), dim=-1)
            if total_accepted_all_blocks[b] < n_token_seq_len:
                out = torch.cat((out, q_sampled_all_blocks[b]), dim=-1)

    return out  # fallback

# =========================
# Config (paths + knobs)
# =========================
# HumanEval parquet file (like your second script). Adjust if needed.
parquet_path = "/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
sample_idx = 2  # choose which single entry to run for profiling

# Keep your original model path & tokenizer (adjust as needed)
model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr2e-6/hf_merged_step_59258"
tokenizer_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-9-2-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_ratio_all_lr2e-6/hf_merged_step_59258"

# Decoding parameters (mirroring your first script defaults)
n_token_seq_len = 16
temperature = 0.9
top_p = 0.9
top_k = 20
repetition_penalty = 1.2
lenience = 1.0
accept_threshold = 0.2
confidence_threshold = 0.9

# Safety caps so it won’t run forever
max_new_tokens = 1024    # hard cap on total new tokens
max_calls = 10240        # hard cap on diffusion_decoding calls

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
alt_eos_id = 151645  # your special EOS fallback

# =========================
# Load ONE HumanEval prompt
# =========================
df = pd.read_parquet(parquet_path)
if sample_idx < 0 or sample_idx >= len(df):
    raise IndexError(f"sample_idx {sample_idx} out of range (0..{len(df)-1})")

row = df.iloc[sample_idx].to_dict()
task_id = row.get("task_id", f"idx_{sample_idx}")
prompt = (
    "Respond only in code.\n" + row["prompt"]
)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f'--- HumanEval sample: {task_id} (idx={sample_idx}) ---')
print(f'Prompt:\n{text}\n')

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_ids = model_inputs["input_ids"]
attention_mask = torch.full_like(input_ids, 1).to(model.device)

# =========================
# Run generation w/ profiling
# =========================
prompt_len = input_ids.shape[1]
total_new_tokens = 0
calls = 0
prev_len = prompt_len
stop_reason = None

t_start = time.perf_counter()

while True:
    # EOS check
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

    # one diffusion step (multi-block)
    generated_ids = diffusion_decoding(
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

    calls += 1
    added = generated_ids.shape[1] - prev_len
    if added > 0:
        total_new_tokens += added
    prev_len = generated_ids.shape[1]

    input_ids = generated_ids
    attention_mask = torch.full_like(input_ids, 1).to(model.device)

    # (optional) stream text
    generated_str = ''.join(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
    print(generated_str)

# =========================
# Summarize + save one-row CSV
# =========================
dt = time.perf_counter() - t_start
toks_per_sec = (total_new_tokens / dt) if dt > 0 else float("nan")

row_out = {
    "task_id": task_id,
    "prompt_tokens": prompt_len,
    "new_tokens": total_new_tokens,
    "calls": calls,
    "time_sec": dt,
    "toks_per_sec": toks_per_sec,
    "stop_reason": stop_reason,
}

print("\n=== Diffusion Decoding Profiling (HumanEval — single sample) ===")
for k, v in row_out.items():
    print(f"{k}: {v}")

# ---------------------------
# Final decoded response (without special tokens)
# ---------------------------
final_out = [
    output_ids[len(inp_ids):]
    for inp_ids, output_ids in zip(model_inputs.input_ids, input_ids)
]
response = ''.join(tokenizer.batch_decode(final_out, skip_special_tokens=False)[0])
print("\n---------Generated Answer (decoded)----------")
print(response)
