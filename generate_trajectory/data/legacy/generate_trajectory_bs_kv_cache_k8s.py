#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

# UTILS
def load_prompt_list(filename, start=0, end=None):
    with open(filename, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError(f"Expected JSON array in {filename}")
    end = len(prompts) if end is None else min(end, len(prompts))
    return prompts[start:end]

def trim_left_padding(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    input_ids_flat = input_ids[0]
    first_non_pad = (input_ids_flat != pad_token_id).nonzero(as_tuple=True)[0][0].item()
    return input_ids[:, first_non_pad:]

def make_left_pad_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    is_pad = input_ids == pad_token_id
    first_non_pad_idx = (~is_pad).float().argmax(dim=1)
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    return (position_ids >= first_non_pad_idx.unsqueeze(1)).long()

def compute_left_pad_lengths(batch_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    # number of left pads per row
    return (batch_ids != pad_token_id).float().argmax(dim=1)

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

# DECODING (w/ KV CACHE, BATCHED)
@torch.inference_mode()
def get_diffusion_decoding_trajectory(
    model, 
    tokenizer, 
    input_ids, 
    attention_mask, 
    n_token_seq_len,
    temperature=0.9, 
    top_p=0.9, 
    top_k=20, 
    repetition_penalty=1.05,
    accept_threshold=0.9,
    past_key_values=None,            # batched PKV or None
    cached_prefix_lens=None          # LongTensor[B] or None
):
    """
    Returns:
      trajectory: dict[row_idx -> List[Tensor]]     # (same format as before)
      kv_full: past_key_values for final 'out'      # list[(k,v)] with batch size B
      kv_converged: pkv sliced to converged prefix  # list[(k,v)] with batch size B
      converged_prefix_lens: LongTensor[B]          # pad + prompt + accepted
    """
    device = model.device
    B = input_ids.size(0)
    prompt_lens = attention_mask.sum(dim=1)   # counts non-pad positions
    pad_lens = compute_left_pad_lengths(input_ids, tokenizer.pad_token_id)

    trajectory = {i: [] for i in range(B)}
    q_sampled = torch.empty(B, n_token_seq_len, dtype=torch.long, device=device)
    for i in range(B):
        choices = input_ids[i, :prompt_lens[i] + pad_lens[i]].tolist()
        q_sampled[i] = torch.tensor(random.choices(choices, k=n_token_seq_len), device=device)

    # initial 'out' = prompt + sampled proposals
    out = torch.cat([input_ids, q_sampled], dim=1)
    for i in range(B):
        trajectory[i].append(out[i].unsqueeze(0))

    logits_processors = LogitsProcessorList()
    if repetition_penalty and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    total_accepted = torch.zeros(B, dtype=torch.long, device=device)
    itr = 0
    q_sampled_ids = {}

    # Helpers
    def _select_rows_pkv(pkv, sel_idx):
        return [(k[sel_idx], v[sel_idx]) for (k, v) in pkv]

    def _slice_pkv_len(past, upto_len_per_row: torch.Tensor):
        bsz = upto_len_per_row.size(0)
        sliced = []
        for (k, v) in past:
            k_rows, v_rows = [], []
            for b in range(bsz):
                L = int(upto_len_per_row[b].item())
                k_rows.append(k[b:b+1, :, :L, :])
                v_rows.append(v[b:b+1, :, :L, :])
            sliced.append((torch.cat(k_rows, dim=0), torch.cat(v_rows, dim=0)))
        return sliced

    # Main accept loop
    while True:
        unfinished_mask = total_accepted < n_token_seq_len
        if not unfinished_mask.any():
            break

        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)  # indices into current batch
        out_unfin = out[idx_unfin]
        attn_mask_unfin = make_left_pad_attention_mask(out_unfin, tokenizer.pad_token_id)

        # Decide whether we can reuse caches (batched)
        can_use_cache = (past_key_values is not None) and (cached_prefix_lens is not None)
        logits_unfin = None

        if can_use_cache:
            conv_lens_unfin = cached_prefix_lens[idx_unfin]  # [Gu]
            has_cache = conv_lens_unfin > 0
            idx_cached = idx_unfin[has_cache]
            idx_plain  = idx_unfin[~has_cache]
            groups_logits = []
            groups_info = []

            # Group A: cached rows → process only the tails with PKV
            if idx_cached.numel() > 0:
                pkv_cached = _select_rows_pkv(past_key_values, idx_cached)

                # Build tails (batches with right-padding)
                tails, tail_lens = [], []
                max_tail = 0
                for j, row in enumerate(idx_cached.tolist()):
                    Lc = int(cached_prefix_lens[row].item())
                    tail = out[row, Lc:]
                    tails.append(tail)
                    tlen = int(tail.size(0))
                    tail_lens.append(tlen)
                    max_tail = max(max_tail, tlen)

                if max_tail == 0:
                    # Edge case: every cached row has zero tail (already converged) -> craft a 1-token dummy to keep shapes.
                    # These logits won't be consumed since total_accepted will stop the loop.
                    max_tail = 1
                    tails = [torch.full((1,), tokenizer.pad_token_id, device=device, dtype=out.dtype) for _ in tails]
                    tail_lens = [0 for _ in tail_lens]  # remember real lens are 0

                pad_id = tokenizer.pad_token_id
                tails_padded = torch.full((len(tails), max_tail), pad_id, device=device, dtype=out.dtype)
                for i_row, t in enumerate(tails):
                    if t.numel() > 0:
                        tails_padded[i_row, :t.numel()] = t

                # Attention masks with PKV are hairy because some models expect length past+tail.
                # Most HF causal LMs ignore mask when PKV is passed. We omit it to stay batched.
                outputs_cached = model(
                    tails_padded,
                    use_cache=True,
                    past_key_values=pkv_cached
                )
                logits_tail = outputs_cached.logits  # [Gc, max_tail, V]

                # Align back to full length per row so downstream indexing stays identical.
                V = logits_tail.size(-1)
                T_full = out_unfin.size(1)
                logits_cached_full = torch.empty(len(idx_cached), T_full, V, device=device, dtype=logits_tail.dtype)
                logits_cached_full.fill_(-1e9)

                for i_row, row in enumerate(idx_cached.tolist()):
                    tlen = tail_lens[i_row]
                    if tlen > 0:
                        logits_cached_full[i_row, -tlen:, :] = logits_tail[i_row, :tlen, :]
                    else:
                        # keep -inf; nothing to use for this row in this step
                        pass

                groups_logits.append(logits_cached_full)
                groups_info.append((idx_cached, logits_cached_full))

            # Group B: plain rows --> full forward without PKV
            if idx_plain.numel() > 0:
                out_plain = out[idx_plain]
                attn_plain = make_left_pad_attention_mask(out_plain, tokenizer.pad_token_id)
                outputs_plain = model(out_plain, attn_plain, use_cache=False)
                logits_plain = outputs_plain.logits
                groups_logits.append(logits_plain)
                groups_info.append((idx_plain, logits_plain))

            # Stitch back in the original unfinished order
            V = groups_logits[0].size(-1)
            logits_unfin = torch.empty(out_unfin.size(0), out_unfin.size(1), V, device=device, dtype=groups_logits[0].dtype)
            logits_unfin.fill_(-1e9)
            for sel_idx, tens in groups_info:
                # Map sel_idx (global) to local indices inside idx_unfin
                # Build a map from global->local
                mapping = {int(g): i for i, g in enumerate(idx_unfin.tolist())}
                local_rows = torch.tensor([mapping[int(g)] for g in sel_idx.tolist()], device=device)
                logits_unfin[local_rows] = tens

        else:
            # No cache yet → single batched forward
            logits_unfin = model(out_unfin, attn_mask_unfin, use_cache=False).logits

        # Per-row accept logic (vectorized over unfinished rows)
        for n, idx in enumerate(idx_unfin):
            logits_row = logits_unfin[n].unsqueeze(0)  # [1, T, V]
            # starting position (same as original)
            start = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx] - 1
            p_logits = logits_row[:, start:, :]       # [1, >=1, V] (first step is right at the boundary)
            p_score = logits_processors(out[idx].unsqueeze(0), p_logits).unsqueeze(0)
            p_prob = F.softmax(p_score, dim=-1)[:, :, :len(tokenizer)]

            if itr == 0:
                gather_ids = q_sampled[idx].unsqueeze(0).unsqueeze(-1)
            else:
                gather_ids = q_sampled_ids[idx.item()].unsqueeze(0).unsqueeze(-1)
            p = p_prob[:, :-1].gather(-1, gather_ids)  # align like original
            p = rearrange(p, '1 n 1 -> 1 n')

            accepted = find_first_true_index(p < accept_threshold)
            num_accepted = int(accepted[0])
            total_accepted[idx] += num_accepted

            cut = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx]
            mid_state_out = out[idx, :cut]

            sample_additional_token = False
            if num_accepted == 0:
                # sample next token at the boundary
                next_token = torch.multinomial(p_prob[:, num_accepted, :], 1).squeeze(0)
                mid_state_out = torch.cat((mid_state_out, next_token), dim=-1)
                total_accepted[idx] += 1
                sample_additional_token = True

            has_rejected = (mid_state_out.size(0) < out[idx].size(0))
            if has_rejected:
                if sample_additional_token:
                    q_probs = p_prob[:, num_accepted + 1:-1, :]
                else:
                    q_probs = p_prob[:, num_accepted:-1, :]
                q_sampled_id = torch.multinomial(q_probs.squeeze(0), 1).squeeze(1)
                q_sampled_ids[idx.item()] = q_sampled_id
                mid_state_out = torch.cat((mid_state_out, q_sampled_id), dim=-1)
                out[idx] = mid_state_out

            trajectory[idx.item()].append(mid_state_out.unsqueeze(0))
        itr += 1

    # Post-pass: capture KV for final sequences and slice to converged prefix for reuse later
    final_attn_mask = make_left_pad_attention_mask(out, tokenizer.pad_token_id)
    outputs = model(out, final_attn_mask, use_cache=True)
    pkv = outputs.past_key_values  # list of (k,v), each [B, H, T, D]

    converged_prefix_lens = pad_lens + prompt_lens + total_accepted  # [B]
    kv_full = pkv
    kv_converged = _slice_pkv_len(pkv, converged_prefix_lens)

    return trajectory, kv_full, kv_converged, converged_prefix_lens


# -------------------- MAIN LOOP --------------------
def main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len,
         use_labels, data_bos_id, data_eos_id, batch_size, save_path):

    # Parse bucket_{bucket_id} from filename
    m = re.search(r"bucket_(\d+)", filename)
    if m:
        bucket_id = m.group(1)
    else:
        print(f"Warning: Could not parse bucket ID from filename '{filename}'. Using 'unknown'.")
        bucket_id = "unknown"

    data = load_prompt_list(filename, start=int(data_bos_id), end=int(data_eos_id))
    data_eos_id = min(len(data), data_eos_id)
    new_data = []

    for start_idx in tqdm(range(int(data_bos_id), int(data_eos_id), batch_size)):
        end_idx = min(start_idx + batch_size, int(data_eos_id))
        batch_indices = torch.arange(start_idx, end_idx, device=model.device)

        print(f"\nProcessing batch from {start_idx} to {end_idx}...\n")

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": data[i - int(data_bos_id)]}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for i in batch_indices
        ]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = model_inputs["input_ids"]
        iterations = torch.zeros(len(batch_indices), dtype=torch.int, device=model.device)

        # ---- KV across outer iterations (per active row) ----
        cached_pkv = None                 # list[(k,v)] batched or None
        cached_prefix_lens = None         # LongTensor or None

        dict_lst = []
        while True:
            generated_part = input_ids[:, model_inputs["input_ids"].size(1):]
            eos_found = (generated_part == tokenizer.eos_token_id).any(dim=1)
            still_active = ~eos_found
            if still_active.sum() == 0:
                break
            if (iterations[still_active][0] * n_token_seq_len) > max_new_seq_len:
                break

            input_ids_active = input_ids[still_active]
            attn_mask_active = make_left_pad_attention_mask(input_ids_active, tokenizer.pad_token_id)
            batch_indices_active = batch_indices[still_active]
            iterations_active = iterations[still_active]

            # Slice caches to active rows if available
            past_key_values_active = None
            cached_prefix_lens_active = None
            if cached_pkv is not None and cached_prefix_lens is not None:
                sel_idx = still_active.nonzero(as_tuple=False).squeeze(1)
                past_key_values_active = [(k[sel_idx], v[sel_idx]) for (k, v) in cached_pkv]
                cached_prefix_lens_active = cached_prefix_lens[still_active]

            print(f'performing diffusion decoding for iterations: {iterations_active}', flush=True)
            diffusion_trajectory_ids_active, kv_full_active, kv_converged_active, new_prefix_lens_active = get_diffusion_decoding_trajectory(
                model, tokenizer, input_ids_active, attn_mask_active,
                n_token_seq_len=n_token_seq_len, temperature=1.0, top_p=0.9,
                top_k=None, repetition_penalty=None, accept_threshold=0.99,
                past_key_values=past_key_values_active,          # <<< reuse cache (batched)
                cached_prefix_lens=cached_prefix_lens_active
            )
            print(f'finishing diffusion decoding...', flush=True)

            next_input_ids = []
            for n, idx in enumerate(batch_indices_active):
                traj = diffusion_trajectory_ids_active[n]
                dic = {
                    "diffusion_itr_id": f"itr_{iterations_active[n].item()}",
                    "data_id": f"bucket_{bucket_id}_data_{idx.item()}",
                    "prompt_ids": trim_left_padding(input_ids_active[n].unsqueeze(0), tokenizer.pad_token_id).cpu(),
                    "answer_trajectory_ids": [step[0][-n_token_seq_len:].cpu() for step in traj],
                    "teacher_output_ids": trim_left_padding(traj[-1], tokenizer.pad_token_id)[0].cpu()
                }
                iterations_active[n] += 1
                next_input_ids.append(traj[-1][0])
                dict_lst.append(dic)

            input_ids = torch.stack(next_input_ids, dim=0)
            batch_indices = batch_indices_active
            iterations = iterations_active

            # Promote new caches to the whole-batch structures for next outer step
            cached_pkv = kv_converged_active
            cached_prefix_lens = new_prefix_lens_active

        # ---- Group and write out ----
        grouped_by_data_id = defaultdict(list)
        for dic in dict_lst:
            grouped_by_data_id[dic["data_id"]].append(dic)

        for data_id, group in grouped_by_data_id.items():
            best_teacher_output = max(group, key=lambda x: len(x["teacher_output_ids"]))["teacher_output_ids"]
            for dic in group:
                dic["teacher_output_ids"] = best_teacher_output
                dic["prompt_ids"] = dic["prompt_ids"].tolist()
                dic["answer_trajectory_ids"] = [a.tolist() for a in dic["answer_trajectory_ids"]]
                dic["teacher_output_ids"] = dic["teacher_output_ids"].tolist()
                new_data.append(dic)

    print("Diffusion trajectory has been collected.")
    os.makedirs(save_path, exist_ok=True)
    new_file_name = f"{Path(filename).stem}_jacobi_len{n_token_seq_len}_labels_{use_labels}_maxlen{max_new_seq_len}_{data_bos_id}_{data_eos_id}.json"
    new_file_path = os.path.join(save_path, new_file_name)

    with open(new_file_path, "w") as f:
        json.dump(new_data, f)


# ---------------- ENTRY -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_start_id", default=0)
    parser.add_argument("--data_bos_id", default=0)
    parser.add_argument("--data_eos_id", default=40)
    parser.add_argument("--use_labels", action="store_true")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    
    main(args.filename, model, tokenizer, args.n_token_seq_len, args.max_new_seq_len,
         args.use_labels, args.data_bos_id, args.data_eos_id, args.batch_size, args.save_path)
