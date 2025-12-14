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
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

import re
from transformers.cache_utils import DynamicCache

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from cllm2_qwen2_modeling_new_cache32 import get_diffusion_decoding_trajectory

Qwen2ForCausalLM.get_diffusion_decoding_trajectory = get_diffusion_decoding_trajectory

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
    return (batch_ids != pad_token_id).float().argmax(dim=1)

def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)

# MAIN LOOP
def main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len,
         use_labels, data_bos_id, data_eos_id, batch_size, save_path):

    # Parse bucket_{bucket_id} from filename
    m = re.search(r"bucket_(\d+)", filename)
    if m:
        bucket_id = m.group(1)
    else:
        print(f"Warning: Could not parse bucket ID from filename '{filename}'. Using 'unknown'.")
        bucket_id = "unknown"
    
    # fixed to 0~25000 to initially load all data
    data = load_prompt_list(filename, start=0, end=25000)
    data_eos_id = min(len(data), int(data_eos_id))
    new_data = []

    for start_idx in tqdm(range(int(data_bos_id), int(data_eos_id), batch_size)):
        end_idx = min(start_idx + batch_size, int(data_eos_id))
        batch_indices = torch.arange(start_idx, end_idx, device=model.device)

        print(f"\nProcessing batch from {start_idx} to {end_idx}...\n")

        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": data[i - int(data_bos_id)]}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for i in batch_indices
        ]

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"] 
        iterations = torch.zeros(len(batch_indices), dtype=torch.int, device=model.device)

        prefill_phase = True
        past_key_values_active = None

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
            
            def _select_batch_in_legacy_pkv(legacy_pkv, keep_idx: torch.Tensor):
                new_layers = []
                for layer in legacy_pkv:
                    pieces = []
                    for t in layer:
                        # each t: [B, n_heads, seq, head_dim]
                        pieces.append(t.index_select(0, keep_idx))
                    new_layers.append(tuple(pieces))
                return tuple(new_layers)

            # update past_key_values_active
            if past_key_values_active:
                keep_idx = still_active.nonzero(as_tuple=False).squeeze(-1)
                legacy = past_key_values_active.to_legacy_cache()
                legacy = _select_batch_in_legacy_pkv(legacy, keep_idx)
                past_key_values_active = DynamicCache.from_legacy_cache(legacy)

            batch_indices_active = batch_indices[still_active]
            iterations_active = iterations[still_active]

            print(f'performing diffusion decoding for iterations: {iterations_active}', flush=True)
            if prefill_phase:
                past_key_values_active = model.get_diffusion_decoding_trajectory(
                    input_ids=input_ids_active,
                    attention_mask=attn_mask_active,
                    past_key_values=past_key_values_active,
                    use_cache=True,
                    prefill_phase=prefill_phase,
                    n_token_seq_len=n_token_seq_len,
                    temperature = 1.0,
                    top_p = 0.9,
                    top_k = None,
                    repetition_penalty = None, 
                    lenience = 1.,
                    accept_threshold = 0.99,
                    tokenizer=tokenizer,
                )
                print(f'finishing prefilling...', flush=True)
                prefill_phase = False
                continue
            else:
                diffusion_trajectory_ids_active, past_key_values_active = model.get_diffusion_decoding_trajectory(
                    input_ids=input_ids_active,
                    attention_mask=attn_mask_active,
                    past_key_values=past_key_values_active,
                    use_cache=True,
                    prefill_phase=prefill_phase,
                    n_token_seq_len=n_token_seq_len,
                    temperature = 1.0,
                    top_p = 0.9,
                    top_k = None,
                    repetition_penalty = None, 
                    lenience = 1.,
                    accept_threshold = 0.99,
                    tokenizer=tokenizer,
                )

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

        print(f'finishing diffusion decoding...', flush=True)
        grouped_by_data_id = defaultdict(list)
        for dic in dict_lst:
            grouped_by_data_id[dic["data_id"]].append(dic)

        for data_id, group in grouped_by_data_id.items():
            best_teacher_output = max(group, key=lambda x: len(x["teacher_output_ids"]))["teacher_output_ids"]
            for dic in group:
                dic["teacher_output_ids"] = best_teacher_output
                # Now convert to list for JSON
                dic["prompt_ids"] = dic["prompt_ids"].tolist()
                dic["answer_trajectory_ids"] = [a.tolist() for a in dic["answer_trajectory_ids"]]
                dic["teacher_output_ids"] = dic["teacher_output_ids"].tolist()
                new_data.append(dic)

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
