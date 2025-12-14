import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import random
import argparse
from datasets import load_dataset
import datasets
import transformers
import sqlite3
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import copy

import numpy as np
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm

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

def trim_left_padding(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Trim left-padding (leading pad_token_id) from a [1, n] tensor.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "Expected shape [1, n]"
    
    input_ids_flat = input_ids[0]  # shape [n]
    # 找到第一个不是 pad_token_id 的位置
    non_pad_indices = (input_ids_flat != pad_token_id).nonzero(as_tuple=True)[0]
    
    start_idx = non_pad_indices[0].item()
    trimmed = input_ids[:, start_idx:]  # 保留从第一个非pad开始的部分
    return trimmed


def make_left_pad_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Create an attention mask that only masks out the left-padded tokens,
    assuming left-padding was applied by the tokenizer.

    This function sets the attention mask to 0 for the leading (leftmost)
    consecutive pad_token_id tokens, and 1 elsewhere — including any pad_token_ids
    that may appear later during generation (which should not be masked).

    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len), containing token IDs.
        pad_token_id (int): The ID used for padding tokens.

    Returns:
        torch.Tensor: Attention mask of shape (batch_size, seq_len),
                      with 0s for left padding and 1s elsewhere.
    """
    # Identify padding positions
    is_pad = input_ids == pad_token_id  # [B, L]

    # Find the index of the first non-padding token for each sample
    first_non_pad_idx = (~is_pad).float().argmax(dim=1)  # [B]

    # Create position indices
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, L]

    # Mask positions before the first non-padding token
    attention_mask = (position_ids >= first_non_pad_idx.unsqueeze(1)).long()  # [B, L]
    return attention_mask

def compute_left_pad_lengths(batch_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    first_nonpad_idx = (batch_ids != pad_token_id).float().argmax(dim=1)
    return first_nonpad_idx

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)


@torch.inference_mode()
def get_diffusion_decoding_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    n_token_seq_len,
    temperature = 0.9,
    top_p = 0.9, 
    top_k = 20,
    repetition_penalty = 1.05, 
    lenience = 1.,
    accept_threshold = 0.9,
    ):

    batch, prompt_lens, out, device = input_ids.shape[0], attention_mask.sum(dim=1), input_ids.clone(), input_ids.device
    pad_lens = compute_left_pad_lengths(input_ids, tokenizer.pad_token_id)
    
    ### Initialization draft distribution q(x) with 0-1 distribution from prompt
    trajectory = {i: [] for i in range(batch)}
    q_sampled = torch.empty(batch, n_token_seq_len, dtype=torch.long, device=model.device)
    for i in range(batch):
        choices = input_ids[i, :prompt_lens[i]+pad_lens[i]].tolist()
        q_sampled[i] = torch.tensor(random.choices(choices, k=n_token_seq_len),
                                device=model.device)
    out = torch.cat([out, q_sampled], dim=1)
    for i in range(batch):
        trajectory[i].append(out[i].reshape(1, -1))
    
    ### Initialize LogitsProcessor with GenerationConfig
    logits_processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    
    ### Diffusion decoding
    total_accepted = torch.zeros(batch, dtype=torch.long, device=model.device)
    itr=0
    q_sampled_ids = {}
    while True:
        
        ### 1) Group input by convergence & Use only unconverged samples
        unfinished_mask = total_accepted < n_token_seq_len      # [B] bool
        if not unfinished_mask.any():
            break 

        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
        out_unfin = out[idx_unfin]                        # [B_un, L]

        ### 2）Verify and speculate with larger network within a forward pass
        out_attention_mask_unfin = make_left_pad_attention_mask(out_unfin, tokenizer.pad_token_id).to(model.device)
        logits = model(out_unfin, out_attention_mask_unfin).logits
        
        for n, idx in enumerate(idx_unfin):
            start = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx] - 1
            p_logits = logits[n, start:, :]
            p_score = logits_processors(out[idx].unsqueeze(0), p_logits).unsqueeze(dim=0)

            p_prob = nn.functional.softmax(p_score, dim=-1)[:, :, :len(tokenizer)]
            
            p, prob_next = p_prob[:, :-1], p_prob[:, -1]       

            if itr == 0:
                p = p.gather(-1, q_sampled[idx].unsqueeze(dim=0).unsqueeze(dim=-1))
            else:
                p = p.gather(-1, q_sampled_ids.get(idx.item()).unsqueeze(dim=0).unsqueeze(dim=-1))
                
            p = rearrange(p, '1 n 1 -> 1 n')
 
            accepted = find_first_true_index(p < accept_threshold)
            num_accepted = int(accepted[0])
            total_accepted[idx] += num_accepted
        
            cut = pad_lens[idx] + prompt_lens[idx] + total_accepted[idx]
            mid_state_out = out[idx, :cut]
        
            # Additional sample if necessary
            sample_additional_token = False
            if num_accepted == 0:
                next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1).squeeze(dim=0)
                mid_state_out = torch.cat((mid_state_out, next_token), dim = -1)
                total_accepted[idx] += 1
                sample_additional_token = True
        
            has_rejected = (mid_state_out.shape[0] < out[idx].shape[0])
            if has_rejected:
                ### update q(x) with self-speculated p(x) and sample new drafts tokens
                if sample_additional_token:
                    q_probs = p_prob[:, num_accepted+1:-1, :]
                else:
                    q_probs = p_prob[:, num_accepted:-1, :]
                q_sampled_id = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1).squeeze(dim=0)
                q_sampled_ids[idx.item()] = q_sampled_id
                mid_state_out = torch.cat((mid_state_out, q_sampled_id), dim = -1)
                out[idx] = mid_state_out
                
            trajectory[idx.item()].append(mid_state_out.unsqueeze(dim=0))
            
        itr+=1
    
    return trajectory

def main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len, use_aug, use_labels, data_bos_id, data_eos_id, data_start_id, batch_size):

    if 'openthoughts2' in filename.lower():
        data = []
        with open(filename, 'r') as f:
            for idx, line in enumerate(f):
                if idx < int(data_bos_id):
                    continue
                if idx > int(data_eos_id):
                    break
                data.append(json.loads(line))
    

    counter = 0
    new_data = []
    
    for start_idx in tqdm(range(int(data_bos_id), int(data_eos_id), batch_size)):
        end_idx = min(start_idx + batch_size, int(data_eos_id))
        # batch_indices为样本在整个数据集中的位置
        batch_indices = torch.arange(start_idx, end_idx, device=model.device)

        prompts = []
        for i in batch_indices:
            idx = i - int(data_bos_id)
            d = data[idx]
            prompt = d['conversations'][0]["value"]
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            prompt_with_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt_with_template)

        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,        
            truncation=True,     
        ).to(model.device)

        input_ids = model_inputs["input_ids"]         
        attention_mask = model_inputs["attention_mask"] 
        still_active = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        iterations = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)
            
        dict_lst = []
        while True:
        
            eos_found = []
            # print(f'input_ids.shape: {input_ids.shape}')
            for i in range(input_ids.shape[0]):
                generated_part = input_ids[i, model_inputs["input_ids"].size(1):]
                eos_found.append((generated_part == tokenizer.eos_token_id).any())
        
            eos_found = torch.stack(eos_found)  # shape [B]
            still_active = ~eos_found  
            # print(f'still_active.shape: {still_active.shape}')
        
            if still_active.sum() == 0:
                break
        
            # 从仍活跃的样本中提取 input_ids, attention_mask, batch_indices, iteration_active(当前样本执行了几次Jacobi forward)
            input_ids_active = input_ids[still_active]
            attention_mask_active = make_left_pad_attention_mask(input_ids_active, tokenizer.pad_token_id).to(model.device)
            batch_indices_active = batch_indices[still_active]
            iterations_active = iterations[still_active]
        
            if iterations_active[0] * n_token_seq_len > max_new_seq_len:
                print(f'Total length exceeds {max_new_seq_len}. Exit...')
                break
        
            # 执行 diffusion decoding
            # diffusion_trajectory_ids_active: [ [trajectory_ids_for_data_i] ,...,]
            diffusion_trajectory_ids_active = get_diffusion_decoding_trajectory(
                model,
                tokenizer,
                input_ids=input_ids_active,
                attention_mask=attention_mask_active,
                n_token_seq_len=n_token_seq_len,
                temperature=1.0,
                top_p=0.9, 
                top_k=None,
                repetition_penalty=None, 
                lenience=1.0,
                accept_threshold=0.99,
            ) 

            # 收集此次forward后的所有数据的trajectory
            input_ids = []
            for n in range(input_ids_active.shape[0]):
                diffusion_trajectory_ids = diffusion_trajectory_ids_active[n]
                dict = {}
                dict['diffusion_itr_id'] = f'itr_{iterations_active[n].item()}'
                iterations_active[n] += 1
                dict['data_id'] = f'data_{batch_indices_active[n].item()}'
                prompt_ids = trim_left_padding(input_ids_active[n].unsqueeze(dim=0), tokenizer.pad_token_id)
                dict['prompt_ids'] = prompt_ids.tolist()
                dict["answer_trajectory_ids"] = [
                    id[0][-n_token_seq_len:].tolist() for id in diffusion_trajectory_ids
                ]
                teacher_output_ids = trim_left_padding(diffusion_trajectory_ids[-1], tokenizer.pad_token_id)
                dict['teacher_output_ids'] = teacher_output_ids[0].tolist()
                input_ids.append(diffusion_trajectory_ids[-1][0])
                dict_lst.append(dict)

            input_ids = torch.stack(input_ids, dim=0)
            batch_indices = batch_indices_active
            iterations = iterations_active
            print(f'Iterations: {iterations}')

            # generated_str = ''.join(tokenizer.batch_decode(input_ids, skip_special_tokens=False))
            # print(generated_str) 

        # Select the longest teacher_output_ids as labels from teacher
        from collections import defaultdict
        
        # Step 1: 按 data_id 分组 dic_list
        grouped_by_data_id = defaultdict(list)
        for dic in dict_lst:
            grouped_by_data_id[dic['data_id']].append(dic)
        
        # Step 2: 每组选出最长的 teacher_output_ids，并统一赋值
        for data_id, group in grouped_by_data_id.items():
            # 找到当前组中 teacher_output_ids 最长的那个
            best_teacher_output = max(group, key=lambda x: len(x['teacher_output_ids']))['teacher_output_ids']
        
            # 把这一组所有 dic 的 teacher_output_ids 替换为最长的
            for dic in group:
                dic['teacher_output_ids'] = best_teacher_output
                new_data.append(dic)
    
        print('Diffusion trajectory has been collected.')
        save_path = 'data/collected_diffusion_trajectory_bs/'    
        new_file_name = f"{filename.lower().split('/')[-1]}_jacobi_n_token_seq_len{n_token_seq_len}_labels{use_labels}_max_seq_len{max_new_seq_len}_{data_bos_id}_{data_eos_id}.json"
        new_file_path = os.path.join(save_path, new_file_name)
    
        # create directory for a path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(new_file_path, 'w') as f_merged:
            # print(f'Updating new data')
            json.dump(new_data, f_merged)  



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str,
                        default="data/raw_data/openthoughts2_1m.json")
    parser.add_argument("--n_token_seq_len", type=int, default=32)
    parser.add_argument("--max_new_seq_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data_start_id", default=0)
    parser.add_argument("--data_bos_id", default=0)
    parser.add_argument("--data_eos_id", default=100)
    parser.add_argument("--use_aug", action='store_true')
    parser.add_argument("--use_labels", action='store_true')
    args = parser.parse_args()
    filename = args.filename
    model_name = args.model
    n_token_seq_len = args.n_token_seq_len
    max_new_seq_len = args.max_new_seq_len
    batch_size = args.batch_size
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='cuda',
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2"
            )
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/kousiqi/kousiqi/ckpts/OpenThinker2-7B")
    tokenizer.padding_side = "left"

    main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len, args.use_aug, args.use_labels, args.data_bos_id, args.data_eos_id, args.data_start_id, args.batch_size)
