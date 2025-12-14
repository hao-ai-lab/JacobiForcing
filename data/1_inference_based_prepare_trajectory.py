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

# sampling helpers
# def log(t, eps = 1e-20):
#     return torch.log(t.clamp(min = eps))

# def gumbel_noise(t):
#     noise = torch.zeros_like(t).uniform_(0, 1)
#     return -log(-log(noise))

# def gumbel_sample(t, temperature = 1., dim = -1):
#     return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# def top_k(logits, thres = 0.9):
#     k = math.ceil((1 - thres) * logits.shape[-1])
#     val, ind = torch.topk(logits, k)
#     probs = torch.full_like(logits, float('-inf'))
#     probs.scatter_(-1, ind, val)
#     return probs

# def safe_div(num, den, eps = 1e-10):
#     return num / max(den, eps)

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
    accept_threshold = 0.1,
    ):

    batch, prompt_len, out, device = 1, int(torch.sum(attention_mask[0])), input_ids.clone(), input_ids.device
    seq_lens = torch.full((batch,), prompt_len, device = device, dtype = torch.long)

    ### Initialization draft distribution q(x) with 0-1 distribution from prompt
    trajectory = []
    q_sampled = []
    q_logits_all = []
    for _ in range(n_token_seq_len):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())]).to(dtype=torch.long, device=model.device).unsqueeze(dim=0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((batch, len(tokenizer)), float('-inf'), device=model.device)
        q_logits.scatter_(1, q_sample, 0.0) 
        q_sampled.append(q_sample)
        q_logits_all.append(q_logits)
    q_sampled = torch.cat(q_sampled, dim = 1)
    q_logits_all = torch.stack(q_logits_all, dim = -2)
    q_logits = q_logits_all
    trajectory.append(out)

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
    total_accepted = 0
    itr=0
    while total_accepted < n_token_seq_len:

        ### verify and speculate with larger network within a forward pass
        out_attention_mask = torch.full_like(out, 1).to(model.device)
        logits = model(out, out_attention_mask).logits
        p_logits = logits[:, prompt_len+total_accepted-1:, :]
        # only support bsz=1 now
        p_scores = logits_processors(out, p_logits.squeeze(dim=0)).unsqueeze(dim=0)
        q_scores = logits_processors(out, q_logits.squeeze(dim=0)).unsqueeze(dim=0)

        ### prob and prob of draft distribution (p(x) and q(x))
        p_prob = nn.functional.softmax(p_scores, dim=-1)[:, :, :len(tokenizer)]
        q_prob = nn.functional.softmax(q_scores, dim=-1)[:, :, :len(tokenizer)]

        p, prob_next = p_prob[:, :-1], p_prob[:, -1]

        p = p.gather(-1, q_sampled.unsqueeze(dim=-1))
        q = q_prob.gather(-1, q_sampled.unsqueeze(dim=-1)) * lenience
        
        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)
        threshold = torch.ones_like(q).float() * accept_threshold

        accepted = find_first_true_index(
                (r > (p / q)) | (p < threshold)
            )

        num_accepted = int(accepted[0])
        total_accepted += num_accepted
        out = out[:, :prompt_len+total_accepted]

        has_rejected = (num_accepted < q.shape[1])

        ### sample the additional token to better bound the worst case
        sample_additional_token = False
        if num_accepted == 0: 
            next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
            out = torch.cat((out, next_token), dim = -1)
            total_accepted += 1
            sample_additional_token = True
        elif has_rejected:
            adjusted_prob = F.relu(p_prob[:, num_accepted, :] - q_prob[:, num_accepted, :])
            adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
            prob_next = adjusted_prob
            # if all p_prob < q_prob, prob_next becomes nan, then we do not sample the additional token
            if torch.isnan(prob_next).any():
                pass
            else:
                next_token = torch.multinomial(prob_next, num_samples=1)
                out = torch.cat((out, next_token), dim = -1)
                total_accepted += 1                
                sample_additional_token = True

        if not has_rejected:
            next_token = torch.multinomial(prob_next, num_samples=1)
            out = torch.cat((out, next_token), dim = -1)
            total_accepted += 1
            trajectory.append(out)
            continue

        if has_rejected:
            ### update q(x) with self-speculated p(x) and sample new drafts tokens
            if sample_additional_token:
                q_logits = p_logits[:, num_accepted+1:-1, :]
                q_probs = p_prob[:, num_accepted+1:-1, :]
            else:
                q_logits = p_logits[:, num_accepted:-1, :]
                q_probs = p_prob[:, num_accepted:-1, :]
            q_sampled = torch.multinomial(q_probs.squeeze(dim=0), num_samples=1).reshape(1, -1)
            out = torch.cat((out, q_sampled), dim = -1)
            trajectory.append(out)
            itr+=1
        
    eos_reached = len(torch.where(trajectory[-1] == tokenizer.eos_token_id)[0])>0
    generated_str = ''.join(tokenizer.batch_decode(out[0, prompt_len:], skip_special_tokens=False))
    print(f'Converge in {itr} steps')

    return trajectory, eos_reached, itr

def preprocess_openthoughts2(data, tokenizer):
    
    train_dataset = []
    for i in tqdm(range(len(data))):
        
        d = data[i]
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

        prompt_with_template_ids = tokenizer(prompt_with_template, return_tensors="pt")['input_ids']
        inputs = torch.Tensor(prompt_with_template_ids).unsqueeze(0).to(dtype=torch.int)

        labels = tokenizer(prompt_with_template + d['conversations'][1]["value"], return_tensors="pt")['input_ids'][0]
        labels_ids = torch.concat((labels, torch.tensor([tokenizer.eos_token_id])), dim=-1).to(dtype=torch.int)
        
        train_dataset.append(dict(sources_input_ids=inputs, sources_len=[
            input.ne(tokenizer.pad_token_id).sum().item() for input in inputs], labels_ids=labels_ids))
        
    return train_dataset

def main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len, use_aug, use_labels, data_bos_id, data_eos_id, data_start_id):

    if 'openthoughts2' in filename.lower():
        data = []
        with open(filename, 'r') as f:
            for idx, line in enumerate(f):
                if idx < int(data_bos_id):
                    continue
                if idx > int(data_eos_id):
                    break
                data.append(json.loads(line))
        train_dataset = preprocess_openthoughts2(data, tokenizer)

    counter = 0
    new_data = []

    for i in tqdm(range(int(data_bos_id), int(data_eos_id))):

        idx = i - int(data_bos_id)
        d = train_dataset[idx]
        inputs = torch.Tensor(d['sources_input_ids']).squeeze(dim=0).to(device=model.device, dtype=torch.int)

        itr = 0
        eos_reached = False
        dic_list = []
        iteration_steps_list = []

        while itr * n_token_seq_len < max_new_seq_len and not eos_reached:
        
            dic = {}
            dic['data_id'] = f'data_{i}'
            dic['diffusion_itr_id'] = f'itr_{itr}'
            dic['prompt_ids_len'] = d['sources_len']

            attention_mask = torch.full_like(inputs, 1, dtype=torch.int).to(model.device)
            dic['prompt_ids'] = inputs.tolist()

            diffusion_trajectory_ids, eos_reached, iteration_steps = get_diffusion_decoding_trajectory(
                model, 
                tokenizer, 
                inputs, 
                attention_mask, 
                n_token_seq_len,
                filter_thres=0.9,
                temperature = 1.,
                lenience = 1.
            )
            
            iteration_steps_list.append(iteration_steps)

            dic["answer_trajectory_ids"] = [
                id[0][-n_token_seq_len:].tolist() for id in diffusion_trajectory_ids
            ]

            if use_labels:
                dic['labels_ids'] = d['labels_ids'].tolist()

            inputs = diffusion_trajectory_ids[-1]
            dic['teacher_output_ids'] = inputs[0].tolist()

            dic_list.append(dic)
            itr += 1

            # print(f'Writing counter = {counter}...')
            counter += 1

            # if itr % 5 == 0:
            #     if iteration_steps_list:
            #         avg_steps = sum(iteration_steps_list)/len(iteration_steps_list)
            #         print(f"n-token-seq-len: {n_token_seq_len}; Average converge steps: {avg_steps:.2f}")

        # Select the longest teacher_output_ids as labels from teacher
        if dic_list:
            best_teacher_output = max(dic_list, key=lambda x: len(x['teacher_output_ids']))['teacher_output_ids']

            for dic in dic_list:
                dic['teacher_output_ids'] = best_teacher_output
                new_data.append(dic)
    
        # print('Diffusion trajectory has been collected.')
        save_path = 'data/decode_new_ver_collected_diffusion_trajectory/'    
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
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='cuda',
                torch_dtype=torch.bfloat16, 
                attn_implementation="flash_attention_2"
            )
    tokenizer = AutoTokenizer.from_pretrained("/data/phd/kousiqi/kousiqi/ckpts/OpenThinker2-7B")

    main(filename, model, tokenizer, n_token_seq_len, max_new_seq_len, args.use_aug, args.use_labels, args.data_bos_id, args.data_eos_id, args.data_start_id)
