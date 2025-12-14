from transformers import Qwen2ForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
import json
from tqdm import tqdm
import time

import os

import pandas as pd

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy
Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy

# ---------------------------
# Load model/tokenizer once
# ---------------------------
model_name = "/home/lah003/models/0915_w16_blk32_cllm_progressive_21k"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/home/lah003/models/Qwen2.5-Coder-7B-Instruct")
model.eval()

eos_id = tokenizer.eos_token_id
# ---------------------------
# Generation/profiling config
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np

n_token_seq_len_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
decode_n_token_seq_len_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
n_repeat = 100
results = {n: [] for n in n_token_seq_len_list}

for n_token_seq_len in n_token_seq_len_list:
    for decode_n_token_seq_len in decode_n_token_seq_len_list:
        times = []
        for _ in range(n_repeat):
            prompt = """Please continue to complete the function."""

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
            input_ids = model_inputs["input_ids"]
            attention_mask = torch.full_like(input_ids, 1, device=model.device)
        
            # per-example stats
            iters = []
            total_new_tokens = 0
            calls = 0
            prev_len = input_ids.shape[1]
            prompt_len = prev_len
            stop_reason = None
            prefill_phase = True
            generated_ids = input_ids
            
            prefill_drafted_n_gram = None
            
            gen_only_time = 0
        
            # prefill phase
            # pass in random-init draft
            q_sampled = []
            for _ in range(n_token_seq_len):
                q_sample = torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                q_sampled.append(q_sample)
            prefill_draft_token_ids = torch.cat(q_sampled, dim=1)  # shape [1, n_token_seq_len]
            
            prefill_input_ids = prefill_draft_token_ids
            
            # `jacobi_forward_greedy` will return iteration result from first iteration
            past_key_values, first_correct_token, prefill_drafted_n_gram, iter_count = model.jacobi_forward_greedy(
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
            itr_count = 0
        
            # generation phase
            # ---- Initialize a draft tail (any tokens work; we'll fix on the first pass).
            # We keep your "random from prompt" init to avoid extra forward passes.
            q_sampled = []
            for _ in range(decode_n_token_seq_len-1):
                q_sample = torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                q_sampled.append(q_sample)
            q_sampled = torch.cat(q_sampled, dim=1)  # shape [1, n_token_seq_len-1]
            input_ids = torch.cat((first_correct_token.view(1,-1), q_sampled),dim=-1)
        
            t_gen_start = time.perf_counter()
            _ = model.forward(
                input_ids=input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            t_gen_time = time.perf_counter() - t_gen_start
            times.append(t_gen_time)
    
        avg_time = np.mean(times)
        results[n_token_seq_len].append(avg_time)


plt.figure(figsize=(8, 6))
for n in n_token_seq_len_list[1:]:
    plt.plot(decode_n_token_seq_len_list[1:], results[n][1:], marker='o', label=f"n_token_seq_len={n}")
    
    plt.xlabel("decode_n_token_seq_len")
    plt.ylabel("Average gen_only_time (s)")
    plt.title(f"Generation Time vs decode_n_token_seq_len (avg over {n_repeat} runs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("decode_profile.png", dpi=300, bbox_inches="tight")


    