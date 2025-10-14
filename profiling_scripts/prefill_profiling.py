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
alt_eos_id = 151645  # keep your special EOS as a fallback

# ---------------------------
# Generation/profiling config
# ---------------------------

# Safety caps so a sample can't run forever.
max_new_tokens = 1024     # hard cap on total new tokens per prompt
max_calls = 1024          # hard cap on number of diffusion_decoding calls per prompt

# ---------------------------
# Iterate the dataset
# ---------------------------
all_rows = []
t0_overall = time.perf_counter()
all_generations = []

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

import matplotlib.pyplot as plt

n_token_seq_len_list = [8, 16, 32, 64, 128, 256, 512]
results = {}

num_trials = 100 

for n in n_token_seq_len_list:
    times = []
    for trial in range(num_trials):
        q_sampled = []
        for _ in range(n):
            q_sample = torch.tensor(
                [random.choice(generated_ids[0].tolist())], 
                dtype=torch.long, 
                device=model.device
            ).unsqueeze(0)
            q_sampled.append(q_sample)
        prefill_draft_token_ids = torch.cat(q_sampled, dim=1)  # shape [1, n]

        t_start = time.time()
        _ = model.jacobi_forward_greedy(
            input_ids=prefill_draft_token_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            prefill_phase=True,
            n_token_seq_len=n,
            tokenizer=tokenizer,
            eos_token_id=eos_id,
        )
        t_end = time.time()
        times.append(t_end - t_start)

    avg_time = sum(times) / len(times)
    results[n] = avg_time
    print(f"n_token_seq_len={n}, avg_time={avg_time:.4f}s")

plt.figure(figsize=(6,4))
plt.plot(list(results.keys())[1:], list(results.values())[1:], marker="o")
plt.xlabel("n_token_seq_len")
plt.ylabel("Avg time (s)")
plt.title("Prefill runtime vs prefill tokens length")
plt.grid(True)
plt.savefig("prefill_profile.png", dpi=300, bbox_inches="tight")

