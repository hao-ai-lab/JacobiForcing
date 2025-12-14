from transformers import AutoTokenizer, Qwen2ForCausalLM
import torch
import math
import time
import pandas as pd
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from modeling.cllm2_qwen2_modeling_two_condition_cache16 import diffusion_forward
Qwen2ForCausalLM.diffusion_forward = diffusion_forward

def make_left_pad_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    is_pad = input_ids == pad_token_id
    first_non_pad_idx = (~is_pad).float().argmax(dim=1)
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    attention_mask = (position_ids >= first_non_pad_idx.unsqueeze(1)).long()
    return attention_mask

def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean()) if s.size and not pd.isna(s).all() else float("nan")

# Dataset (Humaneval-100)
df = pd.read_parquet("/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
records = df.head(100).to_dict(orient="records")

# Model
model_name = "/home/lah003/models/shiftedattn-9-3-coder-7B-ntok16_soft_ce_oci_datav1_59k_stp_ar_10_cyclic_prog_noise_all_lr1e-6"
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("/home/lah003/models/Qwen2.5-7B-Instruct")
tokenizer.padding_side = "left"
model.eval()

eos_id = tokenizer.eos_token_id
alt_eos_id = 151645

# =======================
# Config TODO: make configurable from positional arguments
# =======================
batch_size = 1
n_token_seq_len = 16
temperature = 0.9
top_p = 0.9
top_k = 20
repetition_penalty = 1.2
lenience = 1.0
accept_threshold = 0.1

max_new_tokens = 1280
max_calls = 1024
max_context_len = 1024
# =======================

def _slice_kv(pkv, keep_idx: torch.Tensor):
    assert torch.is_tensor(keep_idx), "expected tensor."
    def _slice(x):
        if x.dim() > 0 and x.size(0) >= int(keep_idx.max().item()) + 1:
            return x.index_select(0, keep_idx.to(x.device))
        else:
            raise ValueError(f"expected tensor with at least {int(keep_idx.max().item()) + 1} elements")
    return _slice(kv)

def _eos_mask_per_sample(input_ids, prompt_lens):
    B, L = input_ids.shape
    device = input_ids.device
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    region = pos >= prompt_lens.unsqueeze(1)
    eos_hit = torch.zeros(B, dtype=torch.bool, device=device)
    if eos_id is not None:
        eos_hit |= ((input_ids == eos_id) & region).any(dim=1)
    if alt_eos_id is not None:
        eos_hit |= ((input_ids == alt_eos_id) & region).any(dim=1)
    return eos_hit

def _prepare_batch(prompts):
    messages = [{"role": "user", "content": p} for p in prompts]
    texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_dict=False)
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_ids = model_inputs["input_ids"]
    attn = make_left_pad_attention_mask(input_ids, tokenizer.pad_token_id).to(model.device)
    return input_ids, attn

@torch.inference_mode()
def run_batch(task_ids, prompts):
    input_ids, attention_mask = _prepare_batch(prompts)
    device = input_ids.device
    B = input_ids.size(0)

    # Per-item stats
    stats = []
    eff_lens = attention_mask.sum(dim=1).to(torch.long)
    for i in range(B):
        stats.append({
            "task_id": task_ids[i],
            "prompt_tokens": int(eff_lens[i].item()),
            "new_tokens": 0,
            "calls": 0,
            "total_iterations": 0,
            "iter_entries": 0,
            "time_start": time.perf_counter(),
            "time_sec": 0.0,
            "toks_per_sec": float("nan"),
            "stop_reason": None,
        })

    prompt_lens = eff_lens.clone()
    prev_eff_lens = eff_lens.clone()

    prefill_phase = True
    past_key_values = None
    active_to_orig = torch.arange(B, device=device)

    REASONS = {0: "eos", 1: "max_new_tokens", 2: "max_calls", 3: "context_cap"}

    while active_to_orig.numel() > 0:
        # Stop checks for active rows
        eos_hit = _eos_mask_per_sample(input_ids, prompt_lens)
        over_new = (prev_eff_lens - prompt_lens) >= max_new_tokens
        over_calls = torch.tensor([stats[int(active_to_orig[i].item())]["calls"] >= max_calls for i in range(len(active_to_orig))],
                                  device=device, dtype=torch.bool)
        over_ctx = (attention_mask.sum(dim=1) >= max_context_len)

        stop_code = torch.full((input_ids.size(0),), -1, dtype=torch.int8, device=device)
        stop_code = torch.where(eos_hit, torch.tensor(0, dtype=torch.int8, device=device), stop_code)
        stop_code = torch.where((stop_code.eq(-1) & over_new), torch.tensor(1, dtype=torch.int8, device=device), stop_code)
        stop_code = torch.where((stop_code.eq(-1) & over_calls), torch.tensor(2, dtype=torch.int8, device=device), stop_code)
        stop_code = torch.where((stop_code.eq(-1) & over_ctx), torch.tensor(3, dtype=torch.int8, device=device), stop_code)

        # Commit finished rows to stats and drop them
        finished_mask = stop_code.ne(-1)
        if finished_mask.any():
            fin_idx = torch.nonzero(finished_mask, as_tuple=False).squeeze(-1)
            for j in fin_idx.tolist():
                orig = int(active_to_orig[j].item())
                t_now = time.perf_counter()
                stats[orig]["time_sec"] = t_now - stats[orig]["time_start"]
                stats[orig]["stop_reason"] = REASONS[int(stop_code[j].item())]

                ntoks = stats[orig]["new_tokens"]
                stats[orig]["toks_per_sec"] = (ntoks / stats[orig]["time_sec"]) if stats[orig]["time_sec"] > 0 else float("nan")

            keep_mask = ~finished_mask
            if keep_mask.sum() == 0:
                break
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            input_ids = input_ids.index_select(0, keep_idx)
            attention_mask = attention_mask.index_select(0, keep_idx)
            prompt_lens = prompt_lens.index_select(0, keep_idx)
            prev_eff_lens = prev_eff_lens.index_select(0, keep_idx)
            active_to_orig = active_to_orig.index_select(0, keep_idx)
            past_key_values = _slice_kv(past_key_values, keep_idx)

        if prefill_phase:
            ret = model.diffusion_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                prefill_phase=True,
                n_token_seq_len=n_token_seq_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                lenience=lenience,
                accept_threshold=accept_threshold,
                tokenizer=tokenizer,
            )

            _, past_key_values, itr = ret
            generated_ids = input_ids
            prefill_phase = False
        else:
            ret = model.diffusion_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=False,
                n_token_seq_len=n_token_seq_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                lenience=lenience,
                accept_threshold=accept_threshold,
                tokenizer=tokenizer,
            )
            generated_ids, past_key_values, itr = ret

        for k in range(active_to_orig.numel()):
            orig = int(active_to_orig[k].item())
            stats[orig]["calls"] += 1
            if itr is not None:
                stats[orig]["total_iterations"] += (itr + 1)
                stats[orig]["iter_entries"] += 1

        # Update lengths
        new_attn = make_left_pad_attention_mask(generated_ids, tokenizer.pad_token_id).to(model.device)
        new_eff = new_attn.sum(dim=1)
        added = (new_eff - prev_eff_lens).clamp_min(0)
        for k in range(active_to_orig.numel()):
            orig = int(active_to_orig[k].item())
            stats[orig]["new_tokens"] += int(added[k].item())

        prev_eff_lens = new_eff
        input_ids = generated_ids
        attention_mask = new_attn

    for i in range(len(stats)):
        if stats[i]["stop_reason"] is None:
            stats[i]["stop_reason"] = "unknown"
            stats[i]["time_sec"] = time.perf_counter() - stats[i]["time_start"]
            ntoks = stats[i]["new_tokens"]
            stats[i]["toks_per_sec"] = (ntoks / stats[i]["time_sec"]) if stats[i]["time_sec"] > 0 else float("nan")

    rows = []
    for s in stats:
        total_iterations = s["total_iterations"] if s["iter_entries"] > 0 else float("nan")
        avg_iter_per_call = (s["total_iterations"] / s["iter_entries"]) if s["iter_entries"] > 0 else float("nan")
        avg_iter_per_token = (s["total_iterations"] / s["new_tokens"]) if s["iter_entries"] > 0 and s["new_tokens"] > 0 else float("nan")
        rows.append({
            "task_id": s["task_id"],
            "prompt_tokens": s["prompt_tokens"],
            "new_tokens": s["new_tokens"],
            "calls": s["calls"],
            "total_iterations": total_iterations,
            "avg_iter_per_call": avg_iter_per_call,
            "avg_iter_per_token": avg_iter_per_token,
            "time_sec": s["time_sec"],
            "toks_per_sec": s["toks_per_sec"],
            "stop_reason": s["stop_reason"],
        })
    return rows

# Iterate dataset in batches
all_rows = []
t0_overall = time.perf_counter()

def _mk_prompt(row):
    #return ("You are given a partially completed Python function with the header and the doc string. "
    #        "Complete the following function according to given information:\n\n" + row["prompt"])
    prompt = """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```python
{}
```
""".strip().format(
        row["prompt"].strip()
    )
    #return (
    #    "Respond only in code.\n" + row["prompt"]
    #)
    return prompt

for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    task_ids = [r.get("task_id", f"idx_{i+j}") for j, r in enumerate(batch)]
    prompts = [_mk_prompt(r) for r in batch]

    rows = run_batch(task_ids, prompts)
    for j, r in enumerate(rows):
        r["index"] = i + j
        all_rows.append(r)

    done = min(i+batch_size, len(records))

    eos_rows = [r for r in rows if r["stop_reason"] == "eos"]
    avg_iter_call = _safe_mean(pd.Series([r["avg_iter_per_call"] for r in rows]))
    print(f"====[{done}/{len(records)}] batch_eos={len(eos_rows)}/{len(rows)} "
          f"avg_iter/call={avg_iter_call:.2f}====")

t_overall = time.perf_counter() - t0_overall
df_profile = pd.DataFrame(all_rows)
csv_path = "diffusion_bs_kv_profile_humaneval100.csv"
df_profile.to_csv(csv_path, index=False)

df_eos = df_profile[df_profile["stop_reason"] == "eos"].copy()
n_eos = len(df_eos)
n_total = len(df_profile)

print("\n=== Diffusion Decoding Profiling (Humaneval-100, batched) — EOS-only ===")
print(f"Examples (eos): {n_eos} / {n_total}   Total wall time: {t_overall:.2f}s")
print(f"Avg new tokens / prompt: {_safe_mean(df_eos['new_tokens']):.2f}")
print(f"Avg calls / prompt: {_safe_mean(df_eos['calls']):.2f}")
print(f"Avg iterations / call: {_safe_mean(df_eos['avg_iter_per_call']):.2f}")
print(f"Avg iterations / token: {_safe_mean(df_eos['avg_iter_per_token']):.2f}")
print(f"Avg toks/sec: {_safe_mean(df_eos['toks_per_sec']):.2f}")

print("\nStop reasons (all examples):")
print(df_profile["stop_reason"].value_counts())

# Save EOS-only too
df_eos.to_csv("diffusion_bs_kv_profile_humaneval100_eos.csv", index=False)
