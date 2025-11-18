import os
import torch
from transformers import AutoConfig, Qwen2ForCausalLM
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def normalize_ckpt_keys(state):
    new_state = {}
    changed = 0
    for k, v in state.items():
        new_k = k
        # strip repeatedly in case of 'module.module.' etc.
        while True:
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
                changed += 1
                continue
            if new_k.startswith("_fsdp_wrapped_module."):
                new_k = new_k[len("_fsdp_wrapped_module."):]
                changed += 1
                continue
            break
        new_state[new_k] = v
    print(f"Stripped prefixes on {changed} keys.")
    # sanity check
    if "module.lm_head.weight" in state and "lm_head.weight" not in new_state:
        raise RuntimeError("Expected lm_head.weight after prefix strip, but didn't find it.")
    return new_state

ckpt_parent = "/data/numa0/train-tests/ckpts/v2/shiftedattn-10-16-cllm-qwen2p5-coder-7B-ntok16-distill32_distill_data_v2_ar_1_only_cyclic_progressive_noise_all_lr1e-6/checkpoint-212000"
ckpt_dir = os.path.join(ckpt_parent, "")
tag = "global_step212000"
output_dtype = torch.bfloat16
output_dir = os.path.join(ckpt_dir, "hf_merged_step_212000")
os.makedirs(output_dir, exist_ok=True)

# 1) Reconstruct FP32 state_dict
fp32_state = get_fp32_state_dict_from_zero_checkpoint(ckpt_parent, tag=tag, lazy_mode=False)
print(f"Successfully loaded {len(fp32_state)} tensors from DeepSpeed ZeRO checkpoint.")

# 2) Convert to bfloat16
bf16_state = {k: v.to(dtype=output_dtype) for k, v in fp32_state.items()}
bf16_state = normalize_ckpt_keys(bf16_state)

# 3) init HF model
config = AutoConfig.from_pretrained(ckpt_parent)
model = Qwen2ForCausalLM(config).to(dtype=output_dtype)

model_sd = model.state_dict()
model_keys = set(model_sd.keys())
ckpt_keys = set(bf16_state.keys())

common = sorted(model_keys & ckpt_keys)
missing_before = sorted(model_keys - ckpt_keys)
unexpected_before = sorted(ckpt_keys - model_keys)

shape_mismatches = []
for k in common:
    if model_sd[k].shape != bf16_state[k].shape:
        shape_mismatches.append((k, tuple(model_sd[k].shape), tuple(bf16_state[k].shape)))

def head(items, n=20):
    return items[:n] if len(items) > n else items

print(f"\n=== Key comparison (pre-load) ===")
print(f"Model keys:      {len(model_keys)}")
print(f"Checkpoint keys: {len(ckpt_keys)}")
print(f"Common keys:     {len(common)}")
print(f"Missing keys:    {len(missing_before)}")
print(f"Unexpected keys: {len(unexpected_before)}")
print(f"Shape mismatches:{len(shape_mismatches)}\n")

if missing_before:
    print("• Missing (first few):")
    for k in head(missing_before): print("  -", k)
if unexpected_before:
    print("\n• Unexpected (first few):")
    for k in head(unexpected_before): print("  -", k)
if shape_mismatches:
    print("\n• Shape mismatches (first few):")
    for k, mshape, cshape in head(shape_mismatches):
        print(f"  - {k}: model {mshape} vs ckpt {cshape}")

# Save reports
with open(os.path.join(output_dir, "missing_keys.txt"), "w") as f:
    f.write("\n".join(missing_before))
with open(os.path.join(output_dir, "unexpected_keys.txt"), "w") as f:
    f.write("\n".join(unexpected_before))
with open(os.path.join(output_dir, "shape_mismatches.tsv"), "w") as f:
    for k, mshape, cshape in shape_mismatches:
        f.write(f"{k}\t{mshape}\t{cshape}\n")

# 4) Load with strict=False so we can proceed while inspecting diffs
missing_after, unexpected_after = model.load_state_dict(bf16_state, strict=False)

print(f"\n=== load_state_dict results (PyTorch report) ===")
print(f"Missing:   {len(missing_after)}")
print(f"Unexpected:{len(unexpected_after)}")
if missing_after:
    print("• Missing after load (first few):")
    for k in head(sorted(missing_after)): print("  -", k)
if unexpected_after:
    print("\n• Unexpected after load (first few):")
    for k in head(sorted(unexpected_after)): print("  -", k)

# 5) Save merged model
model.save_pretrained(output_dir, safe_serialization=True)
print(f"\n Merged model saved to: {output_dir}")
print(f" Load with: Qwen2ForCausalLM.from_pretrained('{output_dir}')")
print(f" Full reports: missing_keys.txt, unexpected_keys.txt, shape_mismatches.tsv in {output_dir}")
