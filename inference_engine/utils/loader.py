import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    if path is None:
        raise ValueError(f"load_model called with path=None. This should not happen.")
    if not os.path.isdir(path):
        raise ValueError(f"load_model: path does not exist or is not a directory: {path}")
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Try safetensors first, then fall back to .bin files
    checkpoint_files = glob(os.path.join(path, "*.safetensors"))
    if not checkpoint_files:
        checkpoint_files = glob(os.path.join(path, "*.bin"))
        print(f"[LOADER] No safetensors found, loading from {len(checkpoint_files)} .bin files")
    else:
        print(f"[LOADER] Loading from {len(checkpoint_files)} safetensors files")
    
    for file in checkpoint_files:
        # Handle both safetensors and PyTorch .bin files
        if file.endswith('.safetensors'):
            with safe_open(file, "pt", "cpu") as f:
                weight_dict = {k: f.get_tensor(k) for k in f.keys()}
        else:  # .bin file
            weight_dict = torch.load(file, map_location='cpu')
        
        for weight_name, weight_tensor in weight_dict.items():
            for k in packed_modules_mapping:
                if k in weight_name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = weight_name.replace(k, v)
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, weight_tensor, shard_id)
                    break
            else:
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight_tensor)
