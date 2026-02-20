import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    pad: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    jacobi_enabled: bool = True
    
    jacobi_block_len: int = 64
    jacobi_max_blocks: int = 2
    jacobi_spawn_ratio: float = 0.8
    jacobi_lookahead_start_ratio: float = 0.0
    jacobi_n_gram_pool_size: int = 4
    jacobi_max_iterations: int = 128

    def __post_init__(self):
        # Auto-detect DeepSpeed checkpoint directory
        model_path = self.model
        if os.path.isdir(model_path):
            # Check if this is a DeepSpeed checkpoint directory
            checkpoint_dirs = [d for d in os.listdir(model_path)
                              if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_path, d))]
            if checkpoint_dirs:
                # Use the latest checkpoint directory
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
                model_path = os.path.join(model_path, latest_checkpoint)
                print(f"[CONFIG] Using DeepSpeed checkpoint: {model_path}")

        assert os.path.isdir(model_path), f"Model path does not exist: {model_path}"
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(model_path)
        # Ensure torch_dtype is set (some configs may not have it)
        if not hasattr(self.hf_config, 'torch_dtype') or self.hf_config.torch_dtype is None:
            import torch
            self.hf_config.torch_dtype = torch.float16
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
