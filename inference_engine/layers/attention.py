import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from inference_engine.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def load_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Load KV from paged cache into contiguous tensors (inverse of store_kvcache_kernel)."""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    # Read from cache
    cache_offsets = slot * D + tl.arange(0, D)
    key = tl.load(k_cache_ptr + cache_offsets)
    value = tl.load(v_cache_ptr + cache_offsets)
    # Write to contiguous output
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    tl.store(key_ptr + key_offsets, key)
    tl.store(value_ptr + value_offsets, value)


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, num_heads: int, head_dim: int):
    """Load KV from paged cache into contiguous tensors."""
    N = slot_mapping.numel()
    D = num_heads * head_dim
    key = torch.empty(N, num_heads, head_dim, device=k_cache.device, dtype=k_cache.dtype)
    value = torch.empty(N, num_heads, head_dim, device=v_cache.device, dtype=v_cache.dtype)
    load_kvcache_kernel[(N,)](k_cache, v_cache, key, key.stride(0), value, value.stride(0), slot_mapping, D)
    return key, value


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Jacobi forward: use flash_attn_with_kvcache to match AR decode numerically
        # Detected by: is_prefill=True, block_tables set, cache_seqlens set
        is_jacobi_forward = (context.is_prefill and 
                            context.block_tables is not None and 
                            context.cache_seqlens is not None)
        
        if is_jacobi_forward:
            # Jacobi forward: use flash_attn_with_kvcache (same as AR decode)
            import os
            debug = os.environ.get("ATTN_DEBUG", "0") == "1"
            
            B = context.cache_seqlens.shape[0]
            L = context.seqlen_q
            
            if debug:
                print(f"[ATTN_JACOBI] B={B}, L={L}, cache_seqlens={context.cache_seqlens.tolist()}")
                print(f"[ATTN_JACOBI] q.shape={tuple(q.shape)}, k.shape={tuple(k.shape)}, v.shape={tuple(v.shape)}")
                print(f"[ATTN_JACOBI] slot_mapping[:10]={context.slot_mapping[:10].tolist()}")
            
            # Step 1: Store KV using triton (same as prefill and decode)
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                if debug:
                    print(f"[ATTN_JACOBI] Stored {k.shape[0]} KV entries via triton")
            
            # Step 2: Reshape Q for flash_attn_with_kvcache: (B*L, heads, dim) -> (B, L, heads, dim)
            q_batched = q.view(B, L, self.num_heads, self.head_dim)
            
            # Step 3: Compute attention using flash_attn_with_kvcache
            # cache_seqlens_after_store = cache_seqlens + L (total tokens after storing)
            cache_seqlens_after_store = context.cache_seqlens + L
            
            if debug:
                print(f"[ATTN_JACOBI] cache_seqlens_after_store={cache_seqlens_after_store.tolist()}")
            
            o = flash_attn_with_kvcache(q_batched, k_cache, v_cache,
                                        cache_seqlens=cache_seqlens_after_store, 
                                        block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
            # Output shape: (B, L, num_heads, head_dim) -> flatten to (B*L, num_heads, head_dim)
            o = o.view(B * L, self.num_heads, self.head_dim)
            
            if debug:
                print(f"[ATTN_JACOBI] output.shape={tuple(o.shape)}")
        elif context.is_prefill:
            # Regular prefill (no paged cache) or prefix cache prefill
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
