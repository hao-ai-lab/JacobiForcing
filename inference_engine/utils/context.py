from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # For Jacobi forward: cache_seqlens[i] = number of tokens BEFORE this iteration
    cache_seqlens: torch.Tensor | None = None
    # For Jacobi forward: seqlen_q = L (all sequences have same query length)
    seqlen_q: int = 0
    # Flag for CUDA graph-captured Jacobi mode
    is_jacobi_graphed: bool = False

_CONTEXT = Context()

# Cache for Jacobi contexts (Optimization 2)
# Key: (B, L) -> Context with pre-allocated cu_seqlens tensors
_JACOBI_CONTEXT_CACHE: dict[tuple, Context] = {}

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, cache_seqlens=None, seqlen_q=0, is_jacobi_graphed=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, cache_seqlens, seqlen_q, is_jacobi_graphed)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def get_or_create_jacobi_context(B: int, L: int, device: str = 'cuda') -> Context:
    """
    Get or create a cached Context object for Jacobi decoding (eager mode).
    
    Optimization: Reuses pre-allocated cu_seqlens tensors to avoid
    allocation overhead on each Jacobi iteration.
    
    Args:
        B: Batch size
        L: Block length (sequence length per batch item)
        device: Device for tensors
    
    Returns:
        Cached Context object with pre-allocated cu_seqlens tensors
    """
    global _JACOBI_CONTEXT_CACHE
    
    cache_key = (B, L)
    
    if cache_key not in _JACOBI_CONTEXT_CACHE:
        # Create new context with pre-allocated tensors
        ctx = Context(
            is_prefill=True,
            cu_seqlens_q=torch.zeros(B + 1, dtype=torch.int32, device=device),
            cu_seqlens_k=torch.zeros(B + 1, dtype=torch.int32, device=device),
            max_seqlen_q=L,
            max_seqlen_k=0,  # Will be updated per call
            slot_mapping=None,  # Updated per call from jacobi_buffers
            context_lens=None,  # Not used in Jacobi varlen mode
            block_tables=None,  # Updated per call from jacobi_buffers
            cache_seqlens=None,  # Updated per call from jacobi_buffers
            seqlen_q=L,
            is_jacobi_graphed=False,
        )
        _JACOBI_CONTEXT_CACHE[cache_key] = ctx
    
    return _JACOBI_CONTEXT_CACHE[cache_key]


def set_jacobi_context_active(ctx: Context):
    """Set a cached Jacobi context as the active global context."""
    global _CONTEXT
    _CONTEXT = ctx


def clear_jacobi_context_cache():
    """Clear the Jacobi context cache (call during cleanup)."""
    global _JACOBI_CONTEXT_CACHE
    _JACOBI_CONTEXT_CACHE.clear()
