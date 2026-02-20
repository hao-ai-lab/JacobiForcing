from copy import copy
from enum import Enum, auto
from itertools import count

from inference_engine.sampling_params import SamplingParams

from dataclasses import dataclass, field

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        # flags for decode path
        self.decode_strategy = sampling_params.decode_strategy
        self.jacobi_block_len = sampling_params.jacobi_block_len
        self.jacobi_max_blocks = sampling_params.jacobi_max_blocks
        self.jacobi_spawn_ratio = sampling_params.jacobi_spawn_ratio
        self.jacobi_lookahead_start_ratio = sampling_params.jacobi_lookahead_start_ratio
        self.jacobi_n_gram_pool_size = sampling_params.jacobi_n_gram_pool_size
        self.jacobi_max_iterations = sampling_params.jacobi_max_iterations
        self.jacobi_on_policy = sampling_params.jacobi_on_policy
        
        # Store sampling_params reference for accessing full params when needed
        self.sampling_params = sampling_params
        
        # Draft buffer for Jacobi decoding (speculative tokens, not yet verified)
        self.draft_tokens = None  # list[int] or None (CPU, for compatibility)
        self.draft_tokens_gpu = None  # torch.Tensor or None (GPU, optimization)
        
        # Prefill draft for Jacobi: greedy predictions from prefill to bootstrap first iteration
        self._prefill_draft = None  # list[int] or None
        
        # Pre-allocated speculative blocks for Jacobi (optimization to avoid repeated alloc/dealloc)
        self.num_permanent_spec_blocks = 0  # Number of blocks pre-allocated for speculation
        
        # GPU-cached block table (optimization to avoid CPU→GPU transfer every iteration)
        self.block_table_gpu = None  # torch.Tensor or None
        self.block_table_version = 0  # Increment when block_table changes

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # Validate that token_id is a single integer, not a list/tensor
        if isinstance(token_id, (list, tuple)):
            import sys
            print(f"ERROR: append_token called with list/tuple of length {len(token_id)}!", 
                  file=sys.stderr, flush=True)
            print(f"  Caller should append tokens individually, not as a list", file=sys.stderr, flush=True)
            raise ValueError(f"append_token expects a single integer, got {type(token_id)} with {len(token_id)} elements")
        
        # Convert to Python int if it's a numpy/torch scalar
        if hasattr(token_id, 'item'):
            token_id = int(token_id.item())
        else:
            token_id = int(token_id)
            
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def extend_tokens(self, tokens: list):
        """
        Batch append multiple tokens at once.
        More efficient than calling append_token in a loop.
        
        Args:
            tokens: List of token IDs to append
        """
        if not tokens:
            return
        
        # Convert tokens to Python ints if needed
        int_tokens = []
        for tok in tokens:
            if hasattr(tok, 'item'):
                int_tokens.append(int(tok.item()))
            else:
                int_tokens.append(int(tok))
        
        self.token_ids.extend(int_tokens)
        self.last_token = int_tokens[-1]
        self.num_tokens += len(int_tokens)
    
    def has_draft(self) -> bool:
        """Check if sequence has a draft buffer."""
        return self.draft_tokens is not None and len(self.draft_tokens) > 0
    
    def clear_draft(self):
        """Clear the draft buffer."""
        self.draft_tokens = None

    def __getstate__(self):
        # Serialize all attributes except non-picklable GPU tensors
        state = self.__dict__.copy()
        # Remove GPU tensors that can't be pickled across processes
        state['draft_tokens_gpu'] = None
        state['block_table_gpu'] = None
        return state

    def __setstate__(self, state):
        # Restore all attributes from serialized state
        self.__dict__.update(state)
