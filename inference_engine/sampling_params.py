from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Decode strategy: "autoregressive" [supported], "jacobi" [supported], "jacobi_multiblock_rejection_recycling" [not supported]
    decode_strategy: str = "autoregressive"

    jacobi_block_len: int = 64          # n_token_seq_len in your code
    jacobi_max_iterations: int = 128
    
    # Jacobi multi-block params
    jacobi_max_blocks: int = 2           # K
    jacobi_spawn_ratio: float = 0.85     # r
    jacobi_lookahead_start_ratio: float = 0.0
    jacobi_n_gram_pool_size: int = 4
    
    # On-policy learning: if True, returns rollout records instead of tokens
    # Requires non-greedy decoding (temperature > 0) and decode_strategy == "jacobi"
    jacobi_on_policy: bool = False

    def __post_init__(self):
        # Allow temperature=0 for greedy decoding
        assert self.temperature >= 0.0, "temperature must be non-negative"
        # On-policy learning requires non-greedy decoding
        if self.jacobi_on_policy and self.temperature == 0.0:
            raise ValueError(
                "jacobi_on_policy=True requires temperature > 0 (non-greedy decoding). "
                "On-policy learning is only supported with non-greedy Jacobi decoding."
            )

    @property
    def use_jacobi(self) -> bool:
        return self.jacobi_block_len is not None
