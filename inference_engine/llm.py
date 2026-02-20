# inference_engine/llm.py

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from inference_engine.engine.llm_engine import LLMEngine
from inference_engine.sampling_params import SamplingParams


class LLM(LLMEngine):
    """
    User-facing API for nano-vllm.

    Extends LLMEngine with:
      - `greedy` flag on generate()
      - Jacobi speculative decoding flags, controllable via SamplingParams
        or via generate() keyword arguments.
    """

    def generate(
        self,
        prompts,
        sampling_params: Optional[SamplingParams] = None,
        *,
        # convenience flag for pure greedy / argmax
        greedy: Optional[bool] = None,
        # Jacobi-related knobs (all optional)
        jacobi_enabled: Optional[bool] = None,
        jacobi_block_len: Optional[int] = None,
        jacobi_num_blocks: Optional[int] = None,
        jacobi_spawn_ratio: Optional[float] = None,
        jacobi_lookahead_start_ratio: Optional[float] = None,
        jacobi_ngram_pool_size: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Extra flags:

        - greedy:
            * None  -> keep whatever is in `sampling_params`.
            * True  -> force pure greedy (temperature=0, disable top_k/top_p).
            * False -> ensure non-greedy (temperature>0 if it was 0).

        - jacobi_enabled:
            * None  -> respect `sampling_params.jacobi_enabled`.
            * True  -> enable Jacobi speculative decoding.
            * False -> disable Jacobi, regardless of SamplingParams.

        Jacobi hyper-parameters can be provided either:

          1) In SamplingParams:

                SamplingParams(
                    ...,
                    jacobi_enabled=True,
                    jacobi_block_len=64,
                    jacobi_num_blocks=2,
                    jacobi_spawn_ratio=0.8,
                    jacobi_lookahead_start_ratio=0.0,
                    jacobi_ngram_pool_size=4,
                )

          2) As keyword-arguments to `generate(...)`:

                llm.generate(
                    prompts,
                    sampling_params,
                    jacobi_enabled=True,
                    jacobi_block_len=64,
                    jacobi_num_blocks=2,
                    jacobi_spawn_ratio=0.8,
                    jacobi_lookahead_start_ratio=0.0,
                    jacobi_ngram_pool_size=4,
                )
        """

        if sampling_params is None:
            sampling_params = SamplingParams()

        sp = sampling_params

        # ------------------------------------------------------------
        # 1) Greedy handling
        # ------------------------------------------------------------
        if greedy is not None:
            if greedy:
                # Check for conflict with on-policy learning
                if getattr(sp, "jacobi_on_policy", False):
                    raise ValueError(
                        "Cannot use greedy=True with jacobi_on_policy=True. "
                        "On-policy learning requires non-greedy decoding (temperature > 0)."
                    )
                # Pure argmax decoding: temperature=0, full-vocab.
                sp = replace(sp, temperature=0.0)

                if hasattr(sp, "top_k"):
                    sp = replace(sp, top_k=-1)
                if hasattr(sp, "top_p"):
                    sp = replace(sp, top_p=1.0)
            else:
                # Explicitly non-greedy: if temperature was 0, bump it.
                if getattr(sp, "temperature", 1.0) == 0.0:
                    sp = replace(sp, temperature=1.0)

        # ------------------------------------------------------------
        # 2) Jacobi handling
        # ------------------------------------------------------------
        jacobi_args_provided = any(
            v is not None
            for v in (
                jacobi_block_len,
                jacobi_num_blocks,
                jacobi_spawn_ratio,
                jacobi_lookahead_start_ratio,
                jacobi_ngram_pool_size,
            )
        )

        if jacobi_enabled is not None or jacobi_args_provided:
            update_kwargs: dict[str, Any] = {}

            # Toggle Jacobi explicitly if provided.
            if jacobi_enabled is not None:
                update_kwargs["jacobi_enabled"] = jacobi_enabled
            elif jacobi_args_provided:
                # Supplying Jacobi knobs implies enabling it.
                update_kwargs["jacobi_enabled"] = True

            if jacobi_block_len is not None:
                update_kwargs["jacobi_block_len"] = jacobi_block_len
            if jacobi_num_blocks is not None:
                update_kwargs["jacobi_num_blocks"] = jacobi_num_blocks
            if jacobi_spawn_ratio is not None:
                update_kwargs["jacobi_spawn_ratio"] = jacobi_spawn_ratio
            if jacobi_lookahead_start_ratio is not None:
                update_kwargs["jacobi_lookahead_start_ratio"] = (
                    jacobi_lookahead_start_ratio
                )
            if jacobi_ngram_pool_size is not None:
                update_kwargs["jacobi_ngram_pool_size"] = jacobi_ngram_pool_size

            sp = replace(sp, **update_kwargs)

        # ------------------------------------------------------------
        # 3) Delegate to engine
        # ------------------------------------------------------------
        return super().generate(prompts, sp, **kwargs)
