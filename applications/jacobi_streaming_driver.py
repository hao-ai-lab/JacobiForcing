import time
from typing import Callable, List, Dict, Optional

import torch


@torch.inference_mode()
def jacobi_stream_chat(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    n_token_seq_len: int = 64,
    max_new_tokens: int = 512,
    K: int = 2,
    r: float = 0.8,
    n_gram_pool_size: int = 4,
    on_text: Optional[Callable[[str], None]] = None,
):
    """
    Streaming driver for `jacobi_forward_greedy_multiblock`.

    IMPORTANT: gen_time does NOT include tokenization,
    random sampling, or UI/streaming overhead.

    Args:
        model: Qwen2ForCausalLM with `jacobi_forward_greedy_multiblock` patched
        tokenizer: corresponding tokenizer
        messages: HF chat-style messages
        n_token_seq_len: block length for Jacobi
        max_new_tokens: cap on decoded tokens (approximate, based on decoded text)
        K: max number of concurrent blocks (1 real-active + K-1 pseudo blocks)
        r: spawn threshold as a fraction of n_token_seq_len
        n_gram_pool_size: max size of the n-gram pool used by lookahead logic
        on_text: callback that receives the full text-so-far every time it updates

    Returns:
        assistant_text: final decoded text
        new_tokens: # of tokens in assistant_text (tokenizer-based, minus 1 if EOS)
        gen_time: total time (seconds) spent inside `jacobi_forward_greedy_multiblock`
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Build chat prompt using HF chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]  # [1, L_prompt]
    prompt_len = input_ids.shape[1]

    # Basic attention mask for prefill
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    past_key_values = None
    prefill_phase = True
    generated_ids = input_ids.clone()  # token history for EOS checking

    total_new_tokens_est = 0  # estimate from chunks we actually decode
    calls = 0

    generated_text = ""

    # ---- profiling: only time spent inside jacobi_forward_greedy_multiblock ----
    jacobi_time = 0.0
    model_device = getattr(model, "device", None)
    use_cuda = isinstance(model_device, torch.device) and (model_device.type == "cuda")

    # internal safety cap on number of jacobi calls
    MAX_CALLS = 128

    first_correct_token = None
    prefill_drafted_n_gram = None

    while True:
        # Check EOS in already generated region
        generated_part = generated_ids[:, prompt_len:]
        if eos_id is not None and generated_part.numel() > 0:
            if (generated_part == eos_id).any().item():
                break

        if total_new_tokens_est >= max_new_tokens:
            break
        if calls >= MAX_CALLS:
            break

        if prefill_phase:
            # --------------------------------------------------
            # PREFILL PHASE: random draft tokens from history
            # --------------------------------------------------
            seq_len = generated_ids.shape[1]
            idxs = torch.randint(
                low=0,
                high=seq_len,
                size=(n_token_seq_len,),
                device=generated_ids.device,
            )
            prefill_draft_token_ids = generated_ids[0, idxs].unsqueeze(0)  # [1, n_token_seq_len]
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

            # timed prefill call
            #if use_cuda:
            #    torch.cuda.synchronize(model_device)
            #t0 = time.time()

            past_key_values, first_correct_token, prefill_drafted_n_gram, _ = (
                model.jacobi_forward_greedy_multiblock(
                    input_ids=prefill_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    prefill_phase=True,
                    n_token_seq_len=n_token_seq_len,
                    K=K,
                    r=r,
                    n_gram_pool_size=n_gram_pool_size,
                    tokenizer=tokenizer,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            )

            #if use_cuda:
            #    torch.cuda.synchronize(model_device)
            #t1 = time.time()
            #jacobi_time += (t1 - t0)

            prefill_phase = False
            generated_ids = input_ids  # reset to just the prompt
            calls += 1
            continue

        # --------------------------------------------------
        # GENERATION PHASE
        # --------------------------------------------------
        if calls == 1:
            # First generation call: reuse the full drafted n-gram from prefill
            draft_input_ids = prefill_drafted_n_gram
        else:
            # Subsequent calls: first_correct_token + random tail from history
            seq_len = generated_ids.shape[1]
            tail_len = max(n_token_seq_len - 1, 1)
            idxs = torch.randint(
                low=0,
                high=seq_len,
                size=(tail_len,),
                device=generated_ids.device,
            )
            tail = generated_ids[0, idxs].unsqueeze(0)  # [1, tail_len]
            draft_input_ids = torch.cat(
                (first_correct_token.view(1, -1), tail), dim=-1
            )  # [1, n_token_seq_len]

        # timed generation call
        #if use_cuda:
        #    torch.cuda.synchronize(model_device)
        t0 = time.perf_counter()

        past_key_values, first_correct_token, accepted_n_gram, _ = (
            model.jacobi_forward_greedy_multiblock(
                input_ids=draft_input_ids,
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                prefill_phase=False,
                n_token_seq_len=n_token_seq_len,
                K=K,
                r=r,
                n_gram_pool_size=n_gram_pool_size,
                tokenizer=tokenizer,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        )

        #if use_cuda:
        #    torch.cuda.synchronize(model_device)
        
        t1 = time.perf_counter()
        jacobi_time += (t1 - t0)

        calls += 1

        if accepted_n_gram is None or accepted_n_gram.numel() == 0:
            continue

        # Append to token history for EOS detection
        generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

        token_ids = accepted_n_gram[0].tolist()

        # Filter PADs, cut at EOS
        if pad_id is not None:
            token_ids = [t for t in token_ids if t != pad_id]

        eos_hit = False
        usable_ids = []
        for t in token_ids:
            if eos_id is not None and t == eos_id:
                eos_hit = True
                break
            usable_ids.append(t)

        if usable_ids:
            total_new_tokens_est += len(usable_ids)
            chunk_text = tokenizer.decode(
                usable_ids,
                skip_special_tokens=True,
            )
            if chunk_text:
                generated_text += chunk_text
                if on_text is not None:
                    # Pass the full text so far to the UI
                    on_text(generated_text)

        if eos_hit:
            break

    # gen_time is ONLY Jacobi kernel time now
    gen_time = jacobi_time

    assistant_text = generated_text.strip()
    if assistant_text:
        token_ids = tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        new_tokens = int(token_ids.shape[1])
        # keep your original "-1" behavior
        new_tokens -= 1
    else:
        new_tokens = 0

    return assistant_text, token_ids, total_new_tokens_est, gen_time
