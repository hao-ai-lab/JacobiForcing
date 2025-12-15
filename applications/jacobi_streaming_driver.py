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
    stream_per_token: bool = True,  # NEW
):
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    past_key_values = None
    prefill_phase = True
    generated_ids = input_ids.clone()

    total_new_tokens_est = 0
    calls = 0
    generated_text = ""

    jacobi_time = 0.0
    MAX_CALLS = 128

    first_correct_token = None
    prefill_drafted_n_gram = None

    while True:
        generated_part = generated_ids[:, prompt_len:]
        if eos_id is not None and generated_part.numel() > 0:
            if (generated_part == eos_id).any().item():
                break

        if total_new_tokens_est >= max_new_tokens:
            break
        if calls >= MAX_CALLS:
            break

        if prefill_phase:
            seq_len = generated_ids.shape[1]
            idxs = torch.randint(
                low=0,
                high=seq_len,
                size=(n_token_seq_len,),
                device=generated_ids.device,
            )
            prefill_draft_token_ids = generated_ids[0, idxs].unsqueeze(0)
            prefill_input_ids = torch.cat((input_ids, prefill_draft_token_ids), dim=-1)

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

            prefill_phase = False
            generated_ids = input_ids
            calls += 1
            continue

        if calls == 1:
            draft_input_ids = prefill_drafted_n_gram
        else:
            seq_len = generated_ids.shape[1]
            tail_len = max(n_token_seq_len - 1, 1)
            idxs = torch.randint(
                low=0,
                high=seq_len,
                size=(tail_len,),
                device=generated_ids.device,
            )
            tail = generated_ids[0, idxs].unsqueeze(0)
            draft_input_ids = torch.cat(
                (first_correct_token.view(1, -1), tail), dim=-1
            )

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
        jacobi_time += (time.perf_counter() - t0)
        calls += 1

        if accepted_n_gram is None or accepted_n_gram.numel() == 0:
            continue

        generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

        token_ids = accepted_n_gram[0].tolist()

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

            if stream_per_token:
                # STREAM EVERY TOKEN (UI update per token)
                for t in usable_ids:
                    delta = tokenizer.decode(
                        [t],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if not delta:
                        continue
                    generated_text += delta
                    if on_text is not None:
                        on_text(generated_text)
            else:
                # STREAM PER CHUNK (your old behavior)
                chunk_text = tokenizer.decode(
                    usable_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if chunk_text:
                    generated_text += chunk_text
                    if on_text is not None:
                        on_text(generated_text)

        if eos_hit:
            break

    gen_time = jacobi_time
    assistant_text = generated_text.strip()

    final_token_ids = torch.empty((1, 0), dtype=torch.long)
    if assistant_text:
        final_token_ids = tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        new_tokens = int(final_token_ids.shape[1]) - 1  # keep your "-1" behavior
    else:
        new_tokens = 0

    return assistant_text, final_token_ids, total_new_tokens_est, gen_time
