from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---- helpers ----
def find_first_true_index(bool_tensor, dim=-1):
    # number of leading Falses (index of first True; if none -> length)
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)


# diffusion decoding with 32-token draft
@torch.inference_mode()
def diffusion_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    n_token_seq_len,
    temperature=0.9,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.05,
    lenience=1.0,
    accept_threshold=0.8,  # REAL path acceptance-threshold
    pseudo_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),  # thresholds for PSEUDO scanning
    profile_block_multiple=2,  # 2 * 16 = 32 monitored positions (also used as draft length)
    return_profile=False,
):
    """
    Returns:
      - out: tensor with appended tokens after this block (ONLY the converged-16 part; no extra next-token)
      - itr: number of iterations taken for this 16-token block to converge
      - bundle (if return_profile=True):
          {
            "accepted_rows": [np.float32(32)] * I   # REAL acceptance, cumulative across ALL 32 cols
            "valid_rows":     [np.float32(32)] * I  # 1=valid position this iter (for REAL), 0=padding
            "pseudo_accept_rows": {thr: [np.float32(32)] * I}  # per-position, non-accumulative, 1=accept,0=reject
          }

    Notes:
      * 32-token (2×n) Jacobi draft each iteration.
      * Convergence/termination depend only on the FIRST 16 positions.
    """

    batch = 1
    prompt_len = int(torch.sum(attention_mask[0]))
    out = input_ids.clone()
    device = out.device

    # ALWAYS take vocab size from the model head
    V = model.get_output_embeddings().weight.size(0)

    # set draft length
    DRAFT_LEN = n_token_seq_len * profile_block_multiple

    # init 32-token draft from prompt tokens
    q_sampled_list, q_logits_list = [], []
    for _ in range(DRAFT_LEN):
        q_sample = (
            torch.tensor([random.choice(input_ids[0].tolist())], dtype=torch.long, device=device)
            .unsqueeze(0)
        )
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((batch, V), float('-inf'), device=device)  # V from model head
        q_logits.scatter_(1, q_sample, 0.0)
        q_sampled_list.append(q_sample)
        q_logits_list.append(q_logits)

    q_sampled = torch.cat(q_sampled_list, dim=1)         # [1, 2 x n-token-seq-len]
    q_logits = torch.stack(q_logits_list, dim=-2)        # [1, 2 x n-token-seq-len, V]

    # logits processors
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    logits_processors = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature is not None and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p is not None and top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    # profiling containers
    bundle = None
    if return_profile:
        bundle = {
            "accepted_rows": [],
            "valid_rows": [],
            "pseudo_accept_rows": {float(th): [] for th in pseudo_thresholds},
        }
        # deterministic RNGs
        seed_base = int((int(out.sum().item()) ^ (prompt_len << 16) ^ 0x9E3779B9) & 0xFFFFFFFF)
        rng_py = random.Random(seed_base)
        gen_prof = torch.Generator(device=device)
        gen_prof.manual_seed((seed_base ^ 0xA5A5A5A5) & 0xFFFFFFFF)

    # ---- iterate until the FIRST 16 of the block converge ----
    total_accepted = 0
    itr = 0  # iteration index starts at 0

    # cumulative accepted length across all tokens
    vis_cum_accepted = 0

    while total_accepted < n_token_seq_len:
        # forward
        out_attention_mask = torch.ones_like(out, device=device)
        logits = model(out, out_attention_mask).logits  # [1, L, V_model]
        Vm = logits.size(-1)
        V_use = min(V, Vm)

        p_logits = logits[:, prompt_len + total_accepted - 1 :, :V_use]  # [1, cur+1, V_use]

        # processed scores
        p_scores = logits_processors(out, p_logits.squeeze(0)).unsqueeze(0)           # [1, cur+1, V_use]
        q_scores = logits_processors(out, q_logits[..., :V_use].squeeze(0)).unsqueeze(0)  # [1, cur, V_use]

        # probabilities
        p_prob = nn.functional.softmax(p_scores, dim=-1)  # [1, cur+1, V_use]
        q_prob = nn.functional.softmax(q_scores, dim=-1)  # [1, cur,   V_use]

        # gather real-path and jacobi-path probabilities across current draft
        p_real, prob_next = p_prob[:, :-1], p_prob[:, -1]  # [1,cur,V_use], [1,V_use]
        p_g_all = p_real.gather(-1, q_sampled.unsqueeze(-1)).squeeze(-1)                     # [1,cur]
        q_g_all = (q_prob.gather(-1, q_sampled.unsqueeze(-1)).squeeze(-1)) * lenience        # [1,cur]

        cur_len = p_g_all.shape[1]

        # ---- convergence check on FIRST n-token-seq-len only
        p_g = p_g_all[:, :n_token_seq_len]  # [1,<=16]
        q_g = q_g_all[:, :n_token_seq_len]  # [1,<=16]
        if p_g.shape[1] < n_token_seq_len:
            pad = n_token_seq_len - p_g.shape[1]
            p_g = torch.cat([p_g, torch.ones((1, pad), device=device, dtype=p_g.dtype)], dim=1)
            q_g = torch.cat([q_g, torch.ones((1, pad), device=device, dtype=q_g.dtype)], dim=1)

        r = torch.rand_like(p_g)  # [1,16]
        threshold_real = torch.ones_like(p_g) * accept_threshold  # [1,16]
        cond16 = (r > (p_g / q_g)) | (p_g < threshold_real)  # True == reject
        num_accepted = int(find_first_true_index(cond16)[0])  # 0..16

        total_accepted += num_accepted
        out = out[:, : prompt_len + total_accepted]
        has_rejected = num_accepted < n_token_seq_len

        # ---- PSEUDO ACCEPT: independently scan over ALL 32 positions (non-accumulative; 1=accept,0=reject) ----
        if return_profile:
            # Build a full-length shadow draft of 32 tokens (independent of current q_sampled length)
            q_shadow_tokens, q_shadow_logits = [], []
            out_shadow = out.clone()
            for _ in range(DRAFT_LEN):
                tkn = (
                    torch.tensor([rng_py.choice(input_ids[0].tolist())], device=device, dtype=torch.long)
                    .unsqueeze(0)
                )  # [1,1]
                out_shadow = torch.cat((out_shadow, tkn), dim=1)
                ql = torch.full((batch, V), float('-inf'), device=device)
                ql.scatter_(1, tkn, 0.0)
                q_shadow_tokens.append(tkn)
                q_shadow_logits.append(ql)

            q_sampled_shadow = torch.cat(q_shadow_tokens, dim=1)   # [1,32]
            q_logits_shadow = torch.stack(q_shadow_logits, dim=-2) # [1,32,V]

            attn_shadow = torch.ones_like(out_shadow, device=device)
            logits_shadow = model(out_shadow, attn_shadow).logits   # [1, L', V_model]
            Vm_sh = logits_shadow.size(-1)
            V_use_sh = min(V, Vm_sh)

            p_logits_shadow = logits_shadow[:, prompt_len + total_accepted - 1 :, :V_use_sh]  # [1,33,V]
            p_scores_shadow = logits_processors(out_shadow, p_logits_shadow.squeeze(0)).unsqueeze(0)
            q_scores_shadow = logits_processors(out_shadow, q_logits_shadow[..., :V_use_sh].squeeze(0)).unsqueeze(0)

            p_prob_sh = nn.functional.softmax(p_scores_shadow, dim=-1)  # [1,33,V]
            q_prob_sh = nn.functional.softmax(q_scores_shadow, dim=-1)  # [1,32,V]

            p_pos_sh = p_prob_sh[:, :-1, :]  # [1,32,V]
            q_pos_sh = q_prob_sh             # [1,32,V]

            p_sh = p_pos_sh.gather(-1, q_sampled_shadow.unsqueeze(-1)).squeeze(-1)        # [1,32]
            q_sh = (q_pos_sh.gather(-1, q_sampled_shadow.unsqueeze(-1)).squeeze(-1)) * lenience  # [1,32]
            r_sh = torch.rand(p_sh.shape, dtype=p_sh.dtype, device=device, generator=gen_prof)    # [1,32]

            for th in pseudo_thresholds:
                th_t = torch.ones_like(p_sh) * float(th)
                cond_mask_sh = (r_sh > (p_sh / q_sh)) | (p_sh < th_t)  # True==reject
                acc_mask_sh = (~cond_mask_sh).float()                  # 1=accept, 0=reject
                pseudo_row = acc_mask_sh.squeeze(0).cpu().numpy().astype(np.float32)  # [32]
                bundle["pseudo_accept_rows"][float(th)].append(pseudo_row)

        # decides next-token when needed
        sample_additional_token = False
        if num_accepted == 0:
            next_token = torch.multinomial(p_prob[:, num_accepted, :], num_samples=1)
            out = torch.cat((out, next_token), dim=-1)
            total_accepted += 1
            sample_additional_token = True
        elif has_rejected:
            adjusted = F.relu(p_prob[:, num_accepted, :] - q_prob[:, num_accepted, :])
            adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True)
            if not torch.isnan(adjusted).any():
                next_token = torch.multinomial(adjusted, num_samples=1)
                out = torch.cat((out, next_token), dim=-1)
                total_accepted += 1
                sample_additional_token = True

        if return_profile:
            # cumulative prefix ACROSS ALL tokens (REAL acceptance only)
            accepted_increment = num_accepted + (1 if sample_additional_token else 0)
            vis_cum_accepted = min(DRAFT_LEN, vis_cum_accepted + accepted_increment)

            accepted_row = np.zeros(DRAFT_LEN, dtype=np.float32)
            if vis_cum_accepted > 0:
                accepted_row[:vis_cum_accepted] = 1.0

            valid_row = np.zeros(DRAFT_LEN, dtype=np.float32)
            valid_row[:min(cur_len, DRAFT_LEN)] = 1.0

            bundle["accepted_rows"].append(accepted_row)
            bundle["valid_rows"].append(valid_row)

        if not has_rejected:
            if return_profile:
                return out, itr + 1, bundle
            else:
                return out, itr + 1

        # Jacobi update of draft for remaining positions
        if sample_additional_token:
            new_q_logits = p_logits[:, num_accepted + 1 : -1, :]
            new_q_probs  = p_prob[:,   num_accepted + 1 : -1, :]
        else:
            new_q_logits = p_logits[:, num_accepted : -1, :]
            new_q_probs  = p_prob[:,   num_accepted : -1, :]

        if new_q_probs.shape[1] <= 0:
            if return_profile:
                return out, itr + 1, bundle
            else:
                return out, itr + 1

        q_sampled = torch.multinomial(new_q_probs.squeeze(0), num_samples=1).reshape(1, -1)  # [1,new_len]
        out = torch.cat((out, q_sampled), dim=-1)
        q_logits = new_q_logits

        itr += 1

    if return_profile:
        return out, itr, bundle
    else:
        return out, itr


def main():
    df = pd.read_parquet("/checkpoint/lhu/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    records = df.head(100).to_dict(orient="records")

    model_name = "/checkpoint/lhu/train_ckpts/cllm/shiftedattn-8-31-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_cyclic_progressive_noise_all_lr5e-6/hf_ckpts/hf_merged_step_44500"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645

    # ---------------------------
    # Config
    # ---------------------------
    n_token_seq_len = 16
    temperature = 0.9
    top_p = 0.9
    top_k = 20
    repetition_penalty = 1.2
    lenience = 1.0
    accept_threshold = 0.1  # REAL threshold
    pseudo_thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    BLOCK_LEN = n_token_seq_len * 2  # 32

    # dataset
    all_rows = []
    t0_overall = time.perf_counter()

    # heatmap accumulators (per-iteration rows across all prompts/blocks)
    all_iter_accept_rows = []           # list of [I, 32]
    all_iter_valid_rows = []            # list of [I, 32] (REAL position-valid masks)
    all_iter_pseudo_accept_rows_by_th = {float(th): [] for th in pseudo_thresholds}  # each: list of [I,32]
    all_iter_valid_rows_pseudo = []     # list of [I, 32] (ALL ONES for pseudo)

    for idx, row in enumerate(records):
        task_id = row.get("task_id", f"idx_{idx}")
        prompt = "Respond only in code.\n" + row["prompt"]

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids, device=model.device)

        iters, total_new_tokens, calls = [], 0, 0
        prev_len = input_ids.shape[1]
        prompt_len = prev_len
        stop_reason = None
        t_start = time.perf_counter()

        while True:
            gen_part = input_ids[0, prompt_len:]
            hit_eos = False
            if eos_id is not None:
                hit_eos = (gen_part == eos_id).any().item()
            if not hit_eos:
                hit_eos = (gen_part == alt_eos_id).any().item()
            if hit_eos:
                stop_reason = "eos"
                break

            out_ids, itr_count, bundle = diffusion_decoding(
                model,
                tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_token_seq_len=n_token_seq_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                lenience=lenience,
                accept_threshold=accept_threshold,
                pseudo_thresholds=pseudo_thresholds,
                profile_block_multiple=2,
                return_profile=True,
            )

            if bundle is not None and len(bundle["accepted_rows"]) > 0:
                iter_acc = np.stack(bundle["accepted_rows"], axis=0)  # [I, 32]
                iter_v   = np.stack(bundle["valid_rows"], axis=0)     # [I, 32]
                all_iter_accept_rows.append(iter_acc)
                all_iter_valid_rows.append(iter_v)

                iter_ones = np.ones_like(iter_acc, dtype=np.float32)  # [I,32]
                all_iter_valid_rows_pseudo.append(iter_ones)
                for th in pseudo_thresholds:
                    iter_ps = np.stack(bundle["pseudo_accept_rows"][float(th)], axis=0)  # [I, 32] (1=accept)
                    all_iter_pseudo_accept_rows_by_th[float(th)].append(iter_ps)

            calls += 1
            iters.append(itr_count)
            added = out_ids.shape[1] - prev_len
            if added > 0:
                total_new_tokens += added
                prev_len = out_ids.shape[1]
            input_ids = out_ids
            attention_mask = torch.ones_like(input_ids, device=model.device)

            if total_new_tokens >= 1024:
                stop_reason = "max_new_tokens"
                break
            if calls >= 10240:
                stop_reason = "max_calls"
                break

        dt = time.perf_counter() - t_start
        avg_iter_per_block = (float(np.mean(iters)) if len(iters) > 0 else float("nan"))
        all_rows.append(
            {
                "index": idx,
                "task_id": task_id,
                "prompt_tokens": prompt_len,
                "new_tokens": total_new_tokens,
                "calls": calls,
                "avg_iter_per_block": avg_iter_per_block,
                "time_sec": dt,
                "stop_reason": stop_reason,
            }
        )
        print(
            f"====[{idx+1}/{len(records)}] task_id={task_id} new_toks={total_new_tokens} "
            f"calls={calls} avg_iter/block={avg_iter_per_block:.2f} reason={stop_reason}===="
        )

    t_overall = time.perf_counter() - t0_overall
    df_profile = pd.DataFrame(all_rows)
    df_profile.to_csv("diffusion_profile_humaneval_blocks.csv", index=False)

    print("=== Diffusion Decoding Profiling (blocks) ===")
    print(f"Samples: {len(records)} Total wall time: {t_overall:.2f}s")
    if len(df_profile):
        print(df_profile[["calls", "avg_iter_per_block", "time_sec", "stop_reason"]].to_string(index=False))

    # Build iteration × position heatmaps
    def _build_iter_pos_rate(mats, valid_mats, block_len):
        if len(mats) == 0:
            return np.zeros((1, block_len), dtype=np.float32)
        max_I = max(m.shape[0] for m in mats)
        num = np.zeros((max_I, block_len), dtype=np.float64)
        den = np.zeros((max_I, block_len), dtype=np.float64)
        for M, V in zip(mats, valid_mats):
            I = M.shape[0]
            num[:I, :] += M
            den[:I, :] += V
        rate = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
        return rate.astype(np.float32)

    BLOCK_LEN = n_token_seq_len * 2  # 32

    # REAL acceptance heatmap
    accept_iter_pos = _build_iter_pos_rate(all_iter_accept_rows, all_iter_valid_rows, BLOCK_LEN)
    np.savetxt("accept_heatmap_iter_pos.csv", accept_iter_pos, delimiter=",")
    plt.figure()
    plt.imshow(accept_iter_pos, aspect='auto', origin='upper')
    plt.colorbar()
    plt.xlabel("position in 32-token draft (first 16 determine convergence)")
    plt.ylabel("iteration idx (0 at top)")
    plt.title("Accepted (REAL) — cumulative across all 32")
    plt.tight_layout()
    plt.savefig("accept_heatmap_iter_pos.png", dpi=150)
    plt.close()

    # Pseudo ACCEPT heatmaps (1=accept, 0=reject), independent over all 32
    for th, mats in all_iter_pseudo_accept_rows_by_th.items():
        pseudo_accept_iter_pos = _build_iter_pos_rate(mats, all_iter_valid_rows_pseudo, BLOCK_LEN)
        np.savetxt(f"pseudo_accept_heatmap_iter_pos_thr_{th:.1f}.csv", pseudo_accept_iter_pos, delimiter=",")
        plt.figure()
        plt.imshow(pseudo_accept_iter_pos, aspect='auto', origin='upper')
        plt.colorbar()
        plt.xlabel("position in 32-token draft")
        plt.ylabel("iteration idx (0 at top)")
        plt.title(f"Pseudo-accept — 1=accept, 0=reject (thr={th:.1f})")
        plt.tight_layout()
        plt.savefig(f"pseudo_accept_heatmap_iter_pos_thr_{th:.1f}.png", dpi=150)
        plt.close()

    print("Saved plots:")
    print(" accept_heatmap_iter_pos.png")
    for th in pseudo_thresholds:
        print(f" pseudo_accept_heatmap_iter_pos_thr_{th:.1f}.png")


if __name__ == "__main__":
    main()
