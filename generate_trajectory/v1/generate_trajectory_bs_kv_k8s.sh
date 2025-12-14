#!/bin/bash

# ===== Config =====
json_files=(
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0220_avg15970_min15758_max16187.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0219_avg15545_min15346_max15758.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0218_avg15154_min14970_max15346.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0217_avg14787_min14607_max14970.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0216_avg14432_min14263_max14607.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0215_avg14097_min13934_max14263.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0214_avg13777_min13624_max13934.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0213_avg13474_min13324_max13624.json"
)

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
n_token_seq_len=64
max_new_seq_len=16384
data_start_id=0
batch_size=6
save_path="/checkpoint/lhu/data/CLLM2_data_prep/trajectory_bs_k8s"
log_file="/checkpoint/lhu/data/cllm_logs/generate_trajectory_bs_kv_k8s_batch_5.log"

# ===== Launch jobs =====
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 data/v1/generate_trajectory_bs_kv_k8s.py \
        --filename "${filename}" \
        --model "${model_path}" \
        --n_token_seq_len "${n_token_seq_len}" \
        --max_new_seq_len "${max_new_seq_len}" \
        --data_bos_id ${data_start_id} \
        --batch_size "${batch_size}" \
        --save_path "${save_path}" \
        >"${log_file}" 2>&1 &
done

wait
echo "All trajectory generation processes completed."
