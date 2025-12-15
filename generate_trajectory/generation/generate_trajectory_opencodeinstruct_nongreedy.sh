#!/bin/bash

# ===== Config =====
json_files=(
    "/data/nfs01/lanxiang/data/OpenCodeInstruct_bucketed/bucket_0016_avg328_min325_max330.json"
    "/data/nfs01/lanxiang/data/OpenCodeInstruct_bucketed/bucket_0017_avg332_min330_max334.json"
    "/data/nfs01/lanxiang/data/OpenCodeInstruct_bucketed/bucket_0018_avg336_min334_max338.json"
    "/data/nfs01/lanxiang/data/OpenCodeInstruct_bucketed/bucket_0019_avg341_min338_max343.json"
)

save_path="/data/nfs01/lanxiang/data/CLLM2_data_prep/trajectory_bs_k8s_opencoderinstruct"
log_file="/data/nfs01/lanxiang/data/cllm_logs/generate_trajectory_bs_k8s_batch_1_data16_to_data19.log"
# ===== Config =====

model_path="/home/ubuntu/Qwen2.5-Coder-7B-Instruct"
n_token_seq_len=32
max_new_seq_len=2048
data_start_id=0
data_eos_id=25000
batch_size=128

# ===== Launch jobs =====
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/v2/generate_trajectory_opencodeinstruct_nongreedy.py \
        --filename "${filename}" \
        --model "${model_path}" \
        --n_token_seq_len "${n_token_seq_len}" \
        --max_new_seq_len "${max_new_seq_len}" \
        --data_bos_id ${data_start_id} \
        --data_eos_id ${data_eos_id} \
        --batch_size "${batch_size}" \
        --save_path "${save_path}" \
        >"${log_file}" 2>&1 &
done

wait
echo "All trajectory generation processes completed."
