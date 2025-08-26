#!/bin/bash

# ===== Config =====
json_files=(
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0000_avg197_min110_max219.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0001_avg230_min219_max238.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0002_avg245_min238_max250.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0003_avg255_min250_max260.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0004_avg264_min260_max268.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0005_avg272_min268_max275.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0006_avg278_min275_max281.json"
    "/checkpoint/lhu/data/CLLM2_OpenCodeInstruct/1_bucketed/bucket_0007_avg284_min281_max286.json"
)

save_path="/checkpoint/lhu/data/CLLM2_data_prep_opencoderinstruct/trajectory_bs_k8s"
log_file="/checkpoint/lhu/data/cllm_logs/generate_trajectory_bs_k8s_batch_1_0_to_7_all.log"
# ===== Config =====

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
n_token_seq_len=16
max_new_seq_len=2048
data_start_id=0
data_eos_id=25000
batch_size=16

# ===== Launch jobs =====
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/v2/generate_trajectory_bs_kv_k8s_new_opencodeinstruct.py \
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
