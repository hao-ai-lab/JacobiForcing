#!/bin/bash

# ===== Config =====
json_files=(
    "/data/numa0/train-tests/data/OpenCodeInstruct_bucketed/bucket_0030_avg390_min387_max392.json"
    "/data/numa0/train-tests/data/OpenCodeInstruct_bucketed/bucket_0031_avg394_min392_max397.json"
    "/data/numa0/train-tests/data/OpenCodeInstruct_bucketed/bucket_0032_avg399_min397_max402.json"
    "/data/numa0/train-tests/data/OpenCodeInstruct_bucketed/bucket_0033_avg404_min402_max407.json"
)

save_path="/data/numa0/train-tests/data/opencodeinstruct_generated_trajectory_blk32"
log_file="/data/numa0/train-tests/data/cllm_logs/generate_trajectory_greedy_k8s_batch_1_data30_to_data33.log"
# ===== Config =====

model_path="/data/numa0/train-tests/models/progressive_noise_cllm2_mask_1m_steps"
n_token_seq_len=32
max_new_seq_len=1024
data_start_id=5001
data_eos_id=10000
batch_size=1

# ===== Launch jobs =====
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/v2/generate_trajectory_kv_greedy_opencodeinstruct.py \
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
