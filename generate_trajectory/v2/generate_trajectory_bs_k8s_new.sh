#!/bin/bash

# ===== Config =====
json_files=(
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0045_avg1810_min1800_max1820.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0055_avg2018_min2008_max2030.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0065_avg2243_min2232_max2255.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0075_avg2489_min2476_max2501.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0085_avg2754_min2740_max2768.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0095_avg3047_min3031_max3063.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0105_avg3369_min3352_max3387.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0115_avg3721_min3702_max3739.json"
)

save_path="/checkpoint/lhu/data/CLLM2_data_prep/trajectory_bs_k8s"
log_file="/checkpoint/lhu/data/cllm_logs/generate_trajectory_bs_k8s_batch_8_45_to_115_step_10.log"
# ===== Config =====

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
n_token_seq_len=64
max_new_seq_len=16384
data_start_id=0
data_eos_id=100
batch_size=4

# ===== Launch jobs =====
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/v2/generate_trajectory_bs_kv_k8s_new.py \
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
