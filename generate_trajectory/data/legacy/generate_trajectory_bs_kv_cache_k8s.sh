#!/bin/bash

# Config
json_files=(
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0212_avg13182_min13041_max13324.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0211_avg12906_min12773_max13041.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0210_avg12639_min12506_max12773.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0209_avg12380_min12254_max12506.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0208_avg12127_min12002_max12254.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0207_avg11883_min11763_max12002.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0206_avg11649_min11538_max11763.json"
    "/checkpoint/lhu/data/local_prompts_bucketed_data_openthought_1m/bucket_0205_avg11426_min11316_max11538.json"
)

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
n_token_seq_len=64
max_new_seq_len=16384
data_start_id=0
batch_size=6
save_path="/checkpoint/lhu/data/CLLM2_data_prep/trajectory_bs_k8s"
log_file="/checkpoint/lhu/data/cllm_logs/generate_trajectory_bs_k8s_kv_cached_batch_3.log"

# Launch jobs
for i in "${!json_files[@]}"; do
    cuda_device=$i
    echo "Device CUDA: ${i}"
    filename="${json_files[$i]}"

    # Each GPU gets one file
    echo "Launching process on CUDA:${cuda_device} for file ${filename}"

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 data/generate_trajectory_bs_kv_cache_k8s.py \
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
