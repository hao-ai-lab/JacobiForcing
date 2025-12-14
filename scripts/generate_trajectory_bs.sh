#!/bin/bash

# Set base variables
filename="/data/phd/kousiqi/kousiqi/CLLM2/data/raw_data/openthoughts2_1m.json"
model_path="/data/phd/kousiqi/kousiqi/ckpts/OpenThinker2-7B"
n_token_seq_len=64
max_new_seq_len=16384
data_start_id=13000
batch_size=4

# Loop over chunks of 100, assign to different GPU
for i in {0..7}; do
  data_bos_id=$((data_start_id + i * 100))
  data_eos_id=$((data_bos_id + 100))
  cuda_device=$i

  echo "Launching process on CUDA:${cuda_device} for lines ${data_bos_id} to ${data_eos_id}"

  CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/generate_trajectory_bs.py \
    --filename "${filename}" \
    --model "${model_path}" \
    --n_token_seq_len "${n_token_seq_len}" \
    --max_new_seq_len "${max_new_seq_len}" \
    --data_bos_id "${data_bos_id}" \
    --data_eos_id "${data_eos_id}" \
    --data_start_id "${data_start_id}" \
    --batch_size "${batch_size}"&

done

# Wait for all background jobs to finish

echo "All trajectory generation processes completed."


# data_bos_id=$((data_start_id + 0 * 100))
# data_eos_id=$((data_bos_id + 100))
# cuda_device=1

# echo "Launching process on CUDA:${cuda_device} for lines ${data_bos_id} to ${data_eos_id}"

# CUDA_VISIBLE_DEVICES=${cuda_device} python3 generate_trajectory/generate_trajectory_bs.py \
#     --filename "${filename}" \
#     --model "${model_path}" \
#     --n_token_seq_len "${n_token_seq_len}" \
#     --max_new_seq_len "${max_new_seq_len}" \
#     --data_bos_id "0" \
#     --data_eos_id "2" \
#     --data_start_id "0" \
#     --batch_size "${batch_size}"