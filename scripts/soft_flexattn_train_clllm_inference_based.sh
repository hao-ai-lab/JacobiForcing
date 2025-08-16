export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=cllm2_training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
trajectory_file="/checkpoint/lhu/data/postprocessed_trajectory_collection_merged/14k_filtered_output_postprocessed_merged_all.jsonl"
output_path="/checkpoint/lhu/train_ckpts/cllm/one-pass-efficient-train-cllm-openthinker2-7B-ntok64-eos_tokens-lookup_size_640_cllm_soft_loss_length_capped_16k_flexattn_siqi_data_1"
n_token_seq_size=64
qlora=False

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --lazy_preprocess True \
    --qlora ${qlora}
