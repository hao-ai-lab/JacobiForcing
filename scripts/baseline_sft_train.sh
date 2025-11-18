export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="sft_baseline_data_v1"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
trajectory_file="/checkpoint/lhu/data/postprocessed_trajectory_collection_merged/openhought2_sft_bs_k8s_postprocessed_merged_all.jsonl"
output_path="/checkpoint/lhu/train_ckpts/cllm/sft_baseline_data_v1"
qlora=False

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/baseline_sft_train.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
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
    --qlora ${qlora}
