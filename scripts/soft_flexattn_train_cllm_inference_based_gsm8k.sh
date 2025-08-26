export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training
export WANDB_RUN_NAME="soft_flexattn_train_cllm_inference_based_gsm8k"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/checkpoint/lhu/models/Abel-7B-001"
trajectory_file="/checkpoint/lhu/data/CLLM2_gsm8k/merged_all.jsonl"
output_path="/checkpoint/lhu/train_ckpts/cllm/one-pass-efficient-train-cllm-openthinker2-7B-ntok16_cllm_soft_flexattn_gsm8k_ar_10"
n_token_seq_size=16
qlora=False

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm_wo_offloading.py \
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
    --model_max_length 4096 \
    --lazy_preprocess True \
    --qlora ${qlora}
