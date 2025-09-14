export CUDA_VISIBLE_DEVICES=6,7
export WANDB_PROJECT=cllm2_training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/home/lah003/models/Qwen2.5-Coder-7B-Instruct"
#trajectory_file="/checkpoint/lhu/data/CLLM2_openthought/merged/merged_data_v2_8_27_opencodeinstruct.jsonl"
trajectory_file="/home/lah003/data/merged_09_12_4pm_w16_min0_max1_reverse_progressive/merged_09_12_4pm_w16_min0_max1_reverse_progressive.jsonl"
output_path="/home/lah003/workspace/CLLM2/ckpts/bk32_w16_min0_max1"
n_token_seq_size=32
window_size=16
qlora=False

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm_multiblock_window.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --window_size ${window_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --qlora ${qlora}
