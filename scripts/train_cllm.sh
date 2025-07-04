export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=cllm2_distillation

model_path="/data/phd/kousiqi/kousiqi/ckpts/OpenThinker2-7B"
trajectory_file="data/collected_diffusion_trajectory/merged_output.json"
output_path="/data/phd/kousiqi/kousiqi/ckpts/cllm-openthinker2-7B"
n_token_seq_size=64
qlora=False

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint='localhost:5666' \
    --master_port 10001 \
    train/train_cllm.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 2 \
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
    --qlora ${qlora} \
    --deepspeed "scripts/ds_config.json"