export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training

model_path="/checkpoint/lhu/models/OpenThinker2-7B"
trajectory_file="/checkpoint/lhu/data/CLLM2_openthought/train_openthoughts_split_ratio_8_size_14244_ntok_size_64_lookup_size_640_sampling_ratio_0.2_eos_tokens_termination_with_think_format_without_sysmsg.json"
output_path="/checkpoint/lhu/train_ckpts/cllm/cllm-openthinker2-7B-ntok32-eos_tokens-without_think_format_splitratio_8_size_14244_ntok_sampling_ratio_0.2_lookup_size_640"
n_token_seq_size=64
qlora=False

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/train_cllm.py \
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
    --qlora ${qlora} \
    --deepspeed "scripts/ds_config.json"
