export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=cllm2_training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/data/numa0/train-tests/ckpts/Qwen2.5-Coder-7B-Instruct"
trajectory_file="/data/numa0/train-tests/data/OpenCodeInstruct_training_data_n32w16/merged-traj-data-oct-16-n32w16.jsonl"
output_path="/data/numa0/train-tests/ckpts/rl/batched-noise-conditioned-0102-n32w16_distill_data_v2_cyclic_progressiv_a1c10_lr1e-6"

n_token_seq_size=32
qlora=False

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm_multiblock_batched.py \
    --target_model_path ${model_path} \
    --data_path ${trajectory_file} \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 8 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --qlora ${qlora}

