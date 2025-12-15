export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=cllm2_training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="/checkpoint/lhu/models/Qwen2.5-Coder-7B-Instruct"
#trajectory_file="/checkpoint/lhu/data/CLLM2_openthought/merged/merged_data_v2_8_27_opencodeinstruct.jsonl"
trajectory_file="/checkpoint/lhu/data/CLLM2_openthought/merged/merged_data_v2_8_27_opencodeinstruct_randomly_sampled_pt.jsonl"
output_path="/checkpoint/lhu/train_ckpts/cllm/logitsaligned-8-28-cllm-qwen2p5-coder-7B-ntok16_ce_soft_loss_flexattn_oci_data_v1_437k_samples_ar_10_random_noise_lr5e-6"
n_token_seq_size=16
qlora=False

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    train/soft_flexattn_train_cllm_multiblock.py \
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
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --qlora ${qlora}
