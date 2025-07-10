# CLLM2
1. How to to generate trajectory now?
```
cd /data/phd/kousiqi/kousiqi/CLLM2
bash scripts/generate_trajectory.sh
```

2. How to train cllm2 now?  
```
conda activate /data/phd/kousiqi/kousiqi/envs/cllm2
cd /data/phd/kousiqi/kousiqi/CLLM2
# gradient_accumulation_steps can be modified in ds_config.json
bash scripts/train_cllm.sh
```

3. How to evaluate cllm2 now?  
```
conda activate /data/phd/kousiqi/kousiqi/envs/cllm2
cd /data/phd/kousiqi/kousiqi/CLLM2
CUDA_VISIBLE_DEVICES=0 python random_init.py
```
