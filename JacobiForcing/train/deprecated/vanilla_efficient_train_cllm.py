# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, List

import os
import sys
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
import datasets
import deepspeed
import wandb
wandb.login(key="1a7d1c011ac84de1bcea83e2ee65dee6a19ba2ff")

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from typing import Dict

from efficient_cllm_trainer import CllmTrainer

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

import logging
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",  metadata={"help": "Path to target model"})
    qlora: Optional[bool] = field(default=False, metadata={"help": "Enable QLoRA processing"})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_new_tokens: int = field(
        default=16,
        metadata={
            "help": "Size of n_token_sequence in Jacobi trajectory."
        },
    )
    report_to: str = field(
        default='wandb',
        metadata={
            'help': 'The list of integrations to report the results and logs to.'
        }
    )
    use_gt_labels: bool = False
    remove_unused_columns: bool = field(default=False)

def rank0_print(local_rank, *args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess_training_sequence(
    prompt_ids: List[int],
    prompt_ids_len: int,
    complete_training_sequence_ids: List[int],
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    labels_ids: List[int],
    traj_position_indices: List[int],
) -> Dict:
    """
    Preprocess a single training example for model input.
    """
    prompt_ids = torch.tensor(prompt_ids)
    complete_training_sequence_ids = torch.tensor(complete_training_sequence_ids)
    attention_mask = prompt_ids.ne(tokenizer.pad_token_id)

    result = dict(
        prompt_ids=prompt_ids,
        prompt_ids_len=prompt_ids_len,
        input_ids=complete_training_sequence_ids,
        attention_mask=attention_mask,
        labels_ids=labels_ids,
        traj_position_indices=traj_position_indices
    )
    return result
    
class JacobianDataset(Dataset):
    """Dataset for consistency training."""

    def __init__(self, raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: str,
                 do_eval: bool = False,
                 local_rank: int = -1):
        super().__init__()
        self.tokenizer = tokenizer
        rank0_print(local_rank, "Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        sample = self.raw_data[i]
        ret = preprocess_training_sequence(
            sample["prompt_ids"],
            sample["prompt_ids_len"],
            sample["complete_training_sequence_ids"],
            self.tokenizer,
            self.model,
            labels_ids=sample["labels_ids"],
            traj_position_indices=sample["traj_position_indices"],
        )
        self.cached_data_dict[i] = ret
        return ret


def make_jacobian_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    trajectory_data,
    data_args,
    model: str,
    local_rank: int,
) -> Dict:
    """Make dataset and collator for consistency training."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = JacobianDataset
    rank0_print("Loading data...")
            
    train_dataset = dataset_cls(trajectory_data,
                                tokenizer=tokenizer,
                                model=model,
                                local_rank=local_rank)
    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ["LOCAL_RANK"])
    training_args.local_rank = local_rank
    training_args.qlora = model_args.qlora
    
    torch.set_default_dtype(torch.float)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load config:
    config = transformers.AutoConfig.from_pretrained(
        model_args.target_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        raise ValueError(
            f"Use rope scaling is not supported: "
            f"model_max_length ({training_args.model_max_length}) > orig_ctx_len ({orig_ctx_len})"
        )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.target_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.target_model_path,
        padding_side="right",
        use_fast=False,
    )
    # if 'vicuna' in model_args.target_model_path:
    #     tokenizer.pad_token = tokenizer.unk_token

    if model_args.qlora:
        # Runs w/ qLoRA when qlora tag is enabled is enabled
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
        )
    
        model = get_peft_model(model, config)
        model.config.use_cache = False

    # Load data
    trajectory_dataset = load_dataset(
        'json',
        data_files={
            'train': data_args.data_path,
        },
        split='train',
        cache_dir='/checkpoint/lhu/data/CLLM2_openthought/trajectory_cache'
    )
    data_module = make_jacobian_data_module(tokenizer=tokenizer,
                                            trajectory_data=trajectory_dataset,
                                            data_args=data_args,
                                            model=model_args.target_model_path,
                                            local_rank=training_args.local_rank)

    trainer = CllmTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
