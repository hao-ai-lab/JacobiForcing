#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import math
import pathlib
from typing import Dict, Optional, List, Any

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import default_collate
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import logging
from datasets import load_dataset

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index  # -100

@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",
        metadata={"help": "Path or HF id for the base model"},
    )
    qlora: bool = field(default=False, metadata={"help": "Enable QLoRA"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data JSONL"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024, metadata={"help": "Max seq length for tokenizer/model"}
    )
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to reduce activation memory"},
    )
    bf16: bool = field(
        default=True, metadata={"help": "Train in bfloat16 for efficiency."}
    )
    fp16: bool = field(default=False)

class LabelsDataset(Dataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        ex = self.ds[idx]
        if "labels_ids" not in ex:
            raise KeyError("Each JSONL row must contain 'labels_ids'.")
        seq = ex["labels_ids"]
        if not isinstance(seq, list) or len(seq) == 0:
            raise ValueError("'labels_ids' must be a non-empty list of ints.")
        return {"labels_ids": seq}

def make_collate_fn(tokenizer: transformers.PreTrainedTokenizer, max_len: int):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Trim each sequence at the FIRST eos
        trimmed = []
        for ex in batch:
            seq = ex["labels_ids"]
            if eos_id is not None and eos_id in seq:
                first_eos = seq.index(eos_id)
                seq = seq[: first_eos + 1]  # keep eos, drop after
            trimmed.append(seq)

        # target length
        seq_lens = [len(s) for s in trimmed]
        target_len = min(max(seq_lens), max_len)

        input_ids_list, labels_list, attn_list = [], [], []
        for seq in trimmed:
            seq = seq[:target_len]
            pad_needed = target_len - len(seq)

            # after-eos positions are removed; pad to target_len.
            ids = seq + [pad_id] * pad_needed
            lbl = seq + [IGNORE_TOKEN_ID] * pad_needed
            attn = [1] * len(seq) + [0] * pad_needed

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(lbl, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))

        input_ids = torch.stack(input_ids_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        attention_mask = torch.stack(attn_list, dim=0)
        return {
            "input_ids": input_ids, 
            "labels": labels, 
            "attention_mask": attention_mask
        }

    return collate

def safe_save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    to_save = model
    if hasattr(to_save, "module"):
        to_save = to_save.module
    to_save.save_pretrained(output_dir, safe_serialization=False)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()

    # Config
    config = transformers.AutoConfig.from_pretrained(
        model_args.target_model_path, cache_dir=training_args.cache_dir
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        raise ValueError(
            f"model_max_length ({training_args.model_max_length}) exceeds model context ({orig_ctx_len})."
        )
    config.use_cache = False

    # Model + Tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.target_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None),
        low_cpu_mem_usage=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.target_model_path, padding_side="right", use_fast=False
    )
    # If we added a new pad token above, resize embeddings
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) > model.get_input_embeddings().weight.size(0):
        model.resize_token_embeddings(len(tokenizer))

    if getattr(training_args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    if model_args.qlora:
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=16, lora_dropout=0.05)
        model = get_peft_model(model, lora_cfg)

    hf_ds = load_dataset("json", data_files={"train": data_args.data_path},
                         split="train", cache_dir=training_args.cache_dir)
    train_dataset = LabelsDataset(hf_ds)

    data_collator = make_collate_fn(tokenizer, training_args.model_max_length)

    # --- Regular HF Trainer ---
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train(resume_from_checkpoint=True if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) else None)

    # Save
    safe_save_model(model, tokenizer, training_args.output_dir)

if __name__ == "__main__":
    train()
