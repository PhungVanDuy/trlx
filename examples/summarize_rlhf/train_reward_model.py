import os

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from reward_model import GPT2LMHeadRewardModel
from summarize_dataset import ComparisionDataset
from transformers import TrainingArguments, Trainer, default_data_collator
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
import argparse


import wandb
wandb.init(project="gpt2-supervised-summarize-reward", entity="pvduy")


class DataCollatorReward:
    
    def __call__(self, features):
        input_ids_0 = torch.stack([f["input_ids"][0] for f in features]) # input ids of the post + first summary
        input_ids_1 = torch.stack([f["input_ids"][1] for f in features]) # input ids of the post + second summary
        attention_mask_0 = torch.stack([f["attention_mask"][0] for f in features])
        attention_mask_1 = torch.stack([f["attention_mask"][1] for f in features])
        input_ids = torch.cat([input_ids_0, input_ids_1], dim=0)
        attention_mask = torch.cat([attention_mask_0, attention_mask_1], dim=0)
        batch = {}
        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        batch['labels'] = torch.tensor([0] * input_ids_0.shape[0] + [1] * input_ids_1.shape[0])
        return batch
        
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Reward Model')
    parser.add_argument('--base_model', type=str, default='gpt2-xl', help='Path to base model')
    parser.add_argument('--supervised_model_path', type=str, default='/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize', help='Path to checkpoint of trained supervised model')
    parser.add_argument('--dataset-dir', type=str, default='/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons', help='Path to dataset directory')
    parser.add_argument('--max_input_length', type=int, default=550, help='Max input length')
    parser.add_argument('--output_dir', type=str, default='gpt2-reward-summarize-checkpoint', help='Output directory')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=300, help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--logging_steps', type=int, default=30, help='Logging steps')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=2000, help='Save steps')
    parser.add_argument('--ds_config', type=str, default='ds_config_zero3.json', help='DeepSpeed config file')
    

    args = parser.parse_args()
    set_seed()
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    train_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "train_comparisons.jsonl"), tokenizer)
    dev_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "valid_comparisons.jsonl"), tokenizer)
    test_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "test_comparisons.jsonl"), tokenizer)
    gpt2model = GPT2LMHeadModel.from_pretrained(args.supervised_model_path , use_cache=False)
    gpt2model.resize_token_embeddings(len(tokenizer))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model = GPT2LMHeadRewardModel(gpt2model.config)
    model.get_pretrained_model(gpt2model)
    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        evaluation_strategy="steps",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=1,
        gradient_checkpointing=True,
        half_precision_backend=True,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        deepspeed=args.ds_config
    )
    
    def compute_metrics(eval_preds):
        acc = sum(eval_preds.predictions[0][:, 0] >= eval_preds.predictions[0][:, 1]) / eval_preds.predictions[0].shape[0]
        return {"accuracy": acc}

    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=dev_dataset,
        data_collator=data_collator
    )
    trainer.train()
