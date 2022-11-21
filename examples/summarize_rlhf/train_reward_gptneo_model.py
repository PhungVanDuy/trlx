import os

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from reward_gptneo_model import GPTJRewardModel
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


    set_seed()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")#, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    base = AutoModelForCausalLM.from_pretrained("/fsx/home-duyphung/refactor_summarize_rlhf/trlx/examples/summarize_rlhf/gptneo-supervised-summarize-checkpoint/checkpoint-1000", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token    
    base.resize_token_embeddings(len(tokenizer))
    model = GPTJRewardModel(base.config)
    model.get_pretrained_model(base)
    
    
    train_dataset = ComparisionDataset(os.path.join("/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons", "train_comparisons.jsonl"), tokenizer)
    dev_dataset = ComparisionDataset(os.path.join("/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons", "valid_comparisons.jsonl"), tokenizer)
    test_dataset = ComparisionDataset(os.path.join("/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons", "test_comparisons.jsonl"), tokenizer)
    training_args = TrainingArguments(
        output_dir="/fsx/home-duyphung/refactor_summarize_rlhf/trlx/examples/summarize_rlhf/gptneo-reward-summarize-checkpoint", 
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_checkpointing=True,
        half_precision_backend=True,
        logging_steps=50,
        gradient_accumulation_steps=2,
        fp16=True,
        eval_steps=200,
        save_steps=1000,
        warmup_steps=100,
        num_train_epochs=5,
        learning_rate=2.5e-5,
        deepspeed='./ds_config_gpt_neo_27.json'
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
