import os
os.system('cp modeling_gpt2.py /fsx/home-duyphung/anaconda3/envs/py39/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py')

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2LMHeadRewardModel
from summarize_dataset import ComparisionDataset
from transformers import TrainingArguments, Trainer, default_data_collator
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union


import wandb
wandb.init(project="gpt2-supervised-summarize-reward", entity="pvduy")

# class RewardOutput(ModelOutput):

#     loss: Optional[torch.FloatTensor] = None
#     r0: Optional[torch.FloatTensor] = None
#     r1: Optional[torch.FloatTensor] = None
    
class DataCollatorReward:
    
    def __call__(self, features):
        input_ids_0 = torch.stack([f["input_ids"][0] for f in features])
        input_ids_1 = torch.stack([f["input_ids"][1] for f in features])
        attention_mask_0 = torch.stack([f["attention_mask"][0] for f in features])
        attention_mask_1 = torch.stack([f["attention_mask"][1] for f in features])
        # labels = [f["labels"] for f in features]
        input_ids = torch.cat([input_ids_0, input_ids_1], dim=0)
        attention_mask = torch.cat([attention_mask_0, attention_mask_1], dim=0)
        batch = {}
        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        # batch['labels'] = labels
        return batch
        
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



if __name__ == "__main__":
    set_seed()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    train_dataset = ComparisionDataset("../openai_data/comparisons/train_comparisons.jsonl", tokenizer)
    dev_dataset = ComparisionDataset("../openai_data/comparisons/valid_comparisons.jsonl", tokenizer)
    test_dataset = ComparisionDataset("../openai_data/comparisons/test_comparisons.jsonl", tokenizer)
    gpt2model = GPT2LMHeadModel.from_pretrained("gpt2-sup-summ-ver2/checkpoint-10000", use_cache=False)
    gpt2model.resize_token_embeddings(len(tokenizer))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    model = GPT2LMHeadRewardModel(gpt2model.config)
    model.get_pretrained_model(gpt2model)
    training_args = TrainingArguments(
        output_dir="gpt2-reward-checkpoint-ver2", 
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        half_precision_backend=True,
        logging_steps=30,
        gradient_accumulation_steps=8,
        eval_steps=50000,
        save_steps=2000
    )
    def compute_metrics(eval_preds):
        import ipdb; ipdb.set_trace()
        pred_ids = eval_preds.predictions
        acc = sum(eval_preds.predictions[0] >= eval_preds.predictions[1]) / len(eval_preds.predictions[0])
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
