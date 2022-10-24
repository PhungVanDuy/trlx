import os
os.system('cp modeling_gpt2.py /fsx/home-duyphung/anaconda3/envs/py39/lib/python3.9/site-packages/transformers/models/gpt2/modeling_gpt2.py')

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, GPT2LMHeadRewardModel

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import ComparisionDataset
from transformers import TrainingArguments, Trainer, default_data_collator, GPT2ForSequenceClassification
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
from tqdm import tqdm

import wandb
wandb.init(project="gpt2-supervised-summarize", entity="pvduy")

class RewardOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    r1: Optional[torch.FloatTensor] = None
    r2: Optional[torch.FloatTensor] = None


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def data_collator_prepare(features):
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

if __name__ == "__main__":
    set_seed()
    rw_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    train_dataset = ComparisionDataset("../openai_data/comparisons/train_comparisons.jsonl", rw_tokenizer)
    dev_dataset = ComparisionDataset("../openai_data/comparisons/valid_comparisons.jsonl", rw_tokenizer)
    test_dataset = ComparisionDataset("../openai_data/comparisons/test_comparisons.jsonl", rw_tokenizer)
    gpt2model = GPT2LMHeadRewardModel.from_pretrained('gpt2-reward-checkpoint-ver2/checkpoint-8000')
    gpt2model.resize_token_embeddings(len(rw_tokenizer))
    gpt2model.config.pad_token_id = rw_tokenizer.pad_token_id
    gpt2model.config.pad_token_id = rw_tokenizer.bos_token_id
    rw_tokenizer.pad_token_id = rw_tokenizer.bos_token_id

    from torch.utils.data import DataLoader
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=False, batch_size=32, collate_fn=data_collator_prepare
    )
    gpt2model.cuda()
    gpt2model.eval()
    correct = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = gpt2model(**batch)
            correct += sum(outputs.logits[:, 0] > outputs.logits[:, 1])
    print("Total accuracy: ", correct / len(dev_dataset))
