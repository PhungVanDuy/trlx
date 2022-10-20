import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import ComparisionDataset
from transformers import TrainingArguments, Trainer, default_data_collator, GPT2ForSequenceClassification
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union

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

class RewardModel(nn.Module):

    def __init__(self, gpt2model):
        super(RewardModel, self).__init__()
        self.gpt2model = gpt2model
        self.linear = nn.Linear(1280, 1, bias=False)
    
    def forward(self, input_ids_0, input_ids_1, attention_mask_0, attention_mask_1, labels=None):
        output_0 = self.gpt2model(input_ids_0, attention_mask=attention_mask_0, output_hidden_states=True)
        output_1 = self.gpt2model(input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True)
        hidden_states_0 = output_0.hidden_states[-1]
        hidden_states_1 = output_1.hidden_states[-1]
        logits_0 = self.linear(hidden_states_0)
        logits_1 = self.linear(hidden_states_1)
        batch_size, _ = input_ids_0.shape[:2]
        if self.gpt2model.config.pad_token_id is None:
            sequence_lengths_0 = -1
            sequence_lengths_1 = -1
        else:
            sequence_lengths_0 = torch.ne(input_ids_0, self.gpt2model.config.pad_token_id).sum(-1) - 1
            sequence_lengths_1 = torch.ne(input_ids_1, self.gpt2model.config.pad_token_id).sum(-1) - 1
        
        r0 = logits_0[torch.arange(batch_size, device=logits_0.device), sequence_lengths_0]
        r1 = logits_1[torch.arange(batch_size, device=logits_1.device), sequence_lengths_1]
        r0 = r0.squeeze(-1)
        r1 = r1.squeeze(-1)
        if labels is None:
            return RewardOutput(r1=r0, r2=r1)
        loss = torch.mean(-torch.log(torch.sigmoid(r0 - r1)))
        return RewardOutput(loss=loss, r1=r0, r2=r1)

if __name__ == "__main__":
    set_seed()
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>'
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|tl;dr|>"]})
    """
    tokenizer = GPT2Tokenizer.from_pretrained('../trlx/checkpoint_supervised_gpt_2')
    train_dataset = ComparisionDataset("../openai_data/comparisons/train_comparisons.jsonl", tokenizer)
    dev_dataset = ComparisionDataset("../openai_data/comparisons/valid_comparisons.jsonl", tokenizer)
    test_dataset = ComparisionDataset("../openai_data/comparisons/test_comparisons.jsonl", tokenizer)
    gpt2model = GPT2LMHeadModel.from_pretrained('../trlx/checkpoint_supervised_gpt_2')
    gpt2model.resize_token_embeddings(len(tokenizer))
    gpt2model.config.pad_token_id = tokenizer.pad_token_id
    model = RewardModel(gpt2model)
    training_args = TrainingArguments(
        output_dir="gpt2-reward-checkpoint-new", 
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        half_precision_backend=True,
        logging_steps=30,
        gradient_accumulation_steps=4,
        eval_steps=500,
        save_steps=2000
    )
    def compute_metrics(eval_preds):
        pred_ids = eval_preds.predictions
        acc = sum(eval_preds.predictions[0] >= eval_preds.predictions[1]) / len(eval_preds.predictions[0])
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=dev_dataset,
        data_collator=default_data_collator
    )
    trainer.train('gpt2-reward-checkpoint-new/checkpoint-8000/')
