import os

import pandas as pd
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        #        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            "ChaiML/litv2-6B-rev2",
            use_auth_token="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",
            cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2/",
        )
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ChaiML/litv2-6B-rev2",
            use_auth_token="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",
            cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2/",
        )
        #        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        loss = None
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = rewards.shape[0]

        # Get all values not equal to PAD_ID at the end of the sequence
        chosen_end_scores = rewards[torch.arange(bs), attention_mask.sum(1) - 1]
        if labels is None:
            return {
                "pred_length": chosen_end_scores,
            }
        mse = nn.MSELoss()
        loss = mse(chosen_end_scores, labels)
        return {
            "loss": loss,
            "pred_length": chosen_end_scores,
        }


class ChaiDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts_outputs = []
        self.labels = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            self.prompts_outputs.append(row["prompt_input"] + "\n" + row["prompt_output"])
            self.labels.append(row["remaining_user_messages_scale"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt_output = self.prompts_outputs[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(prompt_output, truncation=True, padding="max_length", max_length=self.max_length)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": float(label),
        }


model = GPTRewardModel("/fsx/home-duyphung/models/litv2-6B-rev2")
layers = model.transformer.h
num_layers = len(layers)
num_unfrozen = int(0.3 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)
tokenizer = AutoTokenizer.from_pretrained(
    "ChaiML/litv2-6B-rev2",
    use_auth_token="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",
    cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2",
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
df_train = pd.read_parquet("1M_train_chai.parquet")
df_val = pd.read_parquet("10k_val_chai.parquet").sample(100)
train_dataset = ChaiDataset(df_train, tokenizer)
val_dataset = ChaiDataset(df_val, tokenizer)


def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    mse = sum(((predictions - labels) ** 2)) / len(predictions)
    return {"mse": mse}


print("starting training")

training_args = TrainingArguments(
    output_dir="/fsx/home-duyphung/chai_ml/rm_checkpoint",
    num_train_epochs=1,
    logging_steps=10,
    gradient_accumulation_steps=1,
    save_strategy="steps",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
    eval_steps=5000,
    save_steps=5000,
    adam_beta1=0.9,
    adam_beta2=0.95,
    warmup_steps=100,
    logging_dir="./logs",
    fp16=True,
    learning_rate=1e-6,
    deepspeed="ds_config_gpt_j.json",
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train("/fsx/home-duyphung/chai_ml/rm_checkpoint/checkpoint-5000")
