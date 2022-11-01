import random
import numpy as np
import torch
from datasets import load_metric
import evaluate
import wandb
wandb.init(project="gpt2-supervised-summarize-ver2", entity="pvduy")
from summarize_dataset import TLDRDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator
)


def main():
    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = TLDRDataset("../openai_data/tldr_filtered/train.jsonl", tokenizer, max_length=550)
    dev_dataset = TLDRDataset("../openai_data/tldr_filtered/valid.jsonl", tokenizer, max_length=550)
    test_dataset = TLDRDataset("../openai_data/tldr_filtered/test.jsonl", tokenizer, max_length=550)
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    rouge = evaluate.load('rouge')
    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    training_args = TrainingArguments(
        output_dir="gpt2-supervised-summarize", 
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        half_precision_backend=True,
        gradient_accumulation_steps=8,
        num_train_epochs=5,
        warmup_steps=300,
        eval_steps=1000,
        save_steps=2000,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    trainer.train()
    trainer.save_model('best_gpt2xl_summ-ver2')

if __name__=="__main__":
    main()
