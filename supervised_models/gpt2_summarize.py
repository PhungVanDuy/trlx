import random
import numpy as np
import torch
from datasets import load_metric
import wandb
wandb.init(project="gpt2-supervised-summarize", entity="pvduy")
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
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', 
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>',
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|tl;dr|>"]})

    train_dataset = TLDRDataset("../openai_data/tldr_filtered/train.jsonl", tokenizer, max_length=532)
    dev_dataset = TLDRDataset("../openai_data/tldr_filtered/valid.jsonl", tokenizer, max_length=532)
    test_dataset = TLDRDataset("../openai_data/tldr_filtered/test.jsonl", tokenizer, max_length=532)
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    rouge = load_metric("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4)
        }

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    training_args = TrainingArguments(
        output_dir="gpt2-sup-summ", 
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        half_precision_backend=True,
        gradient_accumulation_steps=8,
        eval_steps=5000,
        save_steps=10000
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

if __name__=="__main__":
    main()
