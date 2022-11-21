import random
import numpy as np
import torch
from datasets import load_metric
import evaluate
import os
import wandb
wandb.init(project="gpt2-supervised-summarize", entity="pvduy")
from summarize_dataset import TLDRDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
# deepspeed train_gptneo_summarize.py --config ds_config_gpt_neo_27.json
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

import argparse



def main():
    # parser = argparse.ArgumentParser(description='Training GPT-2 Supervised learning')
    # parser.add_argument('--base_model', type=str, default='gpt2-xl', help='GPT2 base model name or path: [gpt2, gpt2-xl, gpt2-medium, gpt2-large]')
    # parser.add_argument('--dataset-dir', type=str, default='/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/tldr_filtered', help='Path to dataset directory')
    # parser.add_argument('--max_input_length', type=int, default=550, help='Max input length')
    # parser.add_argument('--output_dir', type=str, default='gpt2-supervised-summarize-checkpoint', help='Output directory')
    # parser.add_argument('--train_batch_size', type=int, default=1, help='Training batch size')
    # parser.add_argument('--eval_batch_size', type=int, default=1, help='Evaluation batch size')
    # parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs')
    # parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    # parser.add_argument('--warmup_steps', type=int, default=300, help='Warmup steps')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    # parser.add_argument('--logging_steps', type=int, default=30, help='Logging steps')
    # parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation steps')
    # parser.add_argument('--save_steps', type=int, default=2000, help='Save steps')
    # parser.add_argument('--ds_config', type=str, default='ds_config_zero3.json', help='DeepSpeed config file')

    # args = parser.parse_args()
    # import pickle
    #pickle.dump(args, open("args_train_sup.pkl", "wb"))
    #exit()
    import pickle
    args = pickle.load(open("args_train_sup.pkl", "rb"))
    args.output_dir = "gptneo-supervised-summarize-checkpoint"
    args.train_batch_size = 16
    args.gradient_accumulation_steps = 1
    args.learning_rate = 2.5e-5
    args.eval_batch_size = 1
    args.eval_steps = 200
    args.max_input_length = 550
    args.save_steps = 1000
    random.seed(42)
    # Load the GPT tokenizer.
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", use_cache=False)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")#, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token    
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    train_dataset = TLDRDataset(os.path.join(args.dataset_dir, "train.jsonl"), tokenizer, max_length=args.max_input_length)
    dev_dataset = TLDRDataset(os.path.join(args.dataset_dir, "valid.jsonl"), tokenizer, max_length=args.max_input_length)
    test_dataset = TLDRDataset(os.path.join(args.dataset_dir, "test.jsonl"), tokenizer, max_length=args.max_input_length)
    # model = GPT2LMHeadModel.from_pretrained(args.base_model, use_cache=False)
    # model.resize_token_embeddings(len(tokenizer))

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
            logits = logits[0]
        return logits.argmax(dim=-1)

    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
        fp16=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=100,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        deepspeed='./ds_config_gpt_neo_27.json'
        # deepspeed=args.ds_config
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
    trainer.save_model(args.output_dir)

if __name__=="__main__":
    main()
