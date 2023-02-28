import os

import evaluate
from datasets import load_dataset, load_from_disk, load_metric
from torch.utils.data import dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MAX_LENGTH_INPUT = 512
MAX_LENGTH_OUTPUT = 50


class Seq2SeqDataset(dataset.Dataset):
    def __init__(self, tokenizer, type_data="train"):
        # Set up the datasets
        data_path = "CarperAI/openai_summarize_tldr"
        if type_data == "train":
            dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
        else:
            dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid").select(range(2000))
        self.prompts = []
        self.outputs = []
        responses = dataset["label"]
        inputs = dataset["prompt"]
        for i, res in tqdm(enumerate(responses), total=len(responses)):
            self.prompts.append(inputs[i])
            self.outputs.append(res)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        input_text = self.prompts[idx]
        output_text = self.outputs[idx]

        model_input = self.tokenizer(input_text, max_length=MAX_LENGTH_INPUT, padding="max_length", truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(output_text, max_length=MAX_LENGTH_OUTPUT, padding="max_length", truncation=True)[
                "input_ids"
            ]
            model_input["labels"] = labels
            model_input["labels"] = [
                -100 if token == self.tokenizer.pad_token_id else token for token in model_input["labels"]
            ]
        return model_input


import wandb

wandb.init(name="flan-t5-finetune", project="add_t5", entity="pvduy")


if __name__ == "__main__":
    config = {
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 500,
        "batch_size": 4,
        "batch_size_val": 4,
        "warmup_steps": 200,
        "accum_steps": 2,
        "num_beams": 3,
        "output_dir": "flan-t5-qa",
    }

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str,
            references=label_str,
        )

        return rouge_output

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        do_train=True,
        num_train_epochs=3,
        do_eval=False,
        predict_with_generate=True,
        evaluation_strategy="steps",
        adam_beta1=0.9,
        adam_beta2=0.95,
        learning_rate=1e-6,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size_val"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        eval_accumulation_steps=1,
        bf16=True,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=config["accum_steps"],
        deepspeed="ds_config_trlx_t5.json",
        save_total_limit=2,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    rouge = evaluate.load("rouge")

    train_dataset = Seq2SeqDataset(tokenizer, type_data="train")
    val_dataset = Seq2SeqDataset(tokenizer, type_data="val")
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {params}")

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train("flan-t5-qa/checkpoint-2000/")
