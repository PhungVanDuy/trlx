import os

import random
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from reward_model import GPT2LMHeadRewardModel, RewardOutput

from summarize_dataset import ComparisionDataset
from tqdm import tqdm
import argparse

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
    parser = argparse.ArgumentParser(description='Test Reward Model')
    parser.add_argument('--base_model', type=str, default='gpt2-xl', help='Path to base model')
    parser.add_argument('--reward_model_path', type=str, default='/fsx/home-duyphung/trlx/supervised_models/gpt2-reward-model-summarize/checkpoint-2000', help='Path to checkpoint of trained reward model')
    parser.add_argument('--dataset-dir', type=str, default='/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons', help='Path to dataset directory')
    parser.add_argument('--max_input_length', type=int, default=550, help='Max input length')
    parser.add_argument('--output_dir', type=str, default='gpt2-reward-summarize-checkpoint', help='Output directory')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Evaluation batch size')
    
    set_seed()
    args = parser.parse_args()
    rw_tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)

    train_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "train_comparisons.jsonl"), rw_tokenizer)
    dev_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "valid_comparisons.jsonl"), rw_tokenizer)
    test_dataset = ComparisionDataset(os.path.join(args.dataset_dir, "test_comparisons.jsonl"), rw_tokenizer)
    gpt2model = GPT2LMHeadRewardModel.from_pretrained(args.reward_model_path)
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