import sys
from typing import List

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from reward_model import GPT2LMHeadRewardModel
from summarize_dataset import get_dataset_from_jsonl
import trlx
from trlx.data.configs import TRLConfig
import argparse
import os


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Test Reward Model')
    parser.add_argument('--base_model', type=str, default='gpt2-xl', help='Path to base model')
    parser.add_argument('--reward_model_path', type=str, default='/fsx/home-duyphung/trlx/supervised_models/gpt2-reward-model-summarize/checkpoint-2000', help='Path to checkpoint of trained reward model')
    parser.add_argument('--dataset-dir', type=str, default='/fsx/home-duyphung/trlx/openai_data/tldr_filtered', help='Path to dataset directory')
    parser.add_argument('--max_input_length', type=int, default=550, help='Max input length')
    parser.add_argument('--output_dir', type=str, default='gpt2-reward-summarize-checkpoint', help='Output directory')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--cuda-device', type=int, default=2, help='Cuda device')
    
    args = parser.parse_args()
    
    rw_tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    rw_model = GPT2LMHeadRewardModel.from_pretrained(args.reward_model_path)
    rw_model.resize_token_embeddings(len(rw_tokenizer))
    rw_model.config.pad_token_id = rw_tokenizer.pad_token_id
    rw_model.config.pad_token_id = rw_tokenizer.bos_token_id
    rw_tokenizer.pad_token_id = rw_tokenizer.bos_token_id
    rw_model.eval()
    rw_model.to(f"cuda:{args.cuda_device}")
    
    def reward_fn(samples: List[str]):
        original_samples = [text.split('TL;DR:')[0] + 'TL;DR: ' for text in samples]
        original_samples = [text + train_post_summ[text] for text in original_samples]
        
        encodings_dict = rw_tokenizer(
                original_samples, 
                truncation=True, 
                max_length=args.max_input_length, 
                padding="max_length"
        )
        input_ids = torch.tensor(encodings_dict['input_ids']).to(rw_device)
        attn_masks = torch.tensor(encodings_dict['attention_mask']).to(rw_device)
        # duplicate ids and masks
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            scores_ref = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        
        encodings_dict = rw_tokenizer(
                samples, 
                truncation=True, 
                max_length=args.max_input_length, 
                padding="max_length"
        )
        input_ids = torch.tensor(encodings_dict['input_ids']).to(rw_device)
        attn_masks = torch.tensor(encodings_dict['attention_mask']).to(rw_device)
        # duplicate ids and masks
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        return scores.logits[:, 0] - scores_ref.logits[:, 0] # normalize by truth score

    train_openai_summ, train_labels = get_dataset_from_jsonl(os.path.join(args.dataset_dir, "train.jsonl"), False)
    val_openai_summ, val_labels = get_dataset_from_jsonl(os.path.join(args.dataset_dir, "valid.jsonl"), False)
    test_openai_sum, test_labels = get_dataset_from_jsonl(os.path.join(args.dataset_dir, "test.jsonl"), False)
    
    train_post_summ = {}
    for i in range(len(train_openai_summ)):
        tmp = rw_tokenizer.decode(rw_tokenizer(train_openai_summ[i])['input_ids'])
        train_post_summ[tmp] = train_labels[i]

    
    prompts = train_openai_summ # + val_openai_summ

    config = TRLConfig.load_yaml("ppo_config_summ.yml")
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=test_openai_sum[0:100],
        config=config
    )
