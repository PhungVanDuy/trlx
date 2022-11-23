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
import wandb

wandb.init(project="trlx", name="trlx-gpt2-summarize", entity="pvduy")

if __name__ == "__main__":
    
    
    rw_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    rw_model = GPT2LMHeadRewardModel.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-reward-model-summarize/checkpoint-2000')
    rw_model.resize_token_embeddings(len(rw_tokenizer))
    rw_model.config.pad_token_id = rw_tokenizer.pad_token_id
    rw_model.config.pad_token_id = rw_tokenizer.bos_token_id
    rw_tokenizer.pad_token_id = rw_tokenizer.bos_token_id
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(3))
    rw_model.to(rw_device)
    
    def reward_fn(samples: List[str]):
        lst_scores = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i:i+batch_size]
            encodings_dict = rw_tokenizer(
                    sub_samples, 
                    truncation=True, 
                    max_length=550, 
                    padding="max_length"
            )
            input_ids = torch.tensor(encodings_dict['input_ids']).to(rw_device)
            attn_masks = torch.tensor(encodings_dict['attention_mask']).to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            lst_scores.append(sub_scores.logits[:, 0])
        scores = torch.cat(lst_scores, dim=0)
        norms_scores = scores
        return norms_scores

    train_openai_summ, train_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/trlx/openai_data/tldr_filtered", "train.jsonl"), False)
    val_openai_summ, val_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/trlx/openai_data/tldr_filtered", "valid.jsonl"), False)
    test_openai_sum, test_labels = get_dataset_from_jsonl(os.path.join("/fsx/home-duyphung/trlx/openai_data/tldr_filtered", "test.jsonl"), False)
    
    train_post_summ = {}
    for i in range(len(train_openai_summ)):
        tmp = rw_tokenizer.decode(rw_tokenizer(train_openai_summ[i])['input_ids'])
        train_post_summ[tmp] = train_labels[i]
    
    for i in range(len(val_openai_summ)):
        tmp = rw_tokenizer.decode(rw_tokenizer(val_openai_summ[i])['input_ids'])
        train_post_summ[tmp] = val_labels[i]

    prompts = val_openai_summ # train_openai_summ + val_openai_summ
    config = TRLConfig.load_yaml("ppo_config_summ_neo.yml")
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_openai_summ[0:100],
        config=config
    )
