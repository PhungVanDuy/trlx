import sys
sys.path.insert(0, "supervised_models")
from typing import List

import torch
from transformers import pipeline, GPT2LMHeadRewardModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from summarize_dataset import get_dataset_from_jsonl
import trlx
from trlx.data.configs import TRLConfig


if __name__ == "__main__":
    
    rw_tokenizer = GPT2Tokenizer.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-xl')
    rw_model = GPT2LMHeadRewardModel.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-reward-model-summarize/checkpoint-2000')
    rw_model.resize_token_embeddings(len(rw_tokenizer))
    rw_model.config.pad_token_id = rw_tokenizer.pad_token_id
    rw_model.config.pad_token_id = rw_tokenizer.bos_token_id
    rw_tokenizer.pad_token_id = rw_tokenizer.bos_token_id # should load from checkpoint current just for testing pipeline
    rw_model.eval()
    rw_device = 'cuda:2'
    #rw_device = 'cpu'
    rw_model.to(rw_device)
    
    def reward_fn(samples: List[str]):
        encodings_dict = rw_tokenizer(
                samples, 
                truncation=True, 
                max_length=550, 
                padding="max_length"
        )
        input_ids = torch.tensor(encodings_dict['input_ids']).to(rw_device)
        attn_masks = torch.tensor(encodings_dict['attention_mask']).to(rw_device)
        # duplicate ids and masks
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        scores = torch.nn.functional.normalize(scores.logits[:, 0], dim=0)
        scores -= 0.21
#        scores = torch.clamp(scores.logits[:, 0], -2, 2)
        return scores#scores.logits[:, 0]

    # # Take few words off of movies reviews as prompts
    # train_openai_summ, _ = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/train.jsonl", False)
    # val_openai_summ, _ = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/valid.jsonl", False)
    # test_openai_sum, _ = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/test.jsonl", False)
    
    train_openai_summ, train_labels = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/train.jsonl", False)
    val_openai_summ, val_labels = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/valid.jsonl", False)
    test_openai_sum, test_labels = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/test.jsonl", False)
    
    # total = []
    # batch_size = 16
    # from tqdm import tqdm
    # for i in tqdm(range(0, len(train_openai_summ[:000]), batch_size)):
    #     summs = train_openai_summ[i:i+batch_size]
    #     labels = train_labels[i:i+batch_size]
    #     lst = [(x+y) for x, y in zip(summs, labels)]
    #     scores = reward_fn(lst)
    #     total.append(scores)
    # total = torch.cat(total)
    # mean = torch.mean(total)
    # std = torch.std(total)
    # import ipdb; ipdb.set_trace()
    
    prompts = train_openai_summ # + val_openai_summ

    config = TRLConfig.load_yaml("configs/ppo_config_summ.yml")
    
    model = trlx.train(
        "/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=test_openai_sum[0:100],
        config=config
    )
