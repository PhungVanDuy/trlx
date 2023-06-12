import os
from typing import List

import pickle

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import math
from trlx.models.modeling_ppo import PPOConfig


MODEL_BASED = "pvduy/vicuna-13b-v1.1-sft-ver2"
MODEL_BASED_RM = "EleutherAI/gpt-j-6B"
RM_BASED = "reciprocate/dahoas-gptj-rm-static"
RM_REVISION = "676bfd4d"
OUT_DIR = "/mnt/hdd/duyphung/stable_vicuna_version_2.1_inprogress"
DATASET_PATH = "pvduy/oa_vicuna_dolly_grademath_alpaca_leetcode"
    
config = TRLConfig(
    train=TrainConfig(
        seq_length=1024+128,
        epochs=100,
        total_steps=100000,
        batch_size=1,
        minibatch_size=1,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir=OUT_DIR,
    ),
    model=ModelConfig(
        model_path=MODEL_BASED,
        num_layers_unfrozen=12,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=MODEL_BASED,
        truncation_side="left",
        padding_side="left",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-6,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=4,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=None,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=5,
        gen_kwargs={
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1,
            "temperature": 1,
        },
    ),
)


def create_reward_fn():
    if os.environ.get("RANK", "0") == "0":
        tokenizer = AutoTokenizer.from_pretrained("reciprocate/gpt-j_rm_format-oa", revision="501f895")
        tokenizer.truncation_side = "left"

        rm_model = AutoModelForSequenceClassification.from_pretrained("reciprocate/gpt-j_rm_format-oa", revision="501f895")
        rm_model.requires_grad_(False)
        rm_device = torch.cuda.device_count() - 1
        rm_model = rm_model.eval().half().to(rm_device)

        def get_reward(samples: List[str]):
            all_scores = []
            batch_size = 40
            for i in range(0, len(samples), batch_size):
                batch = tokenizer(
                    samples[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                ).to(rm_device)

                with torch.no_grad():
                    scores = rm_model(**batch)[0].squeeze(-1).cpu()
                all_scores.append(scores)
            scores = torch.hstack(all_scores)
            return scores

        def reward_fn(samples, prompts, original_output, **kwargs):
            samples = [
                s[s.find("<|system|>")] if "<|system|>" in s else 
                s[s.find("<|prompter|>"):] for s in samples
            ]
            prompts = [
                p[p.find("<|system|>")] if "<|system|>" in p else
                p[p.find("<|prompter|>"):] for p in prompts
            ]
            original_samples = [p + o for p, o in zip(prompts, original_output)]
            samples = [x + "<|endoftext|>" for x in samples]
            original_samples = [x + "<|endoftext|>" for x in original_samples]
            rewards = get_reward(samples)
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards
    else:
        return True
    return reward_fn


if __name__ == "__main__":

    import pandas as pd
    from datasets import load_dataset
    if 0:
        ds = load_dataset(DATASET_PATH)["train"]
        # train = ds["train"].to_pandas()
        # val = ds["test"].to_pandas().sample(n=1000)
        
        dataset = ds.to_pandas().sample(frac=1).reset_index(drop=True)
        # drop duplicates by prompt column in pandas dataset
        dataset = dataset.drop_duplicates(subset=['prompt'])
    else:
        dataset = pd.read_parquet("instruct_data_trlx_version_1.parquet")
    # split pandas dataset into train and validation random
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(dataset, test_size=1000, random_state=42)
    
    train_prompts = [{"prompt": x["prompt"], "original_output": x["label"]} for _, x in train.iterrows()]
    val_prompts = [{"prompt": x["prompt"], "original_output": x["label"]} for _, x in val.iterrows()]

    reward_fn = create_reward_fn()

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
        stop_sequences=["</s>", "<|prompter|>", "<assistant>"]
    )
