import os
from typing import List

import pickle

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH
from transformers import AutoModelForCausalLM
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
RM_BASED = "Dahoas/gptj-rm-static"
RM_REVISION = "676bfd4d"
OUT_DIR = "/mnt/hdd/duyphung/ppo_oa_vicuna_version_1"
DATASET_PATH = "pvduy/oa_vicuna_dolly_grademath_alpaca_leetcode"
    
config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=100,
        total_steps=100000,
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir=OUT_DIR,
    ),
    model=ModelConfig(
        model_path=MODEL_BASED,
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=MODEL_BASED,
        truncation_side="left",
        padding_side="left",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=64,
        chunk_size=4,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
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

        reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.truncation_side = "left"

        class RewardModel(nn.Module):
            def __init__(self, checkpoint_path, eos_token_id):
                super().__init__()
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                self.transformer = model.transformer
                self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
                self.eos_token_id = eos_token_id

            def forward(self, input_ids):
                states = self.transformer(input_ids)[0]
                rewards = self.v_head(states).squeeze(-1)
                ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
                returns = torch.gather(rewards, 1, ends).squeeze(-1)
                return returns

        reward_model = RewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
        directory = snapshot_download("Dahoas/gptj-rm-static", revision=RM_REVISION)
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint))
        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(reward_device)
        reward_batch_size = 48
        delta_reward = True

        def get_reward(samples):
            input = reward_tokenizer(
                samples,
                padding=True,
                truncation=True,
                max_length=reward_tokenizer.max_len_single_sentence,
                return_tensors="pt",
            ).to(reward_device)

            mbs = reward_batch_size
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = reward_model(input_ids)
                out.extend(rewards)
            return torch.hstack(out)

        def reward_fn(samples, prompts, original_output, **kwargs):
            samples = [s + reward_tokenizer.eos_token for s in samples]
            rewards = get_reward(samples)

            if not delta_reward:
                return rewards

            original_samples = [p + o + reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards
    else:
        return True
    return reward_fn



if __name__ == "__main__":

    import pandas as pd
    from datasets import load_dataset
    
    ds = load_dataset(DATASET_PATH)["train"]
    # train = ds["train"].to_pandas()
    # val = ds["test"].to_pandas().sample(n=1000)
    
    dataset = ds.to_pandas()
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