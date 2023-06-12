import math
import os
import pickle
from typing import List

import torch
import torch.nn as nn
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from huggingface_hub import snapshot_download
from tqdm import tqdm

import trlx
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

MODEL_BASED = "pvduy/vicuna-13b-v1.1-sft-ver2"
MODEL_BASED_RM = "EleutherAI/gpt-j-6B"
RM_BASED = "reciprocate/dahoas-gptj-rm-static"
RM_REVISION = "676bfd4d"
OUT_DIR = "/mnt/hdd/duyphung/ppo_oa_vicuna_version_1"
DATASET_PATH = "pvduy/oa_vicuna_dolly_grademath_alpaca_leetcode"

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024 + 128,
        epochs=100,
        total_steps=100000,
        batch_size=8,
        minibatch_size=1,
        checkpoint_interval=10000,
        eval_interval=100,
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
            "lr": 1.0e-6,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.00003,
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
        ppo_epochs=4,
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
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.95,
            "temperature": 1,
        },
    ),
)


def create_reward_fn():
    if os.environ.get("RANK", "0") == "0":
        rm_tokenizer = T5Tokenizer.from_pretrained("stanfordnlp/SteamSHP-flan-t5-xl")
        rm_tokenizer.padding_side = "left"
        rm_tokenizer.truncation_side = "left"
        rm_model = T5ForConditionalGeneration.from_pretrained("stanfordnlp/SteamSHP-flan-t5-xl")
        rm_device = torch.cuda.device_count() - 1
        rm_model.to(rm_device)

        def get_reward(prompts, outputs):
            scores = []

            for prompt, output in zip(prompts, outputs):
                input_text = (
                    f"POST: {prompt}\n\n RESPONSE A: {output}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE"
                )
                x = rm_tokenizer([input_text], return_tensors="pt").input_ids.to(rm_device)
                outputs = rm_model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
                score = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:, :]).sum(axis=1).item()
                scores.append(score)
            return scores

        def reward_fn(samples, prompts, original_output, **kwargs):
            samples = [s[s.find("<|prompter|>") :] for s in samples]
            prompts = [p[p.find("<|prompter|>") :] for p in prompts]
            outputs = [s[s.rfind("<|assistant|>") + len("<|assistant|>") :] for s in samples]
            rewards = get_reward(prompts, outputs)
            return rewards

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
        dataset = dataset.drop_duplicates(subset=["prompt"])
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
        stop_sequences=["</s>", "<|prompter|>", "<assistant>"],
    )
