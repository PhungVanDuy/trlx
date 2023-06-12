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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

MODEL_BASED = "pvduy/vicuna-13b-v1.1"
MODEL_BASED_RM = "pvduy/vicuna-13b-v1.1-rm-formated"
RM_MODEL = "pvduy/vicuna-13b-v1.1-rm-formated"
RM_REVISION = "main"
OUT_DIR = "/mnt/hdd/duyphung/ppo_oa_vicuna_v1.1_no_sft"
DATASET_PATH = "pvduy/sharegpt_alpaca_oa_vicuna_format"


class GPTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(MODEL_BASED_RM)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_BASED_RM)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.PAD_ID = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        ends = torch.argmax((input_ids == self.PAD_ID).type(torch.float32), dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)
        return rewards


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
            "weight_decay": 1.0e-6,
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
        init_kl_coef=0.1,
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
        cliprange_reward=2,
        gen_kwargs={
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 0.5,
        },
    ),
)


def create_reward_fn():
    if os.environ.get("RANK", "0") == "0":
        # Load the pre-trained reward model
        rw_tokenizer = AutoTokenizer.from_pretrained(MODEL_BASED_RM)
        rw_tokenizer.pad_token = rw_tokenizer.eos_token
        rw_tokenizer.pad_token_id = rw_tokenizer.eos_token_id
        rw_tokenizer.padding_side = "right"
        rw_model = GPTRewardModel()
        directory = snapshot_download(RM_MODEL, revision=RM_REVISION)
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint_path = os.path.join(directory, fpath)
                break

        rw_model.load_state_dict(torch.load(checkpoint_path)["module"])
        rw_model.half()
        rw_model.eval()
        rw_device = torch.cuda.device_count() - 1
        rw_model.to(rw_device)

        def get_reward(samples: List[str]):
            scores_list = []
            batch_size = 40
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = [chosen for chosen in sub_samples]
                encodings_dict = rw_tokenizer(
                    sub_samples,
                    truncation=True,
                    max_length=config.train.seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to(rw_device)
                attn_masks = encodings_dict["attention_mask"].to(rw_device)
                with torch.no_grad():
                    sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
                scores_list.append(sub_scores)
            scores = torch.cat(scores_list, dim=0)
            return scores

        def reward_fn(samples, prompts, original_output, **kwargs):
            rewards = get_reward(samples)
            original_samples = [p + o for p, o in zip(prompts, original_output)]
            original_rewards = get_reward(original_samples)
            return rewards.squeeze() - original_rewards.squeeze()

    else:
        return True
    return reward_fn


if __name__ == "__main__":
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset(DATASET_PATH)
    train = ds["train"].to_pandas()
    val = ds["test"].to_pandas().sample(n=1000)

    train_prompts = [{"prompt": x["prompt"], "original_output": x["label"]} for _, x in train.iterrows()]
    val_prompts = [{"prompt": x["prompt"], "original_output": x["label"]} for _, x in val.iterrows()]

    reward_fn = create_reward_fn()

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
        stop_sequences=["USER:", "</s>", "ASSISTANT:"],
    )
