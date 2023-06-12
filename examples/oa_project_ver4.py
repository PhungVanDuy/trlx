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


class GPTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained("pvduy/vicuna-13b-v1.1-sft")
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("pvduy/vicuna-13b-v1.1-sft")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False

        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


config = TRLConfig(
    train=TrainConfig(
        seq_length=768,
        epochs=100,
        total_steps=100000,
        batch_size=8,
        minibatch_size=1,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="/mnt/hdd/duyphung/ppo_oa_vicuna_v1.1_ver4",
    ),
    model=ModelConfig(
        model_path="pvduy/vicuna-13b-v1.1-sft",
        num_layers_unfrozen=8,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="pvduy/vicuna-13b-v1.1-sft",
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
        num_rollouts=128,
        chunk_size=16,
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
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
        },
    ),
)


def create_reward_fn():
    if os.environ.get("RANK", "0") == "0":
        # Load the pre-trained reward model
        rw_tokenizer = AutoTokenizer.from_pretrained("pvduy/vicuna-13b-v1.1-sft")
        rw_tokenizer.pad_token = rw_tokenizer.eos_token
        rw_tokenizer.pad_token_id = rw_tokenizer.eos_token_id
        rw_tokenizer.padding_side = "right"
        rw_model = GPTRewardModel()
        directory = snapshot_download("pvduy/vicuna-13b-v1.1-rm", revision="058bcbd")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint_path = os.path.join(directory, fpath)
                break

        rw_model.load_state_dict(torch.load(checkpoint_path)["module"])
        rw_model.half()
        rw_model.eval()
        rw_device = 7  # set reward model device
        rw_model.to(rw_device)

        def get_reward(samples: List[str]):
            scores_list = []
            batch_size = 2
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
                input_ids = input_ids.repeat(2, 1)
                attn_masks = attn_masks.repeat(2, 1)
                with torch.no_grad():
                    sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
                scores_list.append(sub_scores["chosen_end_scores"])
            scores = torch.cat(scores_list, dim=0)
            return scores

        def reward_fn(samples, prompts, original_output, **kwargs):
            rewards = get_reward(samples)
            original_samples = [p + o for p, o in zip(prompts, original_output)]
            original_rewards = get_reward(original_samples)
            return rewards - original_rewards

    else:
        return True
    return reward_fn


if __name__ == "__main__":
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("pvduy/instruct_sft_data_without_oig_vicuna_format", split="train")

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
        stop_sequences=["USER:", "</s>", "ASSISTANT:"],
    )
