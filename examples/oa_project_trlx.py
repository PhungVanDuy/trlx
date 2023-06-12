import os
import pickle
from typing import List

import torch
import torch.nn as nn
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
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
        model = AutoModelForCausalLM.from_pretrained("/mnt/nvme/home/duyphung/FastChat/vicuna-13b-fine-tuned")
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme/home/duyphung/FastChat/vicuna-13b-fine-tuned")
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
        eval_interval=400,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="checkpoints/ppo_oa_llama",
    ),
    model=ModelConfig(
        model_path="/mnt/nvme/home/duyphung/FastChat/vicuna-13b-fine-tuned",
        num_layers_unfrozen=8,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="/mnt/nvme/home/duyphung/FastChat/vicuna-13b-fine-tuned",
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
        rw_tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme/home/duyphung/FastChat/vicuna-13b-fine-tuned")
        rw_tokenizer.padding_side = "right"
        rw_model = GPTRewardModel()
        rw_model.load_state_dict(
            torch.load(
                "/mnt/nvme/home/duyphung/chai/oa-project-rm/rm_checkpoint/checkpoint-5000/global_step20000/mp_rank_00_model_states.pt"
            )["module"]
        )
        rw_model.half()
        rw_model.eval()
        rw_device = 7  # set reward model device
        rw_model.to(rw_device)
        lsh = pickle.load(open("oa_data/lsh_table_hashing.pickle", "rb"))
        post_summary_dict = pickle.load(open("oa_data/post_label_dict.pickle", "rb"))

        def get_scores(samples: List[str]):
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

        def min_hash_query(query):
            words = query.lower().split()
            m = MinHash(num_perm=128)
            for word in words:
                m.update(word.encode("utf8"))
            result = lsh.query(m)
            return list(post_summary_dict.keys())[result[0]]

        def reward_fn(samples, prompts, outputs):
            original_samples = prompts
            try:
                original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
            except:
                original_samples = []
                for text in prompts:
                    try:
                        original_samples.append(text + post_summary_dict[text.strip()])
                    except:
                        try:
                            sim_text = min_hash_query(text)
                            original_samples.append(text + post_summary_dict[sim_text])
                        except:
                            original_samples.append(text)
            original_scores = get_scores(original_samples)
            scores = get_scores(samples)
            scores = torch.tensor(scores)
            original_scores = torch.tensor(original_scores)
            norms_scores = scores - original_scores
            return norms_scores

        return reward_fn
    else:
        return True


if __name__ == "__main__":
    if 1:
        import pandas as pd
        from datasets import load_dataset

        ds = load_dataset("pvduy/instruct_sft_data_without_oig_vicuna_format", split="train")

        dataset = ds.to_pandas()
        # split pandas dataset into train and validation random
        from sklearn.model_selection import train_test_split

        train, val = train_test_split(dataset, test_size=1000, random_state=42)

        # Store data into prompt and label pairs
        train_set = [(sample["prompt"], sample["label"]) for _, sample in train.iterrows()]
        val_set = [(sample["prompt"], sample["label"]) for _, sample in val.iterrows()]

        train_posts, train_labels = zip(*train_set)
        val_posts, val_labels = zip(*val_set)

        train_prompts = train["prompt"].tolist()
        val_prompts = val["prompt"].tolist()

        post_label_dict = {}
        for i in range(len(train_prompts)):
            post_label_dict[train_prompts[i]] = train_labels[i]
        for i in range(len(val_prompts)):
            post_label_dict[val_prompts[i]] = val_labels[i]

        minhashes = []
        for doc in tqdm(post_label_dict.keys()):
            words = doc.lower().split()
            m = MinHash(num_perm=128)
            for word in words:
                m.update(word.encode("utf8"))
            minhashes.append(m)

        lsh = MinHashLSH(threshold=0.8, num_perm=128)
        for i, m in enumerate(minhashes):
            lsh.insert(i, m)
        pickle.dump(lsh, open("oa_data/lsh_table_hashing.pickle", "wb"))
        pickle.dump(post_label_dict, open("oa_data/post_label_dict.pickle", "wb"))
        pickle.dump(train_prompts, open("oa_data/train_prompts.pickle", "wb"))
        pickle.dump(val_prompts, open("oa_data/val_prompts.pickle", "wb"))
        exit()
    else:
        train_prompts = pickle.load(open("oa_data/train_prompts.pickle", "rb"))
        import random

        # shuffle train prompts
        random.shuffle(train_prompts)
        # sampling 20% of train prompts
        train_prompts = train_prompts[0 : int(len(train_prompts) * 0.2)]
        val_prompts = pickle.load(open("oa_data/val_prompts.pickle", "rb"))
        print("Length of train prompts: ", len(train_prompts))
        print("Length of val prompts: ", len(val_prompts))
        reward_fn = create_reward_fn()

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
        stop_sequences=["### Human:", "Human:"],
    )
