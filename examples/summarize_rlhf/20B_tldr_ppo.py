import os
from typing import List

import pickle

import torch
from datasets import load_dataset
from reward_model.reward_model import GPTRewardModel
from tqdm import tqdm
from transformers import AutoTokenizer
from datasketch import MinHash, MinHashLSH

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "CarperAI/openai_summarize_tldr_sft"

config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=100,
        total_steps=100000,
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=400,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir="/mnt/hdd/duyphung/ppo_pythia20b_sft_summarize_tldr_new",
    ),
    model=ModelConfig(
        model_path="dmayhem93/neox-20B-Summarization-sft",
        num_layers_unfrozen=12,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="EleutherAI/gpt-neox-20b",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-6,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
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
        num_rollouts=16,
        chunk_size=1,
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
            "max_new_tokens": 50,
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.95,
            "temperature": 0.5,
        },
    ),
)


def create_reward_fn():
    if os.environ.get("RANK", "0") == "0":
        # Load the pre-trained reward model
        rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        rw_tokenizer.pad_token = rw_tokenizer.eos_token
        rw_model = GPTRewardModel(SFT_MODEL_PATH)
        rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
        rw_model.half()
        rw_model.eval()
        rw_device = 7  # set reward model device
        rw_model.to(rw_device)
        lsh = pickle.load(open("1B_data/lsh_table_hashing.pickle", "rb"))
        post_summary_dict = pickle.load(open("20B_data/post_summary_dict.pickle", "rb"))
        
        def get_scores(samples: List[str]):
            scores_list = []
            batch_size = 2
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
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

        def reward_fn(samples: List[str], **kwargs):
            original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
            try:
                original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
            except:
                original_samples = []
                for text in samples:
                    try:
                        original_samples.append(text + post_summary_dict[text.strip()])
                    except:
                        try:
                            print("=== Hashing to find similar text... ====")
                            sim_text = min_hash_query(text)
                            original_samples.append(text + post_summary_dict[sim_text])
                            print("Original text: ", text)
                            print("Similar text: ", sim_text)
                            print("========================================")
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

    if 0:
        # Load the pre-trained reward model
        # rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # rw_tokenizer.pad_token = rw_tokenizer.eos_token
        # rw_model = GPTRewardModel(SFT_MODEL_PATH)
        # rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
        # rw_model.half()
        # rw_model.eval()
        # rw_device = torch.device("cuda:{}".format(0))  # set reward model device
        # rw_model.to(rw_device)

        def get_scores(samples: List[str]):
            scores_list = []
            batch_size = 2
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
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

        def get_prompt_dataset(prompts, max_length):
            """
            Get the prompt after T5 decoding to make sure dictionary
            of prompts and summaries is consistent decode prompt from trlX pipeline
            """
            formatted_prompts = []
            for i in tqdm(range(len(prompts))):
                tmp = tokenizer.decode(
                    tokenizer(
                        prompts[i].split("TL;DR:")[0],
                        truncation=True,
                        max_length=max_length - 5,  # to make sure "TL;DR" dont get truncated
                        add_special_tokens=False,
                    )["input_ids"],
                    skip_special_tokens=True,
                ).strip()
                tmp = tmp + "\nTL;DR:"
                tmp = tokenizer.decode(
                    tokenizer(tmp, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"],
                    skip_special_tokens=True,
                ).strip()
                formatted_prompts.append(tmp)
            return formatted_prompts

        def reward_fn(samples: List[str], **kwargs):
            original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
            try:
                original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
            except:
                original_samples = []
                for text in samples:
                    try:
                        original_samples.append(text + post_summary_dict[text.strip()])
                    except:
                        try:
                            print("=== Hashing to find similar text... ====")
                            sim_text = min_hash_query(text)
                            original_samples.append(text + post_summary_dict[sim_text])
                            print("=== Need to find text: {} ====".format(text))
                            print("=== Found similar text: {} ====".format(sim_text))
                        except:
                            original_samples.append(text)

            original_scores = get_scores(original_samples)
            scores = get_scores(samples)
            scores = torch.tensor(scores)
            original_scores = torch.tensor(original_scores)
            norms_scores = scores - original_scores
            return norms_scores
        
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

        dataset = load_dataset("CarperAI/openai_summarize_tldr")

        # Store data into prompt and label pairs
        train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
        val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]

        # Split contents into summaries and labels
        train_posts, train_summaries = zip(*train_set)
        val_posts, val_summaries = zip(*val_set)

        # Get the OpenAI summaries
        post_summary_dict = {}
        train_prompts = get_prompt_dataset(train_posts, max_length_input)
        for i in range(len(train_prompts)):
            post_summary_dict[train_prompts[i]] = train_summaries[i]
        val_prompts = get_prompt_dataset(val_posts, max_length_input)
        for i in range(len(val_prompts)):
            post_summary_dict[val_prompts[i]] = val_summaries[i]

        minhashes = []
        for doc in tqdm(post_summary_dict.keys()):
            words = doc.lower().split()
            m = MinHash(num_perm=128)
            for word in words:
                m.update(word.encode("utf8"))
            minhashes.append(m)

        lsh = MinHashLSH(threshold=0.8, num_perm=128)
        for i, m in enumerate(minhashes):
            lsh.insert(i, m)
        # check 1B_data folder is exist if not create one
        if not os.path.exists("20B_data"):
            os.makedirs("20B_data")
        pickle.dump(lsh, open("20B_data/lsh_table_hashing.pickle", "wb"))
        pickle.dump(post_summary_dict, open("20B_data/post_summary_dict.pickle", "wb"))
        pickle.dump(train_prompts, open("20B_data/train_prompts.pickle", "wb"))
        pickle.dump(val_prompts, open("20B_data/val_prompts.pickle", "wb"))
        reward_fn = create_reward_fn()
    else:
        train_prompts = pickle.load(open("20B_data/train_prompts.pickle", "rb"))
        val_prompts = pickle.load(open("20B_data/val_prompts.pickle", "rb"))
        reward_fn = create_reward_fn()
        
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
