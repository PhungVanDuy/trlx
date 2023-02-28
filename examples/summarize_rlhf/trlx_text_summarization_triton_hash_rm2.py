import difflib
import math
import os
import pathlib
import pickle
import random
import string
from typing import List

import numpy as np
import pandas as pd
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from reward_model.reward_model import GPTRewardModel
from trlx.data.configs import TRLConfig

triton_host = "localhost:8001"
triton_model = "openai_summarize_tldr_rm_checkpoint"


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


if __name__ == "__main__":
    # Load the pre-trained reward model
    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def get_scores(samples):
        input = reward_tokenizer(samples, padding=True, max_length=550)

        mbs = 24
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

            inputs = [
                prepare_tensor("input_ids", input_ids),
            ]
            result = client.infer(triton_model, inputs)
            rewards = result.as_numpy("rewards")
            if rewards is None:
                raise RuntimeError("No output data")

            out.extend(rewards.flatten())

        return out

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
                    except:
                        original_samples.append(text)

        original_scores = get_scores(original_samples)
        scores = get_scores(samples)
        scores = torch.tensor(scores)
        original_scores = torch.tensor(original_scores)
        norms_scores = scores - original_scores
        return norms_scores

    config_path = pathlib.Path(__file__).parent.joinpath("configs/ppo_config_summ.yml")
    config = TRLConfig.load_yaml(config_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    max_length_input = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    if 1:
        dataset = load_dataset("CarperAI/openai_summarize_tldr")

        # Store data into prompt and label pairs
        train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"].select(range(1000))]
        val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"].select(range(1000))]

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
    else:
        lsh = pickle.load(open("cache_tldr/minhash_lsh_summarize.pickle", "rb"))
        train_prompts = pickle.load(open("cache_tldr/train_prompts_summarize.pickle", "rb"))
        val_prompts = pickle.load(open("cache_tldr/val_prompts_summarize.pickle", "rb"))
        post_summary_dict = pickle.load(open("cache_tldr/post_summary_dict_summarize.pickle", "rb"))

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts[0:1000],  # sampling 1000 validation prompts for evaluation speed in training
        config=config,
    )
