import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from reward_model.reward_model import GPTRewardModel
from trlx.data.configs import TRLConfig

config_path = "configs/ilql_summarize.yml"
default_config = yaml.safe_load(open(config_path))
triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-summarize")

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "EleutherAI/gpt-j-6B"


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    config = TRLConfig.load_yaml(config_path)

    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = GPTRewardModel(SFT_MODEL_PATH)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)

    def reward_fn(samples):
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

    # def reward_fn(samples):
    #     input = reward_tokenizer(samples, padding=True, max_length=1024)

    #     mbs = 24
    #     out = []
    #     for i in range(math.ceil(len(samples) / mbs)):
    #         batch_ixs = slice(i * mbs, (i + 1) * mbs)
    #         input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)

    #         inputs = [
    #             prepare_tensor("input_ids", input_ids),
    #         ]

    #         result = client.infer(triton_model, inputs)
    #         rewards = result.as_numpy("rewards")
    #         if rewards is None:
    #             raise RuntimeError("No output data")

    #         out.extend(rewards.flatten())

    #     return out

    def preprocess(sample):
        sample["prompt_output"] = [
            [sample["prompt"] + " TL;DR:", sample["chosen"][7:]],
            [sample["prompt"] + " TL;DR:", sample["rejected"][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    if 0:
        dataset = pd.read_parquet(
            "/admin/home-duyphung/software/github/pvd/trlx/examples/summarize_rlhf/ilql_openai_tldr_train.parquet"
        )
        val_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
        prompts_outputs = []
        rewards = []
        for _, row in dataset.iterrows():
            sample = [row["input"], row["output"]]
            prompts_outputs.append(sample)
            rewards.append(row["score"])

        eval_prompts = list(val_dataset["prompt"])[:1000]
    else:
        dataset = load_dataset("CarperAI/openai_summarize_comparisons")
        dataset = dataset.map(preprocess)

        prompts_outputs = sum(dataset["train"]["prompt_output"], [])
        rewards = sum(dataset["train"]["reward"], [])
        val_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
        eval_prompts = list(val_dataset["prompt"])[:1000]
    trainer = trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=lambda samples, **kwargs: {"rewards": reward_fn(samples)},
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
