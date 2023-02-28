import math
import os

import numpy as np
import pandas as pd
import torch
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from reward_model.reward_model import GPTRewardModel
from trlx.data.configs import TRLConfig

config_path = "configs/ilql_summarize_t5.yml"
default_config = yaml.safe_load(open(config_path))

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "CarperAI/openai_summarize_tldr_sft"

triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "openai_summarize_tldr_rm_checkpoint")


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    config = TRLConfig.load_yaml(config_path)

    reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def reward_fn(samples):
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

    def preprocess(sample):
        sample["prompt_output"] = [
            [sample["prompt"] + " TL;DR:", sample["chosen"][7:]],
            [sample["prompt"] + " TL;DR:", sample["rejected"][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    dataset["train"] = dataset["train"]
    dataset = dataset.map(preprocess)

    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])
    val_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    eval_prompts = list(val_dataset["prompt"])[:1000]

    # dataset = pd.read_parquet(
    #     "/admin/home-duyphung/software/github/pvd/trlx/examples/summarize_rlhf/ilql_openai_tldr_train.parquet"
    # )
    # val_dataset = load_dataset("CarperAI/openai_summarize_tldr", split="valid")
    # prompts_outputs = []
    # rewards = []
    # for _, row in dataset.iterrows():
    #     sample = [row["input"], row["output"]]
    #     prompts_outputs.append(sample)
    #     rewards.append(row["score"])

    eval_prompts = list(val_dataset["prompt"])[:1000]

    trlx.train(
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
