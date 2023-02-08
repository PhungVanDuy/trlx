import math
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tritonclient.utils import np_to_triton_dtype

import trlx
from examples.summarize_rlhf.scores import reward_fn, reward_fn_2
from trlx.data.configs import TRLConfig

config_path = "configs/ilql_chai.yml"


def main(hparams={}):
    rw_model = AutoModelForSequenceClassification.from_pretrained(
        "Jellywibble/12m-retry-continue-combined-regressor-epoch-1"
    )
    rw_tokenizer = AutoTokenizer.from_pretrained("Jellywibble/12m-retry-continue-combined-regressor-epoch-1")
    rw_tokenizer.pad_token_id = 50256
    rw_tokenizer.truncation_side = "left"
    rw_tokenizer.padding_side = "right"
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 16
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            encodings_dict = rw_tokenizer(
                sub_samples,
                return_tensors="pt",
                return_attention_mask=True,
                padding="longest",
                truncation=True,
                max_length=256,
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            with torch.no_grad():
                reward = rw_model(input_ids, attention_mask=attn_masks).logits.detach().cpu().squeeze()
            scores_list.append(reward)
        scores = torch.cat(scores_list, dim=0)
        return scores

    config = TRLConfig.load_yaml(config_path)

    df_train = pd.read_parquet(
        "/admin/home-duyphung/software/github/pvd/trlx/examples/summarize_rlhf/1M_train_chai_retry.parquet"
    )
    prompts_outputs = []
    for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
        prompts_outputs.append([row["prompt_input"], row["prompt_output"]])
    rewards = df_train["retry_reward"].tolist()
    df_val = pd.read_parquet("/fsx/home-duyphung/all_summarize_data/summ/56k_chai_convs.parquet").sample(n=20)
    eval_prompts = df_val["text_edit"].tolist()

    trainer = trlx.train(
        dataset=(prompts_outputs, rewards),
        metric_fn=lambda samples, **kwargs: {"rewards": get_scores(samples)},
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
