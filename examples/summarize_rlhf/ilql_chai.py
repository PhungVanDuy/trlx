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
from transformers import AutoModelForCausalLM, AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from examples.summarize_rlhf.scores import reward_fn, reward_fn_2
from reward_model.reward_model import GPTRewardModel
from trlx.data.configs import TRLConfig

config_path = "configs/ilql_chai.yml"


class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        #        model = AutoModelForCausalLM.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            "ChaiML/litv2-6B-rev2",
            use_auth_token="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",
            cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2/",
        )
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ChaiML/litv2-6B-rev2",
            use_auth_token="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",
            cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2/",
        )
        #        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        loss = None
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = rewards.shape[0]

        # Get all values not equal to PAD_ID at the end of the sequence
        chosen_end_scores = rewards[torch.arange(bs), attention_mask.sum(1) - 1]
        if labels is None:
            return {
                "pred_length": chosen_end_scores,
            }
        mse = nn.MSELoss()
        loss = mse(chosen_end_scores, labels)
        return {
            "loss": loss,
            "pred_length": chosen_end_scores,
        }


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    rw_model = GPTRewardModel("ChaiML/litv2-6B-rev2")
    rw_model.load_state_dict(torch.load("/fsx/home-duyphung/chai_ml/rm_checkpoint/checkpoint-55000/pytorch_model.bin"))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:{}".format(1))  # set reward model device
    rw_model.to(rw_device)
    rw_tokenizer = AutoTokenizer.from_pretrained(
        "ChaiML/litv2-6B-rev2",
        use_auth_token="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",
        cache_dir="/fsx/home-duyphung/models/litv2-6B-rev2",
    )
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_tokenizer.pad_token_id = rw_tokenizer.eos_token_id

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
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
            scores_list.append(sub_scores["pred_length"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    config = TRLConfig.load_yaml(config_path)

    tokenizer_input = AutoTokenizer.from_pretrained("gpt2", truncation_side="left", padding_side="right")

    def preprocess(sample):
        input_ids = sample["input_ids"]
        prompt = tokenizer_input.decode(input_ids)
        prompt_input = "\n".join(prompt.split("\n")[:-1])
        prompt_output = prompt.split("\n")[-1]
        sample["prompt_input"] = prompt_input
        sample["prompt_output"] = prompt_output
        return sample

    if 0:
        ds = load_dataset(
            "ChaiML/50mConversations", use_auth_token="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", cache_dir="chaiml_50m"
        )
        ds = ds.map(preprocess, num_proc=12)

    else:
        df_train = pd.read_parquet(
            "/admin/home-duyphung/software/github/pvd/trlx/examples/summarize_rlhf/1M_train_chai.parquet"
        )
        # df_train = df_train.sample(n=100_000)
        prompts_outputs = []
        for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
            prompts_outputs.append([row["prompt_input"], row["prompt_output"]])
        rewards = df_train["remaining_user_messages_scale"].tolist()
        df_val = pd.read_parquet("/fsx/home-duyphung/all_summarize_data/summ/56k_chai_convs.parquet").sample(n=1000)
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
