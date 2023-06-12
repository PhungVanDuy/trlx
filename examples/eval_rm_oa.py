import os
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from time import time
import wandb
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import math

class GPTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
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
    


def plot_calibration(model_name: str, dataset_name: str, delta_scores: np.ndarray) -> str:
    space = np.linspace(0, 4, 32)
    perfect_calibration = 1 / (1 + np.exp(-space))

    epsilon = 1 / 4
    probs = []
    for center in space:
        ixs = (center - epsilon < abs(delta_scores)) & (abs(delta_scores) < center + epsilon)
        prob = sum(delta_scores[ixs] > 0) / len(ixs)
        probs.append(prob)

    import matplotlib
    from matplotlib import pyplot

    textcolor = "#333"
    matplotlib.style.use("ggplot")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 15,
        "text.color": textcolor,
        "axes.labelcolor": textcolor,
        "xtick.color": textcolor,
        "ytick.color": textcolor,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.titlesize": 14,
        "figure.figsize": (12, 8),
    })
    pyplot.plot(space, perfect_calibration, label="Perfect calibration", c="grey")
    pyplot.plot(space, probs, label=model_name)

    ax = pyplot.gca()
    ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
    ax.set_facecolor("#fff")
    ax.set_title(f"Preference calibration on {dataset_name}", size=26, y=1.02, fontdict={"fontweight": "normal"})
    ax.set_xlabel("Score difference", size=26)
    ax.set_ylabel("Accuracy", size=26)
    pyplot.legend(loc="best", fontsize=20, title_fontproperties={"weight": "normal", "style": "normal"}, fancybox=False, frameon=False)
    pyplot.tight_layout()

    os.makedirs("calibrations", exist_ok=True)
    image_path = os.path.join("calibrations", f"{model_name}@{dataset_name}.png".replace("/", "_"))
    pyplot.savefig(image_path, dpi=64)
    pyplot.clf()
    return image_path

class RewardModelAlex(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns

if __name__ == "__main__":
    seed = int(os.environ.get("RANK", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_name = "stable_vicuna-rm"

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="autocrit",
        init_kwargs={"wandb": {"name": f"{model_name}"}},
    )
    RM_BASED = "/mnt/hdd/duyphung/stable-vicuna-oa-rm-v1"
    reward_tokenizer = AutoTokenizer.from_pretrained(RM_BASED, use_fast=False)
    reward_model = AutoModelForSequenceClassification.from_pretrained(RM_BASED)
    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_device = "cuda:0"
    reward_model.to(reward_device)

    dataset = load_dataset("pvduy/hh_shp_oa_gpt4_rm_dataset_vicuna_formatoa")
    reward_batch_size = 4
    def get_reward(samples):
        input = reward_tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=reward_tokenizer.max_len_single_sentence,
            return_tensors="pt",
        )

        mbs = reward_batch_size
        out = []
        for i in tqdm(range(math.ceil(len(samples) / mbs))):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = input.input_ids[batch_ixs].to(reward_device)
            attention_mask = input.attention_mask[batch_ixs].to(reward_device)
            with torch.no_grad():
                rewards = reward_model(input_ids, attention_mask=attention_mask).logits.squeeze(-1).detach().cpu()
            out.extend(rewards)
        return torch.hstack(out)
    
    df = dataset['test'].to_pandas().iloc[0:1000, ]
    scores_selected = get_reward(df["chosen"].tolist())
    scores_rejected = get_reward(df["rejected"].tolist())
    delta_scores = scores_selected - scores_rejected
    delta_scores = delta_scores.cpu().numpy()
    
    accuracy = sum((delta_scores > 0)) / len(delta_scores)

    image_path = plot_calibration(model_name, "vicuna_comp", delta_scores)

    accelerator.log({
        "accuracy": accuracy,
        "delta_scores": delta_scores,
        "calibration": wandb.Image(image_path),
    })
    