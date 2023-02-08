import sys
import time
import traceback
from functools import wraps
from typing import List

import numpy as np
import requests
import retry
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

auth_token = "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"

RM_DEVICE = 1

reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_model_path = "ChaiML/reward_2m_gpt2_target_8"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, use_auth_token=auth_token).to(
    RM_DEVICE
)
reward_model.eval()
reward_model.half()

RESPONSE_PENALTY_COEF = 1.2
REWARD_SHIFT = -4.2


def reward_fn(samples, prompts, outputs):
    assert_inputs_are_same_dimension(samples, prompts, outputs)
    zipped_inputs = zip(samples, prompts, outputs)
    scores = [_score_input(s, p, o) for s, p, o in zipped_inputs]
    return scores


def reward_fn_2(samples):
    scores = [_score_input_2(s) for s in samples]
    return scores


def _score_input(sample, prompt, response):
    encoded_input = reward_tokenizer(
        prompt + response.rstrip(), return_tensors="pt", truncation=True, max_length=256
    ).to(reward_model.device)
    logits = reward_model(**encoded_input).logits
    preds = torch.softmax(logits, dim=1)
    # rewards = shifted_logits_with_penalty(logits, response)
    reward = preds[0, 1]
    return reward


def _score_input_2(sample):
    encoded_input = reward_tokenizer(sample, return_tensors="pt", truncation=True, max_length=256).to(
        reward_model.device
    )
    logits = reward_model(**encoded_input).logits
    preds = torch.softmax(logits, dim=1)
    # rewards = shifted_logits_with_penalty(logits, response)
    reward = preds[0, 1]
    return reward


def shifted_logits_with_penalty(logits, response):
    response_len = len(reward_tokenizer.encode(response))
    return logits + REWARD_SHIFT - RESPONSE_PENALTY_COEF * np.exp(1 - response_len)


def assert_inputs_are_same_dimension(x, y, z):
    assert len(x) == len(y)
    assert len(y) == len(z)


class ToyRewardModel:
    def __init__(self):
        self.good_completions = ["Hi my name's"]

    def calculate_reward(self, sentence):
        score = 0.0
        if sentence in self.good_completions:
            score = 1.0
        return score


def dummy_reward_fn(samples: List[str]):
    rm = ToyRewardModel()
    scores = torch.tensor([rm.calculate_reward(s) for s in samples])
    return scores
