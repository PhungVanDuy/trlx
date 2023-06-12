import json
import os
import sys
import torch
from datasets import load_dataset

from typing import List

import trlx
from trlx.data.default_configs import (
    ILQLConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from transformers import AutoTokenizer
from reward_model.reward_model import GPTRewardModel

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
        seq_length=2048,
        batch_size=6,
        epochs=100,
        total_steps=5000,
        checkpoint_interval=1000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateILQLTrainer",
        checkpoint_dir="checkpoints/ilql_summarize_6B",
    ),
    model=ModelConfig(model_path="EleutherAI/pythia-6.7b-deduped", num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/pythia-6.7b-deduped", truncation_side="right"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1.3e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1000000000, eta_min=1e-6)),
    method=ILQLConfig(
        name="ilqlconfig",
        tau=0.7,
        gamma=0.99,
        cql_scale=0.1,
        awac_scale=1,
        alpha=0.01,
        beta=0,
        steps_for_target_q_sync=1,
        two_qs=True,
        gen_kwargs=dict(max_new_tokens=96, top_k=20, beta=[1, 2, 4], temperature=1.0),
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
        
        def get_scores(samples):
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

        def reward_fn(samples: List[str], **kwargs):
            scores = get_scores(samples)
            scores = torch.tensor(scores)
            return scores

        return reward_fn
    else:
        return True



def main(hparams={}):
    
    def preprocess(sample):
        sample["prompt_output"] = [
            [sample['prompt'] + ' TL;DR:', sample['chosen'][7:]],
            [sample['prompt'] + ' TL;DR:', sample['rejected'][7:]],
        ]
        sample["reward"] = [1, -1]
        return sample

    dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    dataset = dataset.map(preprocess)

    prompts_outputs = sum(dataset['train']['prompt_output'], [])
    rewards = sum(dataset['train']['reward'], [])

    eval_prompts = [
        sample[0][0]
        for sample in dataset['test']['prompt_output']
    ]
    eval_prompts = list(set(eval_prompts))[:1000]
    
    reward_fn = create_reward_fn()
    
    trlx.train(
        samples=prompts_outputs,
        rewards=rewards,
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda **kwargs: {"reward": reward_fn(**kwargs)},
    )



if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)