import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../supervised_models")
from typing import List

import torch
from transformers import pipeline

import wandb
from trlx.data.configs import TRLConfig
from trlx.model.accelerate_ppo_model import AcceleratePPOModel
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline.ppo_pipeline import PPOPipeline, PPOPipelineSumm
from trlx.utils.loading import get_model, get_orchestrator, get_pipeline
from reward_model import RewardModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer


if __name__ == "__main__":
    cfg = TRLConfig.load_yaml("configs/ppo_config_summ.yml")

    def reward_fn(samples: List[str]):
        samples = [text.replace("<|endoftext|>", "") + " <|endoftext|>" for text in samples]
        encodings_dict = rw_tokenizer(
                samples, 
                truncation=True, 
                max_length=532, 
                padding="max_length"
        )
        input_ids = torch.tensor(encodings_dict['input_ids']).to("cuda:1")
        attn_masks = torch.tensor(encodings_dict['attention_mask']).to("cuda:1")
        scores = rw_model(input_ids, input_ids, attn_masks, attn_masks)
        scores = torch.tensor(scores.r1)
        return scores
    
    model: AcceleratePPOModel = get_model(cfg.model.model_type)(cfg)
    if model.accelerator.is_main_process:
        wandb.watch(model.model)

    pipeline: PPOPipelineSumm = get_pipeline(cfg.train.pipeline)(model.tokenizer, cfg)
    orch: PPOOrchestrator = get_orchestrator(cfg.train.orchestrator)(
        model, pipeline, reward_fn=reward_fn, chunk_size=cfg.method.chunk_size
    )
    rw_model = RewardModel(orch.ref_model.gpt) # should load from checkpoint current just for testing pipeline
    rw_model.load_state_dict(torch.load("checkpoint_reward_gpt2/pytorch_model.bin"))
    rw_model.to("cuda:1")
    rw_tokenizer = pipeline.tokenizer
    rw_model.eval()

    orch.make_experience(cfg.method.num_rollouts)
    model.learn()

    print("DONE!")
