config_string = """
train:
  seq_length: 48
  epochs: 100
  total_steps: 1600
  batch_size: 2

  checkpoint_interval: 10000
  eval_interval: 100
  save_best: False
  project_name: "trlx-demo-paper-sentiments"
  entity_name: "pvduy"
  group_name: "Pythia20B"

  pipeline: "PromptPipeline"
  trainer: "AcceleratePPOTrainer"

model:
  model_path:  "EleutherAI/gpt-neox-20b"
  num_layers_unfrozen: {unfrozen_layers}

tokenizer:
  tokenizer_path: "EleutherAI/gpt-neox-20b"
  truncation_side: "right"

optimizer:
  name: "adamw"
  kwargs:
    lr: 1.0e-6
    betas: [0.9, 0.95]
    eps: 1.0e-8
    weight_decay:  1.0e-6


scheduler:
  name: "cosine_annealing"
  kwargs:
    T_max: 10000 # train.total_steps
    eta_min: 1.0e-6

method:
  name: "ppoconfig"
  num_rollouts: 128
  chunk_size: 8
  ppo_epochs: 4
  init_kl_coef: 0.05
  target: 6
  horizon: 10000
  gamma: 1
  lam: 0.95
  cliprange: 0.2
  cliprange_value: 0.2
  vf_coef: 1
  scale_reward: False
  ref_mean: null
  ref_std: null
  cliprange_reward: 10
  gen_kwargs:
    max_new_tokens: 40
    top_k: 0
    top_p: 1.0
    do_sample: True
"""

list_unfrozen_layers = [2, -1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

import os

for unfrozen_layers in list_unfrozen_layers:
    print("Running with unfrozen layers: ", unfrozen_layers)
    config = config_string.format(unfrozen_layers=unfrozen_layers)
    # write to file
    with open("configs/ppo_config_sentiments_pythia20b.yml", "w") as f:
        f.write(config)
    # run training
    os.system("accelerate launch examples/ppo_sentiments_20b.py")
    
