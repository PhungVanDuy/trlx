import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy, AutoModel, AutoConfig, PreTrainedModel
import json
from reward_model import GPTRewardModel
import deepspeed
from tqdm import tqdm

    
def create_comparision_dataset(path):
    
    def make_text(post, summarize):
        return f"SUBREDDIT: r/{post['subreddit']}\nTITLE: {post['title']}\nPOST: {post['post']}\nTL;DR: {summarize}"
    
    with open(path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    if "valid" in path:
        dataset = dataset[0:5000]
        
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        post = sample['info']
        chosen_summary = sample['summaries'][sample['choice']]['text']
        rejected_summary = sample['summaries'][1 - sample['choice']]['text']
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair['chosen'] = make_text(post, chosen_summary)
        pair['rejected'] = make_text(post, rejected_summary)
        pairs.append(pair)
    return pairs
         

# class PairwiseTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # forward pass
#         rewards = model(**inputs)
#         rewards_chunked = rewards.view((2, -1))
#         chosen_rewards = rewards_chunked[0]
#         rejected_rewards = rewards_chunked[1]
#         # compute pairwise loss
#         loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
#         return (loss, outputs) if return_outputs else loss

# class PairwiseTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # forward pass
#         assert len(inputs["input_ids"].shape) == 2
#         bs = inputs["input_ids"].shape[0] // 2
#         chosen = inputs["input_ids"][:bs]
#         rejected = inputs["input_ids"][bs:]
#         rewards = model(**inputs)
#         chosen_rewards = rewards[:bs]
#         rejected_rewards = rewards[bs:]
#         # compute pairwise loss. Only backprop on last value before padding
#         loss = 0
#         for i in range(bs):
#             # Retrieve first index where trajectories diverge
#             print((chosen[i] != rejected[i]).nonzero())
#             divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
#             assert divergence_ind > 0
#             # Check if there is any padding otherwise take length of sequence
#             c_inds = (chosen[i] == PAD_ID).nonzero()
#             c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
#             r_inds = (rejected[i] == PAD_ID).nonzero()
#             r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
#             end_ind = max(c_ind, r_ind)
#             # Index into correct reward
#             c_truncated_reward = chosen_rewards[i][divergence_ind : end_ind]
#             r_truncated_reward = rejected_rewards[i][divergence_ind : end_ind]
#             loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
#         loss = loss / bs
#         return (loss, outputs) if return_outputs else loss


dataset_name = "openai_comparison_summary"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

training_args = TrainingArguments(
    output_dir=f'ckpts/{dataset_name}/gpt-j', 
    num_train_epochs=5, 
    logging_steps=10,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    evaluation_strategy="steps",
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1, 
    eval_accumulation_steps=1,
    eval_steps=100,
    save_steps=1700,
    warmup_steps=100,
    logging_dir='./logs', 
    fp16=True, bf16=False, 
    learning_rate=1e-5, 
    deepspeed='./ds_config_gpt_j.json', 
    save_total_limit=1
)
# gptneo trained in jaxh

model = GPTRewardModel("/fsx/home-duyphung/refactor_summarize_rlhf/trlx/examples/summarize_rlhf/gptneo-supervised-summarize-checkpoint/checkpoint-1000")
layers = model.transformer.h
num_layers = len(layers)
num_unfrozen = int(0.3 * num_layers)
for layer in layers[:-num_unfrozen]:
    layer.requires_grad_(False)
# load_checkpoint = False
# if load_checkpoint:
#     model.load_state_dict(torch.load('ckpts/single_context_pairwise/model_fp16.pt'))
# #model.cuda()


max_length = 550
#max_length = max([max(len(tokenizer.encode(text["chosen"])), len(tokenizer.encode(text["rejected"]))) for text in data])
print("Max length: {}".format(max_length))


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer('<|startoftext|>' + chosen + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            rejected_encodings_dict = tokenizer('<|startoftext|>' + rejected + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length", return_tensors="pt")
            self.chosen_input_ids.append(chosen_encodings_dict['input_ids'])
            self.chosen_attn_masks.append(chosen_encodings_dict['attention_mask'])
            self.rejected_input_ids.append(rejected_encodings_dict['input_ids'])
            self.rejected_attn_masks.append(rejected_encodings_dict['attention_mask'])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return self.chosen_input_ids[idx], self.chosen_attn_masks[idx], self.rejected_input_ids[idx], self.rejected_attn_masks[idx]


class DataCollatorReward:
    
    def __call__(self, data):
        batch = {}
        batch['input_ids'] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch['attention_mask'] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch['labels'] = torch.tensor([0]*len(data) + [1] * len(data))
        return batch


# def data_collator(data):
#     return {'input_ids': torch.cat([f[0] for f in data] + [f[2] for f in data]),
#             'attention_mask': torch.cat([f[1] for f in data] + [f[3] for f in data]),
#             'labels': torch.tensor([0]*len(data) + [1] * len(data))}
    
def compute_metrics(eval_preds):
    chosen_mean_scores = eval_preds.predictions[1]#['chosen_mean_scores']
    rejected_mean_scores = eval_preds.predictions[2]#['rejected_mean_scores']
    chosen_end_scores = eval_preds.predictions[3]#['chosen_end_scores']
    rejected_end_scores = eval_preds.predictions[4]#['rejected_end_scores']
    
    result = {}
    acc = sum(chosen_mean_scores > rejected_mean_scores) / len(rejected_mean_scores)
    result['acc_by_mean'] = acc
    print("Accuracy by mean: {}".format(acc))
    
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result['acc_by_end'] = acc
    
    return result

train_pairs = create_comparision_dataset(os.path.join("/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons", "train_comparisons.jsonl"))
train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
val_pairs = create_comparision_dataset(os.path.join("/fsx/home-duyphung/refactor_summarize_rlhf/openai_data/comparisons", "valid_comparisons.jsonl"))
val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

data_collator = DataCollatorReward()

Trainer(model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset, 
        data_collator=data_collator).train()