import os

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class GPTRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained("pvduy/vicuna-13b-v1.1")
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("pvduy/vicuna-13b-v1.1")
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
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
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False

        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            try:
                c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
                r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
                # Append the last rewards to the list of end scores
                chosen_end_scores.append(c_truncated_reward[-1])
                rejected_end_scores.append(r_truncated_reward[-1])
            except:
                print("Error")
                print(self.tokenizer.batch_decode(input_ids))
                continue

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }


def create_comparison_dataset(path="pvduy/hh_shp_oa_gpt4_rm_dataset", split="train"):
    dataset = load_dataset(path, split=split)
    if split == "test":
        dataset = dataset.select(range(5000))
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        prompt = prompt.replace("</s>", "")
        chosen_summary = sample["chosen"] + "</s>"
        rejected_summary = sample["rejected"] + "</s>"
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        if len(chosen_summary.split()) > 2000 or len(rejected_summary.split()) > 2000:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                continue
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        return batch

def convert_rm_data_to_vicuna(df):
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("Human: ", "USER: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Response: Assistant:", "\nASSISTANT:"))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Input:\n", "\nInput: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Instruction:\n", "\nInstruction: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\nUSER: ", "</s>\nUSER: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\nAssistant: ", "\nASSISTANT: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\nAssistant:", "\nASSISTANT:"))
    return df

def convert_sft_data_to_vicuna(df):
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Response: ASSISTANT:", "\nASSISTANT:"))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Input:\n", "\nInput: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\n\n### Instruction:\n", "\nInstruction: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace(" ASSISTANT: ", "\nASSISTANT: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace(" ASSISTANT:", "\nASSISTANT:"))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\nInput: \nASSISTANT: ", "\nASSISTANT: "))
    df['prompt'] = df['prompt'].apply(lambda x: x.replace("\nInput: \nASSISTANT:", "\nASSISTANT:"))
    return df

def convert_vicuna_to_trlx(data):

    def convert_conv(conv):
        turns = conv['conversations']
        conv_str = ""
        label = ""
        for (i, turn) in enumerate(turns):
            if i == 0 and turn['from'] == "gpt":
                return None, None
            if i == len(turns) - 1:
                if turn['from'] == "gpt":
                    label = turn['value']
                    return conv_str, label
                else:
                    return None, None

            if turn['from'] == "human":
                conv_str += "USER: " + turn['value'] + "\n"
            else:
                conv_str += "ASSISTANT: " + turn['value'] + "</s>\n"
    lst_prompt = []
    lst_label = []
    for conv in data:
        prompt, label = convert_conv(conv)
        if prompt is None:
            continue
        lst_prompt.append(prompt)
        lst_label.append(label)
    df = pd.DataFrame({"prompt": lst_prompt, "label": lst_label})
    return df




def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("pvduy/vicuna-13b-v1.1")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    training_args = TrainingArguments(
        output_dir="/mnt/hdd/duyphung/rm_checkpoint_vicuna_ver1.1/",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=False,
        bf16=True,
        learning_rate=1e-5,
        deepspeed="ds_config_rm.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = GPTRewardModel()

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.layers
    num_layers = len(layers)
    num_unfrozen = int(0.5 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = "pvduy/hh_shp_oa_gpt4_rm_dataset_vicuna_format"
    train_pairs = create_comparison_dataset(data_path, "train")
    val_pairs = create_comparison_dataset(data_path, "test")

    # Make pairwise datasets for training
    max_length = 1024
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
