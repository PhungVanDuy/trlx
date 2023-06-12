import torch
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from trlx.models.modeling_ilql import AutoModelForCausalLMWithILQLHeads

from_fn = AutoModelForCausalLMWithILQLHeads.from_pretrained
import pandas as pd

model_paths = [
    "/mnt/hdd/duyphung/ilql_summarize_125M/checkpoint_6000/pytorch_model/mp_rank_00_model_states.pt",
    "/mnt/hdd/duyphung/ilql_summarize_1B/checkpoint_5000/pytorch_model/mp_rank_00_model_states.pt",
    "/mnt/hdd/duyphung/ilql_summarize_6B/checkpoint_3000/pytorch_model/mp_rank_00_model_states.pt",
]

model_based = ["EleutherAI/pythia-125m-deduped", "EleutherAI/pythia-1.3b-deduped", "EleutherAI/pythia-6.7b-deduped"]

ilql_names = ["ILQL_125M", "ILQL_1B", "ILQL_6B"]

alphas = [0.0001, 0.0001, 0.01]


df = load_dataset("Dahoas/openai_summarize_tldr_human_eval")["train"].to_pandas()

output_dict = {"prompt": df["prompt"]}

for model_path, model_name, alpha, ilql_name in zip(model_paths, model_based, alphas, ilql_names):
    print("Inference for ", ilql_name)
    model = from_fn(model_name, two_qs=True, alpha=alpha)
    print(model_path)
    temp = torch.load(model_path)["module"]
    new_state_dict = {}
    for key in temp.keys():
        if "ilql" in key:
            new_state_dict[key] = temp[key]
        else:
            new_state_dict[f"base_model.{key}"] = temp[key]
    model.load_state_dict(new_state_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    kwargs = {
        "max_new_tokens": 96,
        "top_k": 20,
        "beta": 1,
        "temperature": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    output = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row["prompt"].strip()
        max_prompt_length = 2048
        tokenized = tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=False, max_length=max_prompt_length
        )
        summary_ids = model.generate(tokenized["input_ids"], attention_mask=tokenized["attention_mask"], **kwargs)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)[len(input_text) :].strip()
        output.append(summary)
    output_dict[ilql_name] = output
df = pd.DataFrame(output_dict)
from datasets import Dataset

ds = Dataset.from_pandas(df).push_to_hub("pvduy/openai_summarize_tldr_human_eval_ilql_result")
