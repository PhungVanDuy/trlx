import torch
from trlx.models.modeling_ilql import AutoModelForCausalLMWithILQLHeads
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from_fn = AutoModelForCausalLMWithILQLHeads.from_pretrained
import pandas as pd
import evaluate
import torch
import pickle
rouge_scorer = evaluate.load('rouge')

model_paths = [
    "/mnt/hdd/duyphung/ilql_summarize_125M/checkpoint_6000/pytorch_model/mp_rank_00_model_states.pt",
    "/mnt/hdd/duyphung/ilql_summarize_1B/checkpoint_5000/pytorch_model/mp_rank_00_model_states.pt",
    "/mnt/hdd/duyphung/ilql_summarize_6B/checkpoint_3000/pytorch_model/mp_rank_00_model_states.pt"
]

model_based = [
    "EleutherAI/pythia-125m-deduped",
    "EleutherAI/pythia-1.3b-deduped",
    "EleutherAI/pythia-6.7b-deduped"
]

ilql_names = [
    "ILQL_125M",
    "ILQL_1B",
    "ILQL_6B"
]

alphas = [0.0001, 0.0001, 0.01]


dataset = load_dataset("CarperAI/openai_summarize_tldr")['test']

df = dataset.to_pandas()
output_dict = {}

for (model_path, model_name, alpha, ilql_name) in zip(model_paths, model_based, alphas, ilql_names):
    print("Inference for ", ilql_name)
    model = from_fn(model_name, two_qs=True, alpha=alpha)
    print(model_path)
    temp = torch.load(model_path)['module']
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
        "temperature": 0.9,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    batch_size = 4
    model.cuda()
    model.eval()
    output = []
    output_dict[model_path] = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size)):
            input_text = df['prompt'][i:i+batch_size].tolist()
            input_text = [x.strip().strip("\n") for x in input_text]
            max_prompt_length = 2048
            tokenized = tokenizer(input_text, return_tensors='pt', padding=True)
            for x in tokenized:
                tokenized[x] = tokenized[x].cuda()
            summary_ids = model.generate(tokenized["input_ids"], attention_mask=tokenized["attention_mask"],  **kwargs)
            summary = [tokenizer.decode(ids, skip_special_tokens=True).split("TL;DR:")[1].strip() for ids in summary_ids]
            output.extend(summary)
    output_dict[model_path]['output'] = output
    metrics = rouge_scorer.compute(predictions=output, references=df['label'].tolist())
    output_dict[model_path]['metrics'] = metrics
    print(metrics)
    pickle.dump(output_dict, open("ilql_rlhf_output_rouge.pkl", "wb"))
