import pickle

from datasets import load_dataset
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

model = AutoModelForCausalLM.from_pretrained("pvduy/llama_30b")
model.half().cuda()
tokenizer = AutoTokenizer.from_pretrained("llama_30b_tokenizer")
tokenizer.pad_token_id = tokenizer.eos_token_id
human_eval = load_dataset("openai_humaneval", split="test")
human_eval_prompts = ["<human>: Generate code follow this instruction: \n" + x + "\n<bot>:"  for x in human_eval['prompt']]
batch_size = 2
lst_output = []
for i in tqdm(range(0, len(human_eval_prompts), batch_size)):
    texts = human_eval_prompts[i:i+batch_size]
    token_dict = tokenizer(texts, return_tensors="pt", padding="longest")
    for x in token_dict:
        token_dict[x] = token_dict[x].cuda()
    outputs = model.generate(inputs=token_dict["input_ids"], attention_mask=token_dict['attention_mask'], max_new_tokens=128, do_sample=True, top_p=1, top_k=0, temperature=1)
    lst_output.extend([tokenizer.decode(x, skip_special_tokens=True) for x in outputs])
    if i % 100 == 0:
        with open("llama_30b_human_eval.pkl", "wb") as f:
            pickle.dump(lst_output, f)

import pandas as pd

df = human_eval.to_pandas()
df['output_pred'] = lst_output
df.to_csv('llama_30b_human_eval.csv', index=False)
