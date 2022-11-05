from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from datasets import load_metric
import evaluate
from tqdm import tqdm
import torch
import pandas as pd


def load_model(path='/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize'):
    tokenizer = GPT2Tokenizer.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-xl')
    gpt2model = GPT2LMHeadModel.from_pretrained(path)
    gpt2model.load_state_dict(torch.load("/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize/ppo_model.bin"))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return gpt2model, tokenizer

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()
    post_list, summarize_list = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/valid.jsonl", return_summary=False)
    lst_pred = []
    lst_summarize = []
    lst_post = []
    rouge = evaluate.load('rouge')
    count = 0
    for post, summarize in tqdm(zip(post_list, summarize_list), total=len(post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True, max_length=512)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        summ_tokens = model.generate(txt_tokens,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=0,
            top_p=1,
            temperature=0.01,
            max_length=550
        )
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        lst_pred.append(pred)
        lst_summarize.append(summarize)
        lst_post.append(post)
        if count % 10 == 0:
            result = rouge.compute(predictions=lst_pred, references=lst_summarize)
            print(result)
        count += 1
        if count  == 1000:
            break
    df = pd.DataFrame.from_dict({"pred_ppo": lst_pred, "truth": lst_summarize, "post": lst_post})
    df.to_csv("ppo_out.csv", index=False)
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)
    print(df['pred_ppo'].values.tolist())
    print("=====================================")
    print(df['truth'].values.tolist())

if __name__=="__main__":
    model, tokenizer = load_model()
    inference(model, tokenizer)

