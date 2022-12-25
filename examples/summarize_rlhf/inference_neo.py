from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from datasets import load_metric
import evaluate
from tqdm import tqdm
import torch
import pandas as pd
from reward_model import GPT2LMHeadRewardModel
from reward_model_inspect.reward_model import GPTRewardModel

from transformers import AutoTokenizer, AutoModelForCausalLM



def load_model(path='gptneo-supervised-summarize-checkpoint/checkpoint-1000', ppo=False):
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    gpt2model = AutoModelForCausalLM.from_pretrained(path)
    if ppo == True:
        gpt2model.load_state_dict(torch.load("val_val_ckpts/gpt.bin"))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    return gpt2model, tokenizer


rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
rw_tokenizer.pad_token = rw_tokenizer.eos_token
rw_model = GPTRewardModel("gptneo-supervised-summarize-checkpoint/checkpoint-1000")
rw_model.load_state_dict(torch.load("reward_model_inspect/ckpts/openai_comparison_summary/gpt-j/checkpoint-1700/pytorch_model.bin"))
rw_model.half()
rw_model.eval()
rw_device = torch.device("cuda:{}".format(1))
rw_model.to(rw_device)


def reward_fn(samples):        
        lst_scores = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i:i+batch_size]
            sub_samples = ['<|startoftext|>' + chosen + '<|endoftext|>' for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                    sub_samples, 
                    truncation=True, 
                    max_length=550, 
                    padding="max_length",
                    return_tensors="pt"
            )
            input_ids = encodings_dict['input_ids'].to(rw_device)
            attn_masks = encodings_dict['attention_mask'].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            lst_scores.append(sub_scores['chosen_end_scores'])
        scores = torch.cat(lst_scores, dim=0)
        norms_scores = scores
        return norms_scores

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()
    post_list, summarize_list = get_dataset_from_jsonl("/admin/home-duyphung/refactor_summarize_rlhf/openai_data/tldr_filtered/test.jsonl", return_summary=False)
    lst_pred = []
    lst_summarize = []
    lst_post = []
    rouge = evaluate.load('rouge')
    count = 0
    for post, summarize in tqdm(zip(post_list, summarize_list), total=len(post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {'max_new_tokens': 50, 'eos_token_id': 50256, 'pad_token_id': 50256}
        summ_tokens = model.generate(
            txt_tokens,
            attention_mask=attention_mask,
            **kwargs
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
        if count  == 100:
            break
    df = pd.DataFrame.from_dict({"pred": lst_pred, "truth": lst_summarize, "post": lst_post})
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)
    return df

if __name__=="__main__":

    
    ppo = True
    model, tokenizer = load_model(ppo=ppo)
    df_sup = inference(model, tokenizer)
    sup_pred = df_sup['pred'].values
    truth = df_sup['truth'].values

    
    scores = []
    scores2 = []
    lst_data = []
    batch_size = 16
    for i in range(0, len(df_sup), batch_size):
        summ = df_sup['pred'].values[i:i+batch_size]
        summ2 = df_sup['truth'].values[i:i+batch_size]
        post = df_sup['post'].values[i:i+batch_size]
        all_data = [post[i] + summ[i] for i in range(len(summ))]
        all_data2 = [post[i] + summ2[i] for i in range(len(summ2))]
        lst_data.extend(all_data)
        scores.extend(list(reward_fn(all_data).cpu().numpy()))
        scores2.extend(list(reward_fn(all_data2).cpu().numpy()))
    df = pd.DataFrame.from_dict({"ppo_pred":lst_data, "score": scores, "score_truth": scores2})
    
    df.to_csv("supervised_with_reward_scores_neo_ppo.csv", index=False)
    print(df.score.values.mean())
    print(df.score_truth.values.mean())