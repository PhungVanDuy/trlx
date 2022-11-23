from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from datasets import load_metric
import evaluate
from tqdm import tqdm
import torch
import pandas as pd
from reward_model import GPT2LMHeadRewardModel


def load_model(path='/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize', ppo=False):
    tokenizer = GPT2Tokenizer.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-xl')
    gpt2model = GPT2LMHeadModel.from_pretrained(path)
    if ppo == True:
        gpt2model.load_state_dict(torch.load("/fsx/home-duyphung/trlx/supervised_models/gpt2-supervised-summarize/ppo_model.bin"))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return gpt2model, tokenizer


rw_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
rw_model = GPT2LMHeadRewardModel.from_pretrained('/fsx/home-duyphung/trlx/supervised_models/gpt2-reward-model-summarize/checkpoint-2000')
rw_model.resize_token_embeddings(len(rw_tokenizer))
rw_model.config.pad_token_id = rw_tokenizer.pad_token_id
rw_model.config.pad_token_id = rw_tokenizer.bos_token_id
rw_tokenizer.pad_token_id = rw_tokenizer.bos_token_id
rw_model.eval()
rw_device = torch.device("cuda:{}".format(1))
rw_model.to(rw_device)

def reward_fn(samples):
    
    encodings_dict = rw_tokenizer(
            samples, 
            truncation=True, 
            max_length=550, 
            padding="max_length"
    )
    input_ids = torch.tensor(encodings_dict['input_ids']).to(rw_device)
    attn_masks = torch.tensor(encodings_dict['attention_mask']).to(rw_device)
    # duplicate ids and masks
    input_ids = input_ids.repeat(2, 1)
    attn_masks = attn_masks.repeat(2, 1)
    with torch.no_grad():
        scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
    scores = scores.logits[:, 0] #- scores_ref.logits[:, 0] # normalize by truth score
    norms_scores = scores#normalize(scores.logits[:, 0],  scores_ref.logits[:, 0])#torch.nn.functional.normalize(scores, dim=0) #- scores_ref.logits[:, 0].mean()
    return norms_scores

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()
    post_list, summarize_list = get_dataset_from_jsonl("/fsx/home-duyphung/trlx/openai_data/tldr_filtered/valid.jsonl", return_summary=False)
    print(len(post_list))
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
            #temperature=0.7,
            #top_k=0,
            #top_p=1,
            #temperature=0.01,
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
        if count  == 100:
            break
    df = pd.DataFrame.from_dict({"pred": lst_pred, "truth": lst_summarize, "post": lst_post})
    # df.to_csv("ppo_out_sp.csv", index=False)
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)
    return df

if __name__=="__main__":
    ppo = True
    model, tokenizer = load_model(ppo=ppo)
    df_sup = inference(model, tokenizer)
    ppo_pred = df_sup['pred'].values#.tolist()
    
    # print("===========================================")
    # print("Is ppo: ", ppo)
    # print("pred: ", sup_pred)
    # print("===========================================")
    ppo = False
    model, tokenizer = load_model(ppo=ppo)
    df_sup = inference(model, tokenizer)
    sup_pred = df_sup['pred'].values#.tolist()
    truth = df_sup['truth'].values#.tolist()
    
    for (ppo, sup, tr) in zip(ppo_pred, sup_pred, truth):
        print("===========================================")
        print("ppo: ", ppo, "\n")
        print("pred: ", sup, "\n")
        print("truth: ", tr, "\n")
        print("===========================================\n")
    
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
    df = pd.DataFrame.from_dict({"supervised_pred":lst_data, "score": scores, "score_truth": scores2})
    
    df.to_csv("supervised_with_reward_scores.csv", index=False)
    print(df.score.values.mean())
    print(df.score_truth.values.mean())
    
    
    # import pickle
    # df_sup.to_csv("sup_out.csv", index=False)
    # pickle.dump(sup_pred, open("sup_pred.pkl", "wb"))
    # model, tokenizer = load_model(ppo=True)
    # df_ppo = inference(model, tokenizer)
    # ppo_pred = df_ppo['pred'].values.tolist()
    # df = pd.DataFrame.from_dict({"post": df_sup['post'].values.tolist(), 
    #                              "sup_pred": sup_pred, "ppo_pred": ppo_pred,
    #                              "truth": df_sup['truth'].values.tolist()})
    # df.to_csv("output_ppo_sup.csv", index=False)
    

