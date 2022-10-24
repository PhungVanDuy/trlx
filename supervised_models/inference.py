from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from datasets import load_metric
import evaluate
from tqdm import tqdm

def load_model(path='gpt2-sup-summ-ver2/checkpoint-10000/'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    gpt2model = GPT2LMHeadModel.from_pretrained(path)
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return gpt2model, tokenizer

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()
    post_list, summarize_list = get_dataset_from_jsonl("../openai_data/tldr_filtered/valid.jsonl", return_summary=False)
    lst_pred = []
    lst_summarize = []
    rouge = evaluate.load('rouge')
    count = 0
    for post, summarize in tqdm(zip(post_list, summarize_list), total=len(post_list)):
        encode_dict = tokenizer(post, return_tensors="pt", padding=False, truncation=True, max_length=512)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        summ_tokens = model.generate(txt_tokens,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=5,
            max_length=512
        )
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("TL;DR:")[1].replace("<|endoftext|>", "")
        lst_pred.append(pred)
        lst_summarize.append(summarize)
        if count % 10 == 0:
            result = rouge.compute(predictions=lst_pred, references=lst_summarize)
            print(result)
        count += 1
            
    result = rouge.compute(predictions=lst_pred, references=lst_summarize)
    print(result)

if __name__=="__main__":
    model, tokenizer = load_model()
    inference(model, tokenizer)

