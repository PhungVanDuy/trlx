from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from summarize_dataset import TLDRDataset, get_dataset_from_jsonl
from datasets import load_metric
from tqdm import tqdm

def load_model(path='../trlx/checkpoint_supervised_gpt_2'):
    tokenizer = GPT2Tokenizer.from_pretrained(path, 
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        pad_token='<|pad|>'
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ["<|tl;dr|>"]})
    gpt2model = GPT2LMHeadModel.from_pretrained(path)
    gpt2model.resize_token_embeddings(len(tokenizer))
    gpt2model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.bos_token_id
    return gpt2model, tokenizer

def inference(model, tokenizer):
    model.to("cuda")
    model.eval()
    post_list, summarize_list = get_dataset_from_jsonl("../openai_data/tldr_filtered/valid.jsonl")
    lst_pred = []
    lst_summarize = []
    count = 0
    for post, summarize in tqdm(zip(post_list, summarize_list), total=len(post_list)):
        txt = post + " <|tl;dr|> "
        txt = '<|startoftext|> ' + txt
        encode_dict = tokenizer(txt, return_tensors="pt", padding=False, truncation=True, max_length=768)
        txt_tokens = encode_dict["input_ids"]
        attention_mask = encode_dict["attention_mask"]
        txt_tokens = txt_tokens.to('cuda')
        attention_mask = attention_mask.to('cuda')
        import ipdb; ipdb.set_trace()
        summ_tokens = model.generate(txt_tokens,
            attention_mask=attention_mask,
            num_beams=5, 
            no_repeat_ngram_size=2,
            bos_token_id=tokenizer.bos_token_id,
            max_length=768
        )
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("<|tl;dr|>")[1].replace("<|endoftext|>", "")
        lst_pred.append(pred)
        lst_summarize.append(summarize)
        count += 1
        if count == 1000:
            break
    rouge = load_metric("rouge")
    rouge_output = rouge.compute(
            predictions=lst_pred, references=lst_summarize, rouge_types=["rouge2"]
    )["rouge2"].mid
    print(rouge_output)

if __name__=="__main__":
    model, tokenizer = load_model()
    
    inference(model, tokenizer)

