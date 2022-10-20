import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

def get_dataset_from_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as f:
        dataset = [json.loads(line) for line in f]
    post_list = [d['post'] for d in dataset]
    summary_list = [d['summary'] for d in dataset]
    return post_list, summary_list

class TLDRDataset(Dataset):

  def __init__(self, train_path, tokenizer, max_length=768):

    self.post_list, self.summarize_list = get_dataset_from_jsonl(train_path)
    if 'valid' in train_path:
        self.post_list = self.post_list[:1000]
        self.summarize_list = self.summarize_list[:1000]
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.input_ids = []
    self.attn_masks = []
    self.text = [post + " <|tl;dr|> " for post in self.post_list]
    
  def __len__(self):
    return len(self.post_list)

  def __getitem__(self, idx):
    txt = self.post_list[idx] + " <|tl;dr|> " + self.summarize_list[idx]
    encodings_dict = self.tokenizer(
        txt, truncation=True, max_length=self.max_length, padding="max_length"
    )
    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_masks = torch.tensor(encodings_dict['attention_mask'])
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attn_masks,
        "labels": input_ids
    }
    
class TLDRPPODataset(Dataset):

  def __init__(self, train_path, tokenizer, max_length=768):

    self.post_list, self.summarize_list = get_dataset_from_jsonl(train_path)
    if 'valid' in train_path:
        post_list = post_list[:100]
        summarize_list = summarize_list[:100]
    self.tokenizer = tokenizer
    self.max_length = max_length

    
    
  def __len__(self):
    return len(self.post_list)

  def __getitem__(self, idx):
    txt = self.post_list[idx] + " <|tl;dr|> "
    txt = '<|startoftext|> ' + txt
    encodings_dict = self.tokenizer(
        txt, truncation=True, max_length=self.max_length, padding="max_length"
    )
    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_masks = torch.tensor(encodings_dict['attention_mask'])
    
    return {
        "input_ids": input_ids,
        "attention_masks": attn_masks,
        "text": txt
    }


class ComparisionDataset(Dataset):

    def __init__(self, comparision_path, tokenizer, max_length=532):
        with open(comparision_path, 'r') as f:
            dataset = [json.loads(line) for line in f]
        if 'valid' in comparision_path:
            dataset = dataset[0:5000]
        self.tokenizer = tokenizer
        self.lst_post = []
        self.lst_summaries_0 = []
        self.lst_summaries_1 = []
        self.labels = []
        self.max_length = max_length

        def make_text(text, summarize):
            return '<|startoftext|>' + text + ' <|tl;dr|> ' + summarize + '<|endoftext|>'

        for sample in dataset:
            self.lst_post.append(sample['info']['post'])
            if sample['choice'] == 0:
                self.lst_summaries_0.append(make_text(sample['info']['post'], sample['summaries'][0]['text']))
                self.lst_summaries_1.append(make_text(sample['info']['post'], sample['summaries'][1]['text']))
            else:
                self.lst_summaries_0.append(make_text(sample['info']['post'], sample['summaries'][1]['text']))
                self.lst_summaries_1.append(make_text(sample['info']['post'], sample['summaries'][0]['text']))
            self.labels.append(0)        


    def __len__(self):
        return len(self.lst_post)
    
    def __getitem__(self, idx):
        summ0 = self.lst_summaries_0[idx]
        summ1 = self.lst_summaries_1[idx]
        choice = self.labels[idx]
        encodings_dict_0 = self.tokenizer(
            summ0, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length"
        )
        encodings_dict_1 = self.tokenizer(
            summ1, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length"
        )
        input_ids_0 = torch.tensor(encodings_dict_0['input_ids'])
        input_ids_1 = torch.tensor(encodings_dict_1['input_ids'])
        attn_masks_0 = torch.tensor(encodings_dict_0['attention_mask'])
        attn_masks_1 = torch.tensor(encodings_dict_1['attention_mask'])

        return {
            "input_ids_0": input_ids_0, 
            "attention_mask_0": attn_masks_0,
            "input_ids_1": input_ids_1, 
            "attention_mask_1": attn_masks_1,
            "labels": choice
        }
