from typing import List

from transformers import AutoTokenizer
import trlx
from trlx.data.configs import TRLConfig
import evaluate

rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')
meteor = evaluate.load('meteor')

if __name__ == "__main__":
    
    def reward_fn(samples: List[str]):
        
        articles = [ 
            sample.split("<sep>")[0].strip() for sample in samples
        ]
        summs = [
            sample.split("<sep>")[1].strip() for sample in samples
        ]
        
        labels = [
            prompt_label[sample] for sample in articles
        ]
    
        scores = [
            meteor.compute(predictions=[summary], references=[label])
            for (summary, label) in zip(summs, labels)
        ]
        # scores = [
        #    (score['rouge1'] + score['rouge2'] + score['rougeL']) / 3. 
        #         for score in scores
        # ]
        scores = [score['meteor'] for score in scores]
        return scores

# RL4LMs: {'do_sample': True, 'top_k': 0, 'temperature': 0.7, 'min_length': 50, 'max_new_tokens': 100}
# 5120 rollouts
# attention mask - encoder outputs - decoder inputs ids - decoder attention mask
# rollouts: {'do_sample': True, 'top_k': 50, 'min_length': 50, 'max_new_tokens': 100}

    config = TRLConfig.load_yaml("ppo_config_cnn_daily.yml")
    print("load dataset")
    from datasets import load_dataset
    dataset = load_dataset("cnn_dailymail", '3.0.0', split="train", cache_dir="data")
    prompts = dataset["article"][0:10000]#[dataset["article"][0]] * 1000
    summaries = dataset["highlights"][0:10000]#[dataset["highlights"][0]] * 1000
    prompts = ["Summarize: " + prompt for prompt in prompts]
    val_dataset = load_dataset("cnn_dailymail", '3.0.0', split="validation", cache_dir="data")
    val_prompts = ["Summarize: " + prompt for prompt in val_dataset['article'][0:1000]]
    val_summaries = val_dataset['highlights'][0:1000]
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if 0:
        prompt_label = {}
        max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
        from tqdm import tqdm
        for i in tqdm(range(len(prompts))):
            key = tokenizer.decode(
                tokenizer(
                    prompts[i],
                    truncation=True,
                    max_length=max_length
                )['input_ids'],
                skip_special_tokens=True, 
            )
            prompt_label[key.strip()] = summaries[i]
        
        for i in tqdm(range(len(val_prompts))):
            key = tokenizer.decode(
                tokenizer(
                    val_prompts[i],
                    truncation=True,
                    max_length=max_length
                )['input_ids'],
                skip_special_tokens=True, 
            )
            prompt_label[key.strip()] = val_summaries[i]
    
        import pickle
        pickle.dump(prompt_label, open("prompt_label.pkl", "wb"))
    else:
        import pickle
        prompt_label = pickle.load(open("prompt_label.pkl", "rb"))
    print("running")
    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts[0:1000],
        config=config
    )
