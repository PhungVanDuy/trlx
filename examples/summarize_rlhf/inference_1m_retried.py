from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Jellywibble/12m-retry-continue-combined-regressor-epoch-1")
model = AutoModelForSequenceClassification.from_pretrained("Jellywibble/12m-retry-continue-combined-regressor-epoch-1")
tokenizer.pad_token_id = 50256
tokenizer.truncation_side = "left"
tokenizer.padding_side = "right"

import pandas as pd

df = pd.read_parquet("1M_train_chai.parquet").sample(n=100000)

batch_size = 64
total_rewards = []
for i in tqdm(range(0, len(df), batch_size)):
    candidates = [x["prompt_input"] + "\n" + x["prompt_output"] for (i, x) in df.iloc[i : i + batch_size].iterrows()]
    tokens = tokenizer(
        candidates, return_tensors="pt", return_attention_mask=True, padding="longest", truncation=True, max_length=256
    )
    reward = model(**tokens).logits.detach().cpu().numpy().tolist()
    total_rewards.extend(reward)

df["retry_reward"] = total_rewards
df.to_parquet("100K_train_chai_retry.parquet")
