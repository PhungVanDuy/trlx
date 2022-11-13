# Summarize from Human Feedback
Using trlx to train summarization models from OpenAI Dataset

## Prepare dataset

Download OpenAI Dataset following this [link](https://drive.google.com/file/d/1SYGPeyPlqsQYF-OMSOnR3ZvkV_lQXUQu/view?usp=share_link)


## Train supervised learning model

```bash
python train_gpt2_summarize.py
```

Current result on dev set:


## Train reward model
```bash
python train_reward_model.py
# python test_reward_model.py # to test reward model accuracy
```

Accuracy on dev set comparision: ~65%


