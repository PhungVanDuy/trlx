from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

flant5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

text = """
Summarize: (CNN)Share, and your gift will be multiplied. That may sound like an esoteric adage, but when Zully Broussard selflessly decided to give one of her kidneys to a stranger, her generosity paired up with big data
. It resulted in six patients receiving transplants. That surprised and wowed her. "I thought I was going to help this one person who I don\'t know, but the fact that so many people can have a life extension, that\'s prett
y big," Broussard told CNN affiliate KGO. She may feel guided in her generosity by a higher power. "Thanks for all the support and prayers," a comment on a Facebook page in her name read. "I know this entire journey is muc
h bigger than all of us. I also know I\'m just the messenger." CNN cannot verify the authenticity of the page. But the power that multiplied Broussard\'s gift was data processing of genetic profiles from donor-recipient pa
irs. It works on a simple swapping principle but takes it to a much higher level, according to California Pacific Medical Center in San Francisco. So high, that it is taking five surgeons, a covey of physician assistants,
nurses and anesthesiologists, and more than 40 support staff to perform surgeries on 12 people. They are extracting six kidneys from donors and implanting them into six recipients. "The ages of the donors and recipients ra
nge from 26 to 70 and include three parent and child pairs, one sibling pair and one brother and sister-in-law pair," the medical center said in a statement. The chain of surgeries is to be wrapped up Friday. In late March
, the medical center is planning to hold a reception for all 12 patients. Here\'s how the super swap works, according to California Pacific Medical Center. Say, your brother needs a kidney to save his life, or at least get
 off of dialysis, and you\'re willing to give him one of yours. But then it turns out that your kidney is not a match for him, and it\'s certain his body would reject it. Your brother can then get on a years-long waiting l
ist for a kidney coming from an organ donor who died. Maybe that will work out -- or not, and time
"""
flant5.eval()

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="data")
tokenizer.padding_side = "left"
tokenizer.truncation_side = "right"
tokenizer.sep_token = "<sep>"
val_prompts = ["Summarize: " + prompt for prompt in dataset["validation"]["article"][0:1000]]
val_summaries = dataset["validation"]["highlights"][0:1000]

encoded_input = tokenizer(val_prompts[0:5], truncation=True, max_length=512, return_tensors="pt", padding="max_length")


for x in encoded_input:
    encoded_input[x] = encoded_input[x].to("cuda")
print(encoded_input["input_ids"])
gen = flant5.generate(
    encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"], max_new_tokens=100, do_sample=True
)
print(tokenizer.batch_decode(gen, skip_special_tokens=True))
