"""
python result.py
"""
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "klue/bert-base"
ckpt_name = "model_save/klue-bert-base-2-NSMC-log_test.pt"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
eval_data = pd.read_csv("data/eda_data.txt", delimiter="\t")
eval_data = eval_data.dropna(axis=0)
eval_data = eval_data[:100]
eval_text = (
    eval_data["document"]
)

# dataset = [
#     {"data": t}
#     for t in eval_text
# ]

submission = []

for data in tqdm(eval_text):
    # print(data)
    text = data
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classification_results = output.logits.argmax(-1)
    
    submission.append([text] + [classification_results.item()])

# print(submission)

eda_data = pd.DataFrame(submission, columns=['document', 'label'])
eda_data.to_csv('data/result.txt', index=False, sep = '\t')

base_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")

for idx, i in enumerate(range(base_data['label'])):
    
    print(idx, i)

    # if 


    # if classification_results == label:
    #     acc += 1