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



base_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
base_data = base_data[['document','label']]
eval_data = pd.read_csv("data/eda_data.txt", delimiter="\t")

all_df = pd.concat([base_data, eval_data], axis=1)      ####
all_df.columns = ['text', 'label', 'a_text', 'a_label']

all_df = all_df.dropna(axis=0)
all_df = all_df.reset_index(drop=True)
# result_data = pd.read_csv('data/result.txt', delimiter = "\t")       

# print(all_df)

# print(result_data)
# print(result_data.isnull().sum())
# print(len(result_data))


model.eval()
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()
# eval_data = eval_data.dropna(axis=0)
all_df = all_df[:120000]
eval_text = (
    all_df["a_text"]
)

dataset = [
    {"data": tokenizer.cls_token + t}
    for t in eval_text
]

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

result_data = pd.DataFrame(submission, columns=['document', 'label'])
result_data.to_csv('data/result.txt', index=False, sep = '\t')             

base_data = all_df[['text','label']]

target_df = pd.concat([base_data, result_data], axis=1)      ####
target_df.columns = ['text', 'label', 'a_text', 'a_label']

attack_df = target_df[(target_df['label'] != target_df['a_label'])]

print(f"acc: {((len(target_df) - len(attack_df)) / len(target_df))*100}%")

attack_df.to_csv('data/attack_result', index=False, sep="\t")