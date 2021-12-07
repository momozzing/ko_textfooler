"""
python interactive.py
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "klue/bert-base"
ckpt_name = "model_save/klue-bert-base-2-NSMC-log_test.pt"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

with torch.no_grad():
    while True:
        t = input("\ntext: ")
        tokens = tokenizer(
            t,
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
        print(f"Result: {'1' if classification_results.item() else '0'}")
