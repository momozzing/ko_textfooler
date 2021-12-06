'''
deepspeed --num_gpus=1 BoolQ_fine_tune.py
'''

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb

#############################################    -> 실험결과 FIX
random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
##################################

task = "NSMC"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# SPECIAL_TOKENS = {
#     "bos_token": "<bos>",
#     "eos_token": "<eos>",
#     "pad_token": "<pad>",
#     "sep_token": "<seq>"
#     }
# SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<seq>"]
# tokenizer.add_special_tokens(SPECIAL_TOKENS)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
).cuda()

# model.resize_token_embeddings(len(tokenizer)) 

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--batch_size", default=128, type=int)
# parser.add_argument("--sep_token", default=tokenizer.sep_token, type=str)
args = parser.parse_args()

wandb.init(project="ko_textfooler", name=f"{model_name}-{task}")
train_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
train_data = train_data.dropna(axis=0)
train_data = train_data[:120000]
train_text, train_labels = (
    train_data["document"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(train_text, train_labels)
]
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
eval_data = eval_data.dropna(axis=0)
eval_data = eval_data[120000:]
eval_text, eval_labels = (
    eval_data["document"].values,
    eval_data["label"].values
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]
eval_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = DeepSpeedCPUAdam(
    lr=3e-5, weight_decay=3e-7, model_params=model.parameters()
)

engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, optimizer=optimizer
)
epochs = 0
# step = 0
for epoch in range(args.epoch):
    epochs += 1
    model.train()
    for train in tqdm(train_loader):
        model.train()
        optimizer.zero_grad()
        text, label = train["data"], train["label"].cuda()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # is_split_into_words=True
            # max_length=100
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )
        loss = output.loss
        engine.backward(loss)
        optimizer.step()
        classification_results = output.logits.argmax(-1)
        # print(classification_results.size(), label.size())   ### size 동일 
        # print(output.logits)
        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1
                
        wandb.log({"loss": loss})
        wandb.log({"acc": acc / len(classification_results)})
    model.eval()
    tmp=[]
    for eval in tqdm(eval_loader):
        eval_text, eval_label = eval["data"], eval["label"].cuda()
        eval_tokens = tokenizer(
            eval_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # is_split_into_words=True
            # max_length=100
        )
        input_ids = eval_tokens.input_ids.cuda()
        attention_mask = eval_tokens.attention_mask.cuda()
        
        eval_out = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=eval_label
        )
            
        eval_classification_results = eval_out.logits.argmax(-1)
        eval_loss = eval_out.loss

        eval_acc = 0
        for res, lab in zip(eval_classification_results, eval_label):
            if res == lab:
                eval_acc += 1
        
        wandb.log({"eval_loss": eval_loss})   ## 이미 다 적용된 상태인듯..
        wandb.log({"eval_acc": eval_acc / len(eval_classification_results)})
        wandb.log({"epoch": epochs})
        torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{epochs}-{task}.pt")
        # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-{random_seed}-mono_post.pt")



    # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{step}-{epoch}-{random_seed}-v12.pt")
    # torch.save(model.state_dict(), f"model_save/{model_name.replace('/', '-')}-{task}-{epoch}-30-64-{random_seed}-base.pt")
