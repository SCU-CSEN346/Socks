import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
PARAMS = json.load(open("../params.json", "r"))

NLLFG_NAME = PARAMS["NLLFG_NAME"]
USERNAME = PARAMS["USERNAME"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
MAXLEN_A = PARAMS["MAXLEN_A"]
MAXLEN_B = PARAMS["MAXLEN_B"]

from transformers import BertTokenizer, BertModel
import torch
import gc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from huggingface_hub import login as hf_login, hf_hub_download
from sklearn.metrics import classification_report

CUDA_DEVICE = "cuda:0"

print(torch.cuda.is_available())
torch.cuda.set_device(CUDA_DEVICE)
torch.cuda.empty_cache()
gc.collect()

repo_name = NLLFG_NAME
value = hf_hub_download(repo_id=f"{USERNAME}/{repo_name}", filename="cls_layer.torch")
cls_layer = torch.load(value, weights_only=False).cuda()

the_model = BertModel.from_pretrained(f"{USERNAME}/{repo_name}").cuda()
the_tokenizer = BertTokenizer.from_pretrained(f"{USERNAME}/{repo_name}", do_lower_case=False)
e = the_model.eval()

class DatasetTaskDecision(Dataset):
    def __init__(self, df, maxlen_A=MAXLEN_A, maxlen_bsq=MAXLEN_B):
        self.df = df
        self.tokenizer = the_tokenizer
        self.maxlen_A = maxlen_A
        self.maxlen_bsq = maxlen_bsq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = int(self.df.loc[index, "label"])
        a = self.df.loc[index, "EssayText"]

        sentence2 = str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence2 = sentence2.strip()

        sentence3 = str(self.df.loc[index, "bsq"])

        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
        tokens3 = self.tokenizer.tokenize(sentence3) if len(sentence3)>0 else ["[UNK]"]

        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]

        if len(tokens3) <= self.maxlen_bsq:
            tokens3 = tokens3 + ['[PAD]' for _ in range(self.maxlen_bsq - len(tokens3))]
        else:
            tokens3 = tokens3[:self.maxlen_bsq]

        tokens = ["[CLS]"]+tokens2+["[SEP]"]+tokens3+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long()

        return tokens_ids_tensor, attn_mask, label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.bert_layer = the_model
        self.cls_layer = cls_layer

    def forward(self, seq, attn_masks):
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        cls_rep = cont_reps.last_hidden_state[:, 0]
        logits = self.cls_layer(cls_rep)
        return logits

net = Classifier()

df = pd.read_csv("dataset_nllfg.csv", index_col=0)

df_train = df[df["set"] == "train"].reset_index()
df_val = df[df["set"] == "val"].reset_index()

train_set = DatasetTaskDecision(df=df_train)
val_set = DatasetTaskDecision(df=df_val)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

def tests_preds(net, dataloader):
    net.eval()
    preds = []
    tests = []
    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            logits = net(seq, attn_masks)
            soft_probs = logits.argmax(1)
            preds += soft_probs.squeeze().tolist()
            tests += labels.tolist()
    return np.array(tests), np.array(preds)

for k, dataloader in {"val": val_loader, "train": train_loader}.items():
    tests, preds = tests_preds(net, dataloader)
    print("Set:", k)
    print(classification_report(tests, preds, target_names=["No", "Yes"]))
    print("\n")
