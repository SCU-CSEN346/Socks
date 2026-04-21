CUDA_DEVICE = "cuda:1"

import json
PARAMS = json.load(open("../params.json", "r"))

NLLFG_NAME = PARAMS["NLLFG_NAME"]
USERNAME = PARAMS["USERNAME"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
MAXLEN_Q = PARAMS["MAXLEN_Q"]
MAXLEN_A = PARAMS["MAXLEN_A"]
MAXLEN_B = PARAMS["MAXLEN_B"]

from transformers import BertTokenizer, BertModel
import torch

print(torch.cuda.is_available())

torch.cuda.set_device(CUDA_DEVICE)

import gc
torch.cuda.empty_cache()
gc.collect()

from huggingface_hub import hf_hub_url, cached_download

repo_name = NLLFG_NAME
config_file_url = hf_hub_url(f"{USERNAME}/"+repo_name, filename="cls_layer.torch")
value = cached_download(config_file_url)
cls_layer = torch.load(value).cuda()

the_model = BertModel.from_pretrained(f"{USERNAME}/"+repo_name).cuda()
the_tokenizer = BertTokenizer.from_pretrained(f"{USERNAME}/"+repo_name, do_lower_case=False)
e = the_model.eval()

from torch.utils.data import Dataset
class DatasetTaskDecision(Dataset):
    def __init__(self, df, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, maxlen_bsq=MAXLEN_B):
        self.df = df
        self.tokenizer = the_tokenizer
        self.maxlen_Q = maxlen_Q
        self.maxlen_A = maxlen_A
        self.maxlen_bsq = maxlen_bsq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = int(self.df.loc[index, "label"])
        q, a = self.df.loc[index, 'Q'], self.df.loc[index, "A"]

        sentence1, sentence2 = str(q), str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence1, sentence2 = sentence1.strip(), sentence2.strip()
        
        sentence3 = str(self.df.loc[index, "bsq"])

        tokens1 = self.tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
        tokens3 = self.tokenizer.tokenize(sentence3) if len(sentence3)>0 else ["[UNK]"]
        
        if len(tokens1) <= self.maxlen_Q:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_Q - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_Q]
                    
        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]

        if len(tokens3) <= self.maxlen_bsq:
            tokens3 = tokens3 + ['[PAD]' for _ in range(self.maxlen_bsq - len(tokens3))]
        else:
            tokens3 = tokens3[:self.maxlen_bsq]
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]+tokens3+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1
        
        return tokens_ids_tensor, attn_mask, label
    

import torch.nn as nn
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

import pandas as pd

df = pd.read_csv("dataset_nllfg.csv", index_col=0)

df_train = df[df["set"] == "train"]
df_val = df[df["set"] == "val"]

df_train = df_train.reset_index()
df_val = df_val.reset_index()

from torch.utils.data import DataLoader

train_set = DatasetTaskDecision(df = df_train)
val_set = DatasetTaskDecision(df = df_val)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)

import numpy as np

def tests_preds(net, dataloader):
    net.eval()
    preds = []
    tests = []
    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            logits = net(seq, attn_masks)
            probs = torch.sigmoid(logits)
            soft_probs = probs.argmax(1)
            preds += soft_probs.squeeze().tolist()
            tests += labels.tolist()
            
    return np.array(tests), np.array(preds)

from sklearn.metrics import classification_report

for k, datatloader in {"val": val_loader, "train": train_loader}.items():
    tests, preds = tests_preds(net, datatloader)
    y_true = tests
    y_pred = preds
    print("Set:", k)
    print(classification_report(y_true, y_pred, target_names=["No", "Yes"]))
    print("\n")