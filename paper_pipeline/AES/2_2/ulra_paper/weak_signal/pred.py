CUDA_DEVICE = f"cuda:0"

import json
import pandas as pd
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
import torch
import gc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import login as hf_login
from huggingface_hub import HfApi

PARAMS = json.load(open("../params.json", "r"))

ESSAY_SET = PARAMS["ESSAY_SET"]
SAMPLING_SIZE = PARAMS["SAMPLING_SIZE"]
MAXLEN_A = PARAMS["MAXLEN_A"]

HF_TOKEN = PARAMS["HF_TOKEN"] 
MODEL_NAME = PARAMS["MODEL_NAME"] 
USERNAME = PARAMS["USERNAME"] 
SEED = PARAMS["SEED"]

from pathlib import Path

for directory in ["data/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

class DatasetTaskClassification(Dataset):
    def __init__(self, df, maxlen_A=MAXLEN_A, label=True, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer
        self.maxlen_A = maxlen_A
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        score = 0.0
        if self.label:
            score = float(self.df.loc[index, "score"])

        a = self.df.loc[index, "essay"]
        sentence2 = str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence2 = sentence2.strip()
                
        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]

        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]
                
        tokens = ["[CLS]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1
        
        return tokens_ids_tensor, attn_mask, score

print(torch.cuda.is_available())

torch.cuda.set_device(CUDA_DEVICE)

import gc
torch.cuda.empty_cache()
gc.collect()

from huggingface_hub import hf_hub_download

repo_name = MODEL_NAME
value = hf_hub_download(f"{USERNAME}/{repo_name}", filename="cls_layer.torch")
cls_layer = torch.load(value, weights_only=False).cuda()

the_model = BertModel.from_pretrained(f"{USERNAME}/"+repo_name).cuda()
the_tokenizer = BertTokenizer.from_pretrained(f"{USERNAME}/"+repo_name, do_lower_case=False)
e = the_model.eval()

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        torch.manual_seed(SEED)
        
        self.bert_layer = the_model
        self.cls_layer = cls_layer
        self.relu = nn.ReLU(inplace=False)

    def forward(self, seq, attn_masks):
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
            
        cls_rep = cont_reps.last_hidden_state[:, 0]

        post_relu = self.relu(cls_rep)
        prelogits = self.cls_layer(post_relu)
        
        return prelogits
    

df = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
df = df[df["split"] == "train"]
indexs = df.index
df = df.reset_index()

data_set = DatasetTaskClassification(df = df, label=False, tokenizer=the_tokenizer)
data_loader = DataLoader(data_set, batch_size = SAMPLING_SIZE+1, num_workers = 2, shuffle=False)

net = RegressionModel()

net.eval()
preds = []
with torch.no_grad():
    for seq, attn_masks, _ in data_loader:
        seq, attn_masks = seq.cuda(), attn_masks.cuda()
        logits = net(seq, attn_masks)
        preds += logits.squeeze().cpu().tolist()

df_train = df.copy()[["essay_id"]]
df_train.index = indexs

df_train["pred"] = preds

df_train.to_csv(f"data/train.csv")