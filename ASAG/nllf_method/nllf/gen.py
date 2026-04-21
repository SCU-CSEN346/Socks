INDEX = 0
assert INDEX in [0,1]
CUDA_DEVICE = f"cuda:{INDEX}"

import json
PARAMS = json.load(open("../params.json", "r"))

USERNAME = PARAMS["USERNAME"]
NLLFG = PARAMS["NLLFG_NAME"]
BATCH_SIZE = PARAMS["PRED_BATCH_SIZE"]
MAXLEN_Q = PARAMS["MAXLEN_Q"]
MAXLEN_A = PARAMS["MAXLEN_A"]
MAXLEN_B = PARAMS["MAXLEN_B"]

from pathlib import Path

directory = "output/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory output/ created successfully!")

import pandas as pd
import numpy as np
import os

data_train_val = pd.read_csv("../../data/train_val.csv", index_col=0).reset_index()
data_train = data_train_val[data_train_val["split"] == "train"]
data_val = data_train_val[data_train_val["split"] == "val"]
data_test = pd.read_csv("../../data/test.csv", index_col=0).reset_index()

from transformers import BertTokenizer, BertModel
import torch

print(torch.cuda.is_available())

torch.cuda.set_device(CUDA_DEVICE)

import gc
torch.cuda.empty_cache()
gc.collect()

from huggingface_hub import hf_hub_url, cached_download

repo_name = NLLFG
config_file_url = hf_hub_url(f"{USERNAME}/"+repo_name, filename="cls_layer.torch")
value = cached_download(config_file_url)
cls_layer = torch.load(value).cuda()

the_model = BertModel.from_pretrained(f"{USERNAME}/"+repo_name).cuda()
the_tokenizer = BertTokenizer.from_pretrained(f"{USERNAME}/"+repo_name, do_lower_case=False)
e = the_model.eval()

def preproccesing(q, a, b, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, maxlen_bsq=MAXLEN_B):
    sentence1 = str(q)
    sentence2 = str(a)
    sentence3 = str(b)
        
    tokens1 = the_tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
    tokens2 = the_tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
    tokens3 = the_tokenizer.tokenize(sentence3) if len(sentence3)>0 else ["[UNK]"]
    
    if len(tokens1) <= maxlen_Q:
        tokens1 = tokens1 + ['[PAD]' for _ in range(maxlen_Q - len(tokens1))]
    else:
        tokens1 = tokens1[:maxlen_Q]

    if len(tokens2) <= maxlen_A:
        tokens2 = tokens2 + ['[PAD]' for _ in range(maxlen_A - len(tokens2))]
    else:
        tokens2 = tokens2[:maxlen_A]

    if len(tokens3) <= maxlen_bsq:
        tokens3 = tokens3 + ['[PAD]' for _ in range(maxlen_bsq - len(tokens3))]
    else:
        tokens3 = tokens3[:maxlen_bsq]
          
    tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]+tokens3+["[SEP]"]
    
    tokens_ids = the_tokenizer.convert_tokens_to_ids(tokens)
    tokens_ids_tensor = torch.tensor(tokens_ids)
    attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1

    return tokens_ids_tensor.cuda(), attn_mask.cuda()

def MixtralClassifier(q, a, b):
    tokens_ids_tensor, attn_mask = preproccesing(q, a, b)
    cont_reps = the_model(tokens_ids_tensor.unsqueeze(0), attention_mask = attn_mask.unsqueeze(0))
    cls_rep = cont_reps.last_hidden_state[:, 0]
    logits = cls_layer(cls_rep)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()[0]

clustered_bsqs = pd.read_csv("../bsq_lab/output/clustered_bsqs.csv")

bsqs = {}
for c in clustered_bsqs[clustered_bsqs["is_centroid"]].cluster:
    bsqs[f"c{c}"] = clustered_bsqs[(clustered_bsqs["is_centroid"]) & (clustered_bsqs["cluster"] == c)].iloc[0].bsq

bsq_to_index = {v: k for k, v in bsqs.items()}

from torch.utils.data import Dataset
class DatasetTaskDecision(Dataset):
    def __init__(self, df, bsq="", maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, maxlen_bsq=MAXLEN_B):
        self.df = df

        self.bsq = bsq

        self.tokenizer = the_tokenizer
        self.maxlen_Q = maxlen_Q
        self.maxlen_A = maxlen_A
        self.maxlen_bsq = maxlen_bsq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence1 = str(self.df.loc[index, 'Q'])
        sentence2 = str(self.df.loc[index, 'A'])

        tokens1 = self.tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]

        if len(tokens1) <= self.maxlen_Q:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_Q - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_Q]

        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]

        sentence3 = str(self.bsq)
        tokens3 = self.tokenizer.tokenize(sentence3) if len(sentence3)>0 else ["[UNK]"]

        if len(tokens3) <= self.maxlen_bsq:
            tokens3 = tokens3 + ['[PAD]' for _ in range(self.maxlen_bsq - len(tokens3))]
        else:
            tokens3 = tokens3[:self.maxlen_bsq]

        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]+tokens3+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1

        ix = self.df.loc[index, 'index']
        
        return ix, tokens_ids_tensor, attn_mask

from torch.utils.data import DataLoader

key = list(bsqs.keys())[0]
bsq = bsqs[key]

train_set = DatasetTaskDecision(df = data_train, bsq = bsq)
val_set = DatasetTaskDecision(df = data_val, bsq = bsq)
test_set = DatasetTaskDecision(df = data_test, bsq = bsq)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)

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

        return cls_rep, logits

net = Classifier()

sub_bsqs = list(bsqs.keys())[:len(bsqs)//2] if '0' in CUDA_DEVICE else list(bsqs.keys())[len(bsqs)//2:]

for key in sub_bsqs:
    with torch.no_grad():
        for set, set_loader in {"train": train_loader, "val": val_loader, "test": test_loader}.items():
            set_loader.dataset.bsq = bsqs[key]
            print(set_loader.dataset.bsq)

            directory = f"output/{set}/{key}"
            for branch in "/idxs /cls_rep /logits".split():
                if not os.path.exists(directory+branch):
                    os.makedirs(directory+branch)

            for it, x in enumerate(set_loader):
                seqs = x[1].cuda()
                atts = x[2].cuda()
                cls_rep, logits = net(seqs, atts)

                cls_rep = cls_rep.detach().cpu()
                logits = logits.detach().cpu()

                idxs = x[0]

                torch.save(idxs, f"{directory}/idxs/it_{it}.pt")
                torch.save(cls_rep, f"{directory}/cls_rep/it_{it}.pt")
                torch.save(logits, f"{directory}/logits/it_{it}.pt")

                print(f"[key: {key}] [{set}] It: {it} complete. Progress: {100*(it+1)/len(set_loader):.0f}%")


