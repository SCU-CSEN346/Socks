import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
PARAMS = json.load(open("../params.json", "r"))

BATCH_SIZE = PARAMS["PRED_BATCH_SIZE"]
MAXLEN_A = PARAMS["MAXLEN_A"]
MAXLEN_B = PARAMS["MAXLEN_B"]
ESSAY_SET = PARAMS["ESSAY_SET"]
USERNAME = PARAMS["USERNAME"]
NLLFG = PARAMS["NLLFG_NAME"]

CUDA_DEVICE = "cuda:0"

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

directory = "output/"
Path(directory).mkdir(parents=True, exist_ok=True)
print("Directory output/ created successfully!")

data = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()

print(torch.cuda.is_available())
torch.cuda.set_device(CUDA_DEVICE)
torch.cuda.empty_cache()
gc.collect()

repo_name = NLLFG
value = hf_hub_download(repo_id=f"{USERNAME}/{repo_name}", filename="cls_layer.torch")
cls_layer = torch.load(value, weights_only=False).cuda()
the_model = BertModel.from_pretrained(f"{USERNAME}/{repo_name}").cuda()
the_tokenizer = BertTokenizer.from_pretrained(f"{USERNAME}/{repo_name}", do_lower_case=False)
e = the_model.eval()

clustered_bsqs = pd.read_csv("../bsq_lab/output/clustered_bsqs.csv")

bsqs = {}
for c in clustered_bsqs[clustered_bsqs["is_centroid"]].cluster:
    bsqs[f"c{c}"] = clustered_bsqs[(clustered_bsqs["is_centroid"]) & (clustered_bsqs["cluster"] == c)].iloc[0].bsq

bsq_to_index = {v: k for k, v in bsqs.items()}

print(f"Total BSQ clusters: {len(bsqs)}")
print(f"Cluster keys: {list(bsqs.keys())}")

class DatasetTaskDecision(Dataset):
    def __init__(self, df, bsq="", maxlen_A=MAXLEN_A, maxlen_bsq=MAXLEN_B):
        self.df = df
        self.bsq = bsq
        self.tokenizer = the_tokenizer
        self.maxlen_A = maxlen_A
        self.maxlen_bsq = maxlen_bsq

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence2 = str(self.df.loc[index, 'EssayText'])

        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]

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

        tokens = ["[CLS]"]+tokens2+["[SEP]"]+tokens3+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long()

        ix = self.df.loc[index, 'index']

        return ix, tokens_ids_tensor, attn_mask

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

key = list(bsqs.keys())[0]
bsq = bsqs[key]

data_set = DatasetTaskDecision(df=data, bsq=bsq)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

all_bsqs = list(bsqs.keys())

for key in all_bsqs:
    with torch.no_grad():
        for set_name, set_loader in {"data": data_loader}.items():
            set_loader.dataset.bsq = bsqs[key]
            print(f"\n--- Processing {key} ---")
            print(f"BSQ: {set_loader.dataset.bsq}")

            directory = f"output/{set_name}/{key}"
            for branch in "/idxs /cls_rep /logits".split():
                Path(directory + branch).mkdir(parents=True, exist_ok=True)

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

                print(f"[key: {key}] [{set_name}] It: {it} complete. Progress: {100*(it+1)/len(set_loader):.0f}%")

print("\nAll BSQs processed successfully!")
