CUDA_DEVICE = "cuda:1"
import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
LM = PARAMS["LM"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
LR = PARAMS["LR"]
EPOCHS = PARAMS["EPOCHS"]
HF_TOKEN = PARAMS["HF_TOKEN"]
USERNAME = PARAMS["USERNAME"]
NLLFG_NAME = PARAMS["NLLFG_NAME"]
MAXLEN_Q = PARAMS["MAXLEN_Q"]
MAXLEN_A = PARAMS["MAXLEN_A"]
MAXLEN_B = PARAMS["MAXLEN_B"]

import pandas as pd

df = pd.read_csv("dataset_nllfg.csv", index_col=0)

df_train = df[df["set"] == "train"]
df_val = df[df["set"] == "val"]

df_train = df_train.sample(frac=1, random_state=SEED)

df_train = df_train.reset_index()
df_val = df_val.reset_index()

from transformers import BertTokenizer, BertModel
import torch

print(torch.cuda.is_available())

torch.cuda.set_device(CUDA_DEVICE)

import gc
torch.cuda.empty_cache()
gc.collect()

model_name = LM
the_model = BertModel.from_pretrained(model_name)
the_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
e = the_model.eval()

average_of_tokens = {
    "Q": df_train["Q"].apply(lambda x: len(the_tokenizer.tokenize(x))).describe().to_dict(),
    "A": df_train["A"].apply(lambda x: len(the_tokenizer.tokenize(x))).describe().to_dict(),
    "bsq": df_train["bsq"].apply(lambda x: len(the_tokenizer.tokenize(x))).describe().to_dict()
}

print(average_of_tokens)

import numpy as np

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
    
from torch.utils.data import DataLoader

train_set = DatasetTaskDecision(df = df_train)
val_set = DatasetTaskDecision(df = df_val)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)

import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        torch.manual_seed(SEED-1)
        
        self.bert_layer = BertModel.from_pretrained(model_name).cuda()
        self.cls_layer = nn.Linear(768, 2).cuda()

    def forward(self, seq, attn_masks):

        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        
        cls_rep = cont_reps.last_hidden_state[:, 0]
        
        logits = self.cls_layer(cls_rep)

        return logits
    
def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits)
    soft_probs = probs.argmax(1)
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc
    
def evaluate(net, criterion, dataloader):
    net.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits, labels).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count

def evaluate_precision_recall_fscore_support(net, dataloader):
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
    return tests, preds

def train(net, criterion, opti, train_loader, val_loader, epochs):

    train_loss_values = []
    val_loss_values = []
    
    train_acc_values = []
    val_acc_values = []   
    
    for ep in range(epochs):
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            opti.zero_grad()  

            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            logits = net(seq, attn_masks)

            loss = criterion(logits, labels)

            loss.backward()
            opti.step()

            if ((it + 1) % 25 == 0) or ((it+1) == (len(train_loader))):
                train_acc = get_accuracy_from_logits(logits, labels)
                train_loss = loss.item()
                print("Iteration {} of epoch {} complete. Train Loss : {} Train Accuracy : {}".format(it+1, ep+1, train_loss, train_acc))
                train_loss_values.append(train_loss)
                train_acc_values.append(train_acc.cpu())

                val_acc, val_loss = evaluate(net, criterion, val_loader)
                print("Iteration {} of epoch {} complete. Validation Loss : {} Validation Accuracy : {}".format(it+1, ep+1, val_loss, val_acc))
                val_loss_values.append(val_loss)
                val_acc_values.append(val_acc.cpu())


        ix_m = np.argmin(val_loss_values)

        ix_M = np.argmax(val_acc_values)
    
    return (ix_m, ix_M)

import torch.optim as optim

net = Classifier()

criterion = nn.CrossEntropyLoss().cuda()

opti = optim.Adam(net.parameters(), lr = LR)

epochs = EPOCHS
_, ix_M = train(net, criterion, opti, train_loader, val_loader, epochs)

def train_iter(net, criterion, opti, train_loader, final_it):
    
    new_epoch = True
    ep = 0
    ix = 0
    while new_epoch:
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            opti.zero_grad()

            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            logits = net(seq, attn_masks)

            loss = criterion(logits, labels)

            loss.backward()

            opti.step()

            if ix + 1 == final_it:
                new_epoch = False
                break

            if ((it + 1) % 25 == 0) or ((it+1) == (len(train_loader))):
                ix += 1
                
        ep += 1

net = Classifier()

criterion = nn.CrossEntropyLoss().cuda()

opti = optim.Adam(net.parameters(), lr = LR)

final_it = ix_M + 1
train_iter(net, criterion, opti, train_loader, final_it)

from huggingface_hub import login as hf_login

hf_token = HF_TOKEN
hf_login(hf_token)

repo_name = NLLFG_NAME
net.bert_layer.push_to_hub(repo_name)

train_loader.dataset.tokenizer.push_to_hub(repo_name)

from huggingface_hub import HfApi

username = USERNAME
torch.save(net.cls_layer, "cls_layer.torch") 
api = HfApi()

api.upload_file(
    path_or_fileobj="cls_layer.torch",
    path_in_repo="cls_layer.torch",
    repo_id=f"{username}/{repo_name}",
)