CUDA_DEVICE = "cuda:0"
import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"] 
FRAC_TRAIN = PARAMS["FRAC_TRAIN"] 
FRAC_VAL = PARAMS["FRAC_VAL"] 
LM = PARAMS["LM"] 
BATCH_SIZE = PARAMS["BATCH_SIZE"] 
SAMPLING_SIZE = PARAMS["SAMPLING_SIZE"] 

MAXLEN_Q = PARAMS["MAXLEN_Q"] 
MAXLEN_A = PARAMS["MAXLEN_A"] 
HF_TOKEN = PARAMS["HF_TOKEN"] 
MODEL_NAME = PARAMS["MODEL_NAME"] 
USERNAME = PARAMS["USERNAME"] 

LR_AES = PARAMS["LR_AES"]
LR_ETA = PARAMS["LR_ETA"]
DECAY = PARAMS["DECAY"]
NUM_EPOCHS = PARAMS["NUM_EPOCHS"]

from pathlib import Path

for directory in ["pred/", "pred/ef/"]:

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

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

class DatasetTaskBinary(Dataset):
    def __init__(self, df, type_signals, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer
        self.maxlen_A = maxlen_A
        self.maxlen_Q = maxlen_Q
        self.type_signals = type_signals

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        q, a = self.df.loc[index, "Q A".split()]
        sentence1 = str(q)
        sentence2 = str(a)

        sentence2 = "" if sentence2 == "nan" else sentence2
        
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()

        tokens1 = self.tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        if len(tokens1) <= self.maxlen_Q:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_Q - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_Q]

        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor_1 = torch.tensor(tokens_ids)
        attn_mask_1 = (tokens_ids_tensor_1 != 1).long() # [PAD] => 1

        features = self.df.loc[index, self.type_signals]
        
        features = torch.FloatTensor(pd.to_numeric(features.values))

        return tokens_ids_tensor_1, attn_mask_1, features
    
class DatasetTaskClassification(Dataset):
    def __init__(self, df, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, label=True, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer
        self.maxlen_A = maxlen_A
        self.maxlen_Q = maxlen_Q
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        score = 0.0
        if self.label:
            score = float(self.df.loc[index, "score"])

        q, a = self.df.loc[index, "Q A".split()]

        sentence1 = str(q)
        sentence2 = str(a)

        sentence2 = "" if sentence2 == "nan" else sentence2

        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()

        tokens1 = self.tokenizer.tokenize(sentence1) if len(sentence1)>0 else ["[UNK]"]
        if len(tokens1) <= self.maxlen_Q:
            tokens1 = tokens1 + ['[PAD]' for _ in range(self.maxlen_Q - len(tokens1))]
        else:
            tokens1 = tokens1[:self.maxlen_Q]
                
        tokens2 = self.tokenizer.tokenize(sentence2) if len(sentence2)>0 else ["[UNK]"]
        if len(tokens2) <= self.maxlen_A:
            tokens2 = tokens2 + ['[PAD]' for _ in range(self.maxlen_A - len(tokens2))]
        else:
            tokens2 = tokens2[:self.maxlen_A]
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1
        
        return tokens_ids_tensor, attn_mask, score

class BinaryModel(nn.Module):
    def __init__(self, num_signals=2, confidence_weights=[], model_name=""):
        super(BinaryModel, self).__init__()
        torch.manual_seed(SEED)
        
        self.num_signals = num_signals

        self.bert_layer = BertModel.from_pretrained(model_name).cuda()
        self.cls_layer = nn.Linear(768, 1).cuda()
        self.relu = nn.ReLU(inplace=False)

        self.confidence_layer = nn.Linear(num_signals, 1, bias=False).cuda()

        with torch.no_grad():
            self.confidence_layer.weight.copy_(torch.tensor([confidence_weights]))

    def forward(self, seq_1, attn_masks_1):

        cont_reps_1 = self.bert_layer(seq_1, attention_mask=attn_masks_1)
                    
        cls_rep_1 = cont_reps_1.last_hidden_state[:, 0]

        post_relu_1 = self.relu(cls_rep_1)

        prelogits_1 = self.cls_layer(post_relu_1)

        confidence_weight = self.confidence_layer.weight[0]
        confidence_weight = confidence_weight.view(-1, 1)

        confidence_weight = torch.sigmoid(confidence_weight)

        return prelogits_1, confidence_weight

class RegressionModel(nn.Module):
    def __init__(self, bin_net: BinaryModel):
        super(RegressionModel, self).__init__()
        torch.manual_seed(SEED)
        
        self.bert_layer = bin_net.bert_layer
        self.cls_layer = bin_net.cls_layer
        self.relu = bin_net.relu

    def forward(self, seq, attn_masks):
        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
            
        cls_rep = cont_reps.last_hidden_state[:, 0]

        post_relu = self.relu(cls_rep)
        prelogits = self.cls_layer(post_relu)
        
        return prelogits

def bin_get_accuracy(prelogits, labels):
    probs = np.array(prelogits)
    labels = np.array(labels)

    soft_probs = (probs>0.5).astype(int)
    acc = (soft_probs == labels).mean()
    return acc


def bin_evaluate(net, dataloader, criterion):
    net.eval()
    preds = []
    tests = []
    raw_loss = 0
    count = 0
    with torch.no_grad():
        for seqs_1, attns_1, features_1 in dataloader:
            seqs_1, attns_1, features_1 = seqs_1.cuda(), attns_1.cuda(), features_1.cuda()

            prelogits_1, confidence_weight_1 = net(seqs_1, attns_1)

            pairwise_differences = prelogits_1 - prelogits_1.T
            sigmoids = torch.sigmoid(pairwise_differences)
            triu_mask = torch.triu(torch.ones_like(sigmoids), diagonal=1) != 0
            sigmoids = sigmoids[triu_mask]

            model_preds = (confidence_weight_1 * sigmoids) + ((1-confidence_weight_1) * (1-sigmoids))

            features_differences = (features_1.unsqueeze(2) - features_1.unsqueeze(2).transpose(0, 2)).permute(1, 0, 2)
            features_differences = features_differences[:, triu_mask]
            labels = (features_differences > 0).float()
            
            mean_loss = criterion(model_preds, labels).item()

            raw_loss +=  mean_loss * labels.shape[0]
            count += labels.shape[0]

            tests += labels.cpu().tolist()
            preds += model_preds.squeeze().cpu().tolist()

    return preds, tests, raw_loss / count

def train(bin_net, criterion, opti, train_loader, val_loader, epochs):
    
    task_val_loss_values = []

    for ep in range(epochs):
        for it, (seqs_1, attns_1, features_1) in enumerate(train_loader):
            seqs_1, attns_1, features_1 = seqs_1.cuda(), attns_1.cuda(), features_1.cuda()
            opti.zero_grad()  

            prelogits_1, confidence_weight_1 = bin_net(seqs_1, attns_1)

            pairwise_differences = prelogits_1 - prelogits_1.T
            sigmoids = torch.sigmoid(pairwise_differences)
            triu_mask = torch.triu(torch.ones_like(sigmoids), diagonal=1) != 0
            sigmoids = sigmoids[triu_mask]

            preds = (confidence_weight_1 * sigmoids) + ((1-confidence_weight_1) * (1-sigmoids))

            features_differences = (features_1.unsqueeze(2) - features_1.unsqueeze(2).transpose(0, 2)).permute(1, 0, 2)
            features_differences = features_differences[:, triu_mask]
            labels = (features_differences > 0).float()

            loss = criterion(preds, labels)

            loss.backward()

            opti.step()

            if (it+1) % 1 in [0] or (it+1) in [len(train_loader)]:

                _, _, task_val_loss = bin_evaluate(bin_net, val_loader, criterion)
                task_val_loss_values.append(task_val_loss)

            if (it+1) % 25 in [0] or (it+1) in [len(train_loader)]:
                print("Iteration {} of epoch {} complete. Task-validation loss: {}".format(it+1, ep+1, task_val_loss))

        ix_m = np.argmin(task_val_loss_values)

    return ix_m

def train_iter(bin_net, criterion, opti, train_loader, final_it):
    
    new_epoch = True
    ep = 0
    ix = 0
    while new_epoch:
        for it, (seqs_1, attns_1, features_1) in enumerate(train_loader):
            seqs_1, attns_1, features_1 = seqs_1.cuda(), attns_1.cuda(), features_1.cuda()
            opti.zero_grad()  

            prelogits_1, confidence_weight_1 = bin_net(seqs_1, attns_1)

            pairwise_differences = prelogits_1 - prelogits_1.T
            sigmoids = torch.sigmoid(pairwise_differences)
            triu_mask = torch.triu(torch.ones_like(sigmoids), diagonal=1) != 0
            sigmoids = sigmoids[triu_mask]

            preds = (confidence_weight_1 * sigmoids) + ((1-confidence_weight_1) * (1-sigmoids))

            features_differences = (features_1.unsqueeze(2) - features_1.unsqueeze(2).transpose(0, 2)).permute(1, 0, 2)
            features_differences = features_differences[:, triu_mask]
            labels = (features_differences > 0).float()

            loss = criterion(preds, labels)

            loss.backward()

            opti.step()

            if ix + 1 == final_it:
                new_epoch = False
                break

            if ((it + 1) % 1 == 0) or ((it+1) == (len(train_loader))):
                ix += 1
                
        ep += 1


train_text_data = pd.read_csv(f"../../data/train.csv", index_col=0)
train_features_data = pd.read_csv("../../nllf_method/ef/data/train.csv", index_col=0)

raw_train_data = train_features_data.copy()
raw_train_data["Q"] = train_text_data["Q"]
raw_train_data["A"] = train_text_data["A"]

train_data = raw_train_data.copy()

val_text_data = pd.read_csv(f"../../data/val.csv", index_col=0)
val_features_data = pd.read_csv("../../nllf_method/ef/data/val.csv", index_col=0)

raw_val_data = val_features_data.copy()
raw_val_data["Q"] = val_text_data["Q"]
raw_val_data["A"] = val_text_data["A"]

val_data = raw_val_data.copy()

feature_names = [c for c in train_features_data.columns if c.startswith("traditional<&>")]
confidence_weights = [np.log(0.9 / (1 - 0.9)) for _ in feature_names]

data_train = train_data.sample(frac=FRAC_TRAIN, random_state=SEED)

data_val = val_data.sample(frac=FRAC_TRAIN, random_state=SEED)

data_train = data_train.reset_index()
data_val = data_val.reset_index()

print(data_train.shape, data_val.shape)

torch.cuda.set_device(CUDA_DEVICE)

torch.cuda.empty_cache()
gc.collect()

model_name = LM
bert_model = BertModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
e = bert_model.eval();

train_set = DatasetTaskBinary(df = data_train, type_signals=feature_names, tokenizer=bert_tokenizer)
val_set = DatasetTaskBinary(df = data_val, type_signals=feature_names, tokenizer=bert_tokenizer)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)

bin_net = BinaryModel(
    num_signals=len(feature_names),
    confidence_weights=confidence_weights,
    model_name = LM
    )

criterion = nn.BCELoss().cuda()

params_aes = [param for name, param in bin_net.named_parameters() if 'bert_layer' in name or 'cls_layer' in name]
params_eta = [param for name, param in bin_net.named_parameters() if 'confidence_layer' in name]

opti = optim.AdamW([
    {'params': params_aes, 'lr': LR_AES},
    {'params': params_eta, 'lr': LR_ETA}
], weight_decay=DECAY)

print("confidence:", bin_net.confidence_layer.weight.detach().cpu()[0])
print(CUDA_DEVICE)

epochs = NUM_EPOCHS
ix_m = train(bin_net, criterion, opti, train_loader, val_loader, epochs)

final_it = ix_m + 1
train_iter(bin_net, criterion, opti, train_loader, final_it)

net = RegressionModel(bin_net)

df = pd.read_csv(f"../../data/test.csv")
df = df.reset_index()

data_set = DatasetTaskClassification(df = df, label=False, tokenizer=bert_tokenizer)
data_loader = DataLoader(data_set, batch_size = SAMPLING_SIZE, num_workers = 2, shuffle=False)

net.eval()
preds = []
with torch.no_grad():
    for seq, attn_masks, _ in data_loader:
        seq, attn_masks = seq.cuda(), attn_masks.cuda()
        logits = net(seq, attn_masks)
        preds += logits.squeeze().cpu().tolist()

df_test = df.copy()[["index"]]
df_test["pred"] = preds

df_test.to_csv(f"pred/ef/test.csv")

hf_token = HF_TOKEN
hf_login(hf_token)

repo_name = MODEL_NAME.replace("lf", "ef")
bin_net.bert_layer.push_to_hub(repo_name)

train_set.tokenizer.push_to_hub(repo_name)

username = USERNAME
torch.save(bin_net.cls_layer, "pred/ef/cls_layer.torch") 
api = HfApi()

api.upload_file(
    path_or_fileobj="pred/ef/cls_layer.torch",
    path_in_repo="cls_layer.torch",
    repo_id=f"{username}/{repo_name}",
)

torch.save(bin_net.confidence_layer, "pred/ef/confidence_layer.torch") 
api = HfApi()

api.upload_file(
    path_or_fileobj="pred/ef/confidence_layer.torch",
    path_in_repo="confidence_layer.torch",
    repo_id=f"{username}/{repo_name}",
)
