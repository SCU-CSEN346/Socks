CUDA_DEVICE = "cuda:1"

INDEX = 0
assert INDEX in [0,1,2]

import json
PARAMS = json.load(open("params.json", "r"))

SEED = PARAMS["SEED"]
FRAC_TRAIN = PARAMS["FRAC_TRAIN"]
FRAC_VAL = PARAMS["FRAC_VAL"]
LM = PARAMS["LM"]
MAX_ITER = PARAMS["MAX_ITER"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
SAMPLING_SIZE = PARAMS["SAMPLING_SIZE"]
LR = PARAMS["LR"]
MAXLEN_Q = PARAMS["MAXLEN_Q"]
MAXLEN_A = PARAMS["MAXLEN_A"]
HF_TOKEN = PARAMS["HF_TOKEN"]
USERNAME = PARAMS["USERNAME"]

FEATURES = "lf ef ef_lf".split()[INDEX]

from pathlib import Path

for directory in ["pred/", f"pred/{FEATURES}/"]:

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

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
    def __init__(self, df, type_signals, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A):
        self.df = df
        self.tokenizer = bert_tokenizer
        self.maxlen_Q = maxlen_Q
        self.maxlen_A = maxlen_A
        self.type_signals = type_signals

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = int(self.df.loc[index, "label"])

        q, a = self.df.loc[index, 'Q1'], self.df.loc[index, "A1"]
        sentence1, sentence2 = str(q), str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence1, sentence2 = sentence1.strip(), sentence2.strip()
                
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
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor_1 = torch.tensor(tokens_ids)
        attn_mask_1 = (tokens_ids_tensor_1 != 1).long()

        q, a = self.df.loc[index, 'Q2'], self.df.loc[index, "A2"]
        sentence1, sentence2 = str(q), str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence1, sentence2 = sentence1.strip(), sentence2.strip()
                
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
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor_2 = torch.tensor(tokens_ids)
        attn_mask_2 = (tokens_ids_tensor_2 != 1).long()

        ts = self.df.loc[index, "type_signal"]

        b_onehot = [int(t==ts) for t in self.type_signals]
        
        b_onehot_tensor = torch.FloatTensor(b_onehot)

        return tokens_ids_tensor_1, attn_mask_1, tokens_ids_tensor_2, attn_mask_2, b_onehot_tensor, label
    
class DatasetTaskClassification(Dataset):
    def __init__(self, df, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, label=True):
        self.df = df
        self.tokenizer = bert_tokenizer
        self.maxlen_Q = maxlen_Q
        self.maxlen_A = maxlen_A
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        score = 0.0
        if self.label:
            score = float(self.df.loc[index, "nota"])

        q, a = self.df.loc[index, 'Q'], self.df.loc[index, "A"]
        sentence1, sentence2 = str(q), str(a)
        sentence2 = "" if sentence2 == "nan" else sentence2
        sentence1, sentence2 = sentence1.strip(), sentence2.strip()
                
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
                
        tokens = ["[CLS]"]+tokens1+["[SEP]"]+tokens2+["[SEP]"]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = (tokens_ids_tensor != 1).long() # [PAD] => 1
        
        return tokens_ids_tensor, attn_mask, score

class BinaryModel(nn.Module):
    def __init__(self, num_signals=2, confidence_weights=None):
        super(BinaryModel, self).__init__()
        torch.manual_seed(SEED)
        
        self.num_signals = num_signals

        self.bert_layer = BertModel.from_pretrained(model_name).cuda()
        self.cls_layer = nn.Linear(768, 1).cuda()
        self.relu = nn.ReLU(inplace=False)

        self.confidence_layer = nn.Linear(num_signals, 1, bias=False).cuda()

        with torch.no_grad():
            self.confidence_layer.weight.copy_(torch.tensor([confidence_weights]))

    def forward(self, seq_1, attn_masks_1, seq_2, attn_masks_2, b_onehots):
        cont_reps_1 = self.bert_layer(seq_1, attention_mask=attn_masks_1)
        cont_reps_2 = self.bert_layer(seq_2, attention_mask=attn_masks_2)
            
        cls_rep_1 = cont_reps_1.last_hidden_state[:, 0]
        cls_rep_2 = cont_reps_2.last_hidden_state[:, 0]

        post_relu_1 = self.relu(cls_rep_1)
        post_relu_2 = self.relu(cls_rep_2)

        prelogits_1 = self.cls_layer(post_relu_1)
        prelogits_2 = self.cls_layer(post_relu_2)
        
        prelogits = prelogits_1 - prelogits_2

        sigmoids = torch.sigmoid(prelogits)

        preconfidence_coef = self.confidence_layer(b_onehots)

        confidence_coef = torch.sigmoid(preconfidence_coef)

        preds = (confidence_coef * sigmoids) + ((1-confidence_coef) * (1 - sigmoids))

        return preds

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
        for seqs_1, attns_1, seqs_2, attns_2, b_onehots, labels in dataloader:
            seqs_1, attns_1, seqs_2, attns_2, b_onehots, labels = seqs_1.cuda(), attns_1.cuda(), seqs_2.cuda(), attns_2.cuda(), b_onehots.cuda(), labels.cuda()

            logits = net(seqs_1, attns_1, seqs_2, attns_2, b_onehots)
            mean_loss = criterion(logits.squeeze(), labels.float()).item()

            raw_loss +=  mean_loss * labels.shape[0]
            count += labels.shape[0]

            tests += labels.cpu().tolist()
            preds += logits.squeeze().cpu().tolist()

    return preds, tests, raw_loss / count

def train(bin_net, criterion, opti, train_loader, val_loader, epochs):
    
    task_val_loss_values = []

    for ep in range(epochs):
        for it, (seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots, labels) in enumerate(train_loader):
            
            opti.zero_grad()  

            seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots, labels = seqs_1.cuda(), attn_masks_1.cuda(), seqs_2.cuda(), attn_masks_2.cuda(), b_onehots.cuda(), labels.cuda()

            logits = bin_net(seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots)
            
            loss = criterion(logits.squeeze(), labels.float())

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
        for it, (seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots, labels) in enumerate(train_loader):
            
            opti.zero_grad()  

            seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots, labels = seqs_1.cuda(), attn_masks_1.cuda(), seqs_2.cuda(), attn_masks_2.cuda(), b_onehots.cuda(), labels.cuda()

            logits = bin_net(seqs_1, attn_masks_1, seqs_2, attn_masks_2, b_onehots)

            loss = criterion(logits.squeeze(), labels.float())

            loss.backward()

            opti.step()

            if ix + 1 == final_it:
                new_epoch = False
                break

            if ((it + 1) % 1 == 0) or ((it+1) == (len(train_loader))):
                ix += 1
                
        ep += 1

if FEATURES != "ef_lf":
    raw_data_train = pd.read_csv(f"data/{FEATURES}/train.csv", index_col=0)
    raw_data_val = pd.read_csv(f"data/{FEATURES}/val.csv", index_col=0)

    if FEATURES == "lf":
        feature_names = [c for c in raw_data_train.columns if "lf_" in c]
    else:
        feature_names = [c for c in raw_data_train.columns if "ef_traditional" in c and c != "ef_traditional<&>A.is(nan)"]
    
    data_train = pd.melt(raw_data_train, 
        id_vars=[c for c in raw_data_train.columns if c not in feature_names], 
        value_vars=feature_names, 
        var_name='type_signal', 
        value_name='label')
        
    data_val = pd.melt(raw_data_val, 
        id_vars=[c for c in raw_data_val.columns if c not in feature_names], 
        value_vars=feature_names, 
        var_name='type_signal', 
        value_name='label')
    
    data_train = data_train.sample(frac=FRAC_TRAIN, random_state=SEED)
    data_val = data_val.sample(frac=FRAC_VAL, random_state=SEED)

else:
    raw_data_train = pd.read_csv(f"data/lf/train.csv", index_col=0)
    raw_data_val = pd.read_csv(f"data/lf/val.csv", index_col=0)

    o_raw_data_train = pd.read_csv(f"data/lf/train.csv", index_col=0)
    o_raw_data_val = pd.read_csv(f"data/lf/val.csv", index_col=0)

    lf_cols = [c for c in raw_data_train.columns if "lf_" in c]
    ef_cols = [c for c in o_raw_data_train.columns if "ef_traditional" in c and c != "ef_traditional<&>A.is(nan)"]

    feature_names = lf_cols + ef_cols

    data_train = pd.melt(raw_data_train, 
        id_vars=[c for c in raw_data_train.columns if c not in lf_cols], 
        value_vars=lf_cols, 
        var_name='type_signal', 
        value_name='label')
        
    data_val = pd.melt(raw_data_val, 
        id_vars=[c for c in raw_data_val.columns if c not in lf_cols], 
        value_vars=lf_cols, 
        var_name='type_signal', 
        value_name='label')
    
    data_train = data_train.sample(frac=FRAC_TRAIN, random_state=SEED)
    data_val = data_val.sample(frac=FRAC_VAL, random_state=SEED)
    
    o_data_train = pd.melt(o_raw_data_train, 
        id_vars=[c for c in o_raw_data_train.columns if c not in ef_cols], 
        value_vars=ef_cols, 
        var_name='type_signal', 
        value_name='label')
        
    o_data_val = pd.melt(o_raw_data_val, 
        id_vars=[c for c in o_raw_data_val.columns if c not in ef_cols], 
        value_vars=ef_cols, 
        var_name='type_signal', 
        value_name='label')
    
    o_data_train = o_data_train.sample(frac=FRAC_TRAIN, random_state=SEED)
    o_data_val = o_data_val.sample(frac=FRAC_VAL, random_state=SEED)

    data_train = pd.concat([data_train, o_data_train])
    data_val = pd.concat([data_val, o_data_val])

    data_train = data_train.sample(frac=1, random_state=SEED)
    data_val = data_val.sample(frac=1, random_state=SEED)


confidence_weights = [np.log(0.9 / (1 - 0.9)) for _ in feature_names]

data_train = data_train.reset_index().iloc[:MAX_ITER]
data_val = data_val.reset_index()

print(data_train.shape, data_val.shape)

model_name = LM
bert_model = BertModel.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
e = bert_model.eval()

torch.cuda.set_device(CUDA_DEVICE)

torch.cuda.empty_cache()
gc.collect()

train_set = DatasetTaskBinary(df = data_train, type_signals=feature_names)
val_set = DatasetTaskBinary(df = data_val, type_signals=feature_names)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, num_workers = 2, shuffle=False)

bin_net = BinaryModel(num_signals=len(feature_names), confidence_weights=confidence_weights)

criterion = nn.BCELoss().cuda()

opti = optim.Adam(bin_net.parameters(), lr = LR)

print("confidence:", bin_net.confidence_layer.weight.detach().cpu()[0])

epochs = 1
ix_m = train(bin_net, criterion, opti, train_loader, val_loader, epochs)

bin_net = BinaryModel(num_signals=len(feature_names), confidence_weights=confidence_weights)

criterion = nn.BCELoss().cuda()

opti = optim.Adam(bin_net.parameters(), lr = LR)

final_it = ix_m + 1
train_iter(bin_net, criterion, opti, train_loader, final_it)

net = RegressionModel(bin_net)

df = pd.read_csv(f"../data/test.csv")
df = df.reset_index()

data_set = DatasetTaskClassification(df = df, label=False)
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

df_test.to_csv(f"pred/{FEATURES}/test.csv")

hf_token = HF_TOKEN
hf_login(hf_token)

if FEATURES != "ef_lf":
    MODEL_NAME = f"bert_qa_extractor_cockatiel_2022_ulra_bce_org_{FEATURES}_signal_it_{ix_m+1}"
else:
    MODEL_NAME = f"bert_qa_extractor_cockatiel_2022_ulra_bce_org_lf_plus_ef_signal_it_{ix_m+1}"
  
repo_name = MODEL_NAME
bin_net.bert_layer.push_to_hub(repo_name)

train_set.tokenizer.push_to_hub(repo_name)

username = USERNAME
torch.save(bin_net.cls_layer, f"pred/{FEATURES}/cls_layer.torch") 
api = HfApi()

api.upload_file(
    path_or_fileobj=f"pred/{FEATURES}/cls_layer.torch",
    path_in_repo="cls_layer.torch",
    repo_id=f"{username}/{repo_name}",
)

torch.save(bin_net.confidence_layer, f"pred/{FEATURES}/confidence_layer.torch") 
api = HfApi()

api.upload_file(
    path_or_fileobj=f"pred/{FEATURES}/confidence_layer.torch",
    path_in_repo="confidence_layer.torch",
    repo_id=f"{username}/{repo_name}",
)

