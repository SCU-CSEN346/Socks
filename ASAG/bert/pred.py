INDEX = 0
assert INDEX in [0,1]

WITH_SIGNAL_FILTERING = [False, True][INDEX]
CUDA_DEVICE = f"cuda:{INDEX}"

import json
PARAMS = json.load(open("params.json", "r"))

S_MAX = PARAMS["S_MAX"]
SEED = PARAMS["SEED"]
LM = PARAMS["LM"]
BATCH_SIZE_TRAIN = PARAMS["BATCH_SIZE_TRAIN"]
BATCH_SIZE_VAL = PARAMS["BATCH_SIZE_VAL"]
LR = PARAMS["LR"]
EPOCHS = PARAMS["EPOCHS"]
HF_TOKEN = PARAMS["HF_TOKEN"]
USERNAME = PARAMS["USERNAME"]
FRAC_TRAIN = PARAMS["FRAC_TRAIN"]
FRAC_VAL = PARAMS["FRAC_VAL"]
MAXLEN_Q = PARAMS["MAXLEN_Q"]
MAXLEN_A = PARAMS["MAXLEN_A"]

import pandas as pd
import numpy as np
import pandas as pd

from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch
import gc
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim    
from scipy import stats
from huggingface_hub import login as hf_login
from huggingface_hub import HfApi

def score2label(score, res=0, size=7, m=0.49, M=1.69):
    step = (M-m) / (size)
    if score < step + m:
        return 1+res
    else:
        return score2label(score-step, res+1, size, m, M)
    
class DatasetTaskDecision(Dataset):
    def __init__(self, df, maxlen_Q=MAXLEN_Q, maxlen_A=MAXLEN_A, sc=False, label=True):
        self.df = df
        self.tokenizer = the_tokenizer
        self.maxlen_Q = maxlen_Q
        self.maxlen_A = maxlen_A
        self.sc = sc
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = 0.0
        if self.label:
            label = float(self.df.loc[index, "nota"])
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
        
        sample_weight = 1.0
        if self.sc:
            sample_weight = float(self.df.loc[index, "signal_filtering"])

        return tokens_ids_tensor, attn_mask, label, sample_weight
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        torch.manual_seed(SEED)
        
        self.bert_layer = BertModel.from_pretrained(model_name).cuda()
        self.cls_layer = nn.Linear(768, 1).cuda()

    def forward(self, seq, attn_masks):

        cont_reps = self.bert_layer(seq, attention_mask=attn_masks)
        
        cls_rep = cont_reps.last_hidden_state[:, 0]
        
        logits = self.cls_layer(cls_rep)

        return logits
        
def get_pearson(logits, labels):
    corr = stats.pearsonr(labels, logits).statistic
    return corr
    
def evaluate(net, criterion, dataloader):
    net.eval()
    mean_loss = 0
    count = 0
    preds = []
    tests = []
    with torch.no_grad():
        for seq, attn_masks, labels, _ in dataloader:
            seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.to(labels.dtype), labels).mean().item()
            count += 1

            tests += labels.cpu().tolist()
            preds += logits.squeeze().cpu().tolist()

    return get_pearson(preds, tests), mean_loss / count

def train(net, criterion, opti, train_loader, val_loader, epochs):

    train_loss_values = []
    val_loss_values = []
    
    train_acc_values = []
    val_acc_values = []   
    
    for ep in range(epochs):
        for it, (seq, attn_masks, labels, sample_weights) in enumerate(train_loader):
            opti.zero_grad()  

            seq, attn_masks, labels, sample_weights = seq.cuda(), attn_masks.cuda(), labels.cuda(), sample_weights.cuda()

            logits = net(seq, attn_masks)

            loss = criterion(logits.squeeze().to(labels.dtype), labels)

            loss = loss * sample_weights

            loss.mean().backward()
            opti.step()

            if ((it + 1) % 25 == 0) or ((it+1) == (len(train_loader))):
                
                train_acc = get_pearson(logits.squeeze().cpu().tolist(), labels.cpu().tolist())
                train_loss = loss.mean().item()
                print("Iteration {} of epoch {} complete. Train Loss : {} Train corr. : {}".format(it+1, ep+1, train_loss, train_acc))
                train_loss_values.append(train_loss)
                train_acc_values.append(train_acc)

                val_acc, val_loss = evaluate(net, criterion, val_loader)
                print("Iteration {} of epoch {} complete. Validation Loss : {} Validation corr. : {}".format(it+1, ep+1, val_loss, val_acc))
                val_loss_values.append(val_loss)
                val_acc_values.append(val_acc)

        ix_m = np.argmin(val_loss_values)

        ix_M = np.argmax(val_acc_values)
    
    return (ix_m, ix_M)

def train_iter(net, criterion, opti, train_loader, final_it):
    
    new_epoch = True
    ep = 0
    ix = 0
    while new_epoch:
        for it, (seq, attn_masks, labels, sample_weights) in enumerate(train_loader):
            opti.zero_grad()

            seq, attn_masks, labels, sample_weights = seq.cuda(), attn_masks.cuda(), labels.cuda(), sample_weights.cuda()

            logits = net(seq, attn_masks)

            loss = criterion(logits.squeeze().to(labels.dtype), labels)
            
            loss = loss * sample_weights

            loss.mean().backward()

            opti.step()

            if ix + 1 == final_it:
                new_epoch = False
                break

            if ((it + 1) % 25 == 0) or ((it+1) == (len(train_loader))):
                ix += 1
                
        ep += 1

for SIGNAL_PATH in ["/llm/weak_signal/data", "/signal_clustering/weak_signal/data"]:

    foo = "w" if WITH_SIGNAL_FILTERING else "wo"
    bar = "z_score"
    if "llm" in SIGNAL_PATH:
        bar = "llm"

    for directory in ["pred/", f"pred/{bar}/", f"pred/{bar}/{foo}_sf/"]:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

    print("Directory pred/ created successfully!")
    
    if WITH_SIGNAL_FILTERING:
        df_gt_1 = pd.read_csv("../llm/weak_signal/data/train_val.csv", index_col=0)
        df_gt_2 = pd.read_csv("../signal_clustering/weak_signal/data/train_val.csv", index_col=0)

        df_gt_1 = df_gt_1[df_gt_1["split"] == "train"]
        df_gt_2 = df_gt_2[df_gt_2["split"] == "train"]

        df_gt = df_gt_1.copy()
        df_gt["other_pred"] = df_gt_2["pred"]

        y_real_2 = df_gt["other_pred"]
        if len(set(y_real_2)) > S_MAX:
            m, M = np.percentile(y_real_2, 1), np.percentile(y_real_2, 99)
            preds = y_real_2.apply(lambda x: score2label(x, res=0, size=S_MAX, m=m, M=M)).values
            preds = np.array([p if 0 < p and p <= S_MAX  else 1 if p <= 0 else S_MAX for p in preds])

        df_gt["other_pred_quantize"] = preds
        signal_filtering = 1 - ((df_gt["pred"] - df_gt["other_pred_quantize"]).abs() / (S_MAX-1))

    df_data = pd.read_csv("../data/train_val.csv")

    df_gt = pd.read_csv(f"..{SIGNAL_PATH}/train_val.csv")
    df_data["nota"] = df_gt["pred"]

    df_train = df_data[df_data["split"] == "train"]
    df_val = df_data[df_data["split"] == "val"]

    m, M = df_train["nota"].min(), df_train["nota"].max()
    df_train["nota"] = (df_train["nota"]-m)/(M-m)
    df_val["nota"] = (df_val["nota"]-m)/(M-m)

    if WITH_SIGNAL_FILTERING:
        df_train["signal_filtering"] = signal_filtering

    df_train = df_train.sample(frac=FRAC_TRAIN, random_state=SEED)
    df_val = df_val.sample(frac=FRAC_VAL, random_state=SEED)

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    print(torch.cuda.is_available())

    torch.cuda.set_device(CUDA_DEVICE)

    torch.cuda.empty_cache()
    gc.collect()

    model_name = LM
    the_model = BertModel.from_pretrained(model_name)
    the_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    e = the_model.eval()

    average_of_tokens = {
        "Q": df_train["Q"].apply(lambda x: len(the_tokenizer.tokenize(x))).describe().to_dict(),
        "A": df_train["A"].apply(lambda x: len(the_tokenizer.tokenize(x))).describe().to_dict()
    }

    print(average_of_tokens)

    train_set = DatasetTaskDecision(df = df_train, sc=WITH_SIGNAL_FILTERING, label=True)
    val_set = DatasetTaskDecision(df = df_val, sc=False, label=True)

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE_TRAIN, num_workers = 2, shuffle=False)
    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE_VAL, num_workers = 2, shuffle=False)

    net = Classifier()

    criterion = nn.MSELoss(reduction="none").cuda()

    opti = optim.Adam(net.parameters(), lr = LR)

    epochs = EPOCHS
    _, ix_M = train(net, criterion, opti, train_loader, val_loader, epochs)

    net = Classifier()

    criterion = nn.MSELoss(reduction="none").cuda()

    opti = optim.Adam(net.parameters(), lr = LR)

    final_it = ix_M + 1
    train_iter(net, criterion, opti, train_loader, final_it)

    df = pd.read_csv(f"../data/test.csv")
    df = df.reset_index()

    data_set = DatasetTaskDecision(df = df, sc=False, label=False)
    data_loader = DataLoader(data_set, batch_size = BATCH_SIZE_TRAIN, num_workers = 2, shuffle=False)

    net.eval()
    preds = []
    with torch.no_grad():
        for seq, attn_masks, _, _ in data_loader:
            seq, attn_masks = seq.cuda(), attn_masks.cuda()
            logits = net(seq, attn_masks)
            preds += logits.squeeze().cpu().tolist()

    df_test = df.copy()[["index"]]
    df_test["pred"] = preds

    df_test.to_csv(f"pred/{bar}/{foo}_sf/test.csv")

    hf_token = HF_TOKEN
    hf_login(hf_token)

    infix = "_linear_weight" if WITH_SIGNAL_FILTERING else ""
    if bar == "llm":
        MODEL_NAME = f"bert_qa_extractor_cockatiel_2022_mixtral_v2{infix}_it_{ix_M + 1}"
    else:
        MODEL_NAME = f"bert_qa_extractor_cockatiel_2022_z_value{infix}_it_{ix_M + 1}"

    repo_name = MODEL_NAME
    net.bert_layer.push_to_hub(repo_name)

    train_loader.dataset.tokenizer.push_to_hub(repo_name)

    username = USERNAME
    torch.save(net.cls_layer, f"pred/{bar}/{foo}_sf/cls_layer.torch") 
    api = HfApi()

    api.upload_file(
        path_or_fileobj=f"pred/{bar}/{foo}_sf/cls_layer.torch",
        path_in_repo="cls_layer.torch",
        repo_id=f"{username}/{repo_name}",
    )