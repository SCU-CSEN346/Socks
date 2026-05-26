import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
FRAC_TRAIN = PARAMS["FRAC_TRAIN"]

import pandas as pd
import numpy as np

bsq_data_train = pd.read_csv("data/bsq_data_train.csv")

clustered_bsqs = pd.read_csv("output/clustered_bsqs.csv")
bsqs = {}
for c in clustered_bsqs[clustered_bsqs["is_centroid"]].cluster:
    bsqs[f"c{c}"] = clustered_bsqs[(clustered_bsqs["is_centroid"]) & (clustered_bsqs["cluster"] == c)].iloc[0].bsq

labels = {}
for key in bsqs.keys():
    labels[key] = pd.read_excel(f"labels/llm_responses_{key}.xlsx", index_col=0)
    labels[key]["llm_label"] = labels[key]["llm"].apply(lambda x: int("Yes" in str(x)))

dataset = labels[list(bsqs.keys())[0]].copy()
dataset["bsq"] = bsqs[list(bsqs.keys())[0]]
for kwy in list(bsqs.keys())[1:]:

    if kwy not in "".split():

        o = labels[kwy].copy()
        o["bsq"] = bsqs[kwy]

        dataset = pd.concat([dataset, o], axis=0, ignore_index=1)
    
dataset = dataset.drop(columns="domain1_score domain2_score".split()).rename(columns={"llm_label": "label"})
dataset = dataset["index essay bsq label".split()]

bsq_to_index = {v: k for k, v in bsqs.items()}

support_bsqs = dataset.groupby("bsq")["label"].unique()[dataset.groupby("bsq")["label"].unique().apply(len) > 1].index
df = dataset[dataset["bsq"].isin(support_bsqs)]

chatgpt_questions = df["bsq"].unique()

df["b"] = df["bsq"].apply(lambda x: bsq_to_index[x])

df = df.sort_values(["index", "b"]).sample(frac=1, random_state=SEED).reset_index()
num_ex = df["index"].unique().shape[0]

frac_train = FRAC_TRAIN
patch = ["train"]*int(num_ex * frac_train) + ["val"]*(num_ex - int(num_ex * frac_train))

map_to_set = {}
for i, ix in enumerate(df["index"].unique()):
    map_to_set[ix] = patch[i]

dataset["set"] = dataset["index"].apply(lambda x: map_to_set[x])
dataset["b"] = dataset["bsq"].apply(lambda x: bsq_to_index[x])

print("Dataset size:", dataset["set"].value_counts().to_dict())

dataset.to_csv("dataset_nllfg.csv")

print("NLLFG dataset was generated successfully!")