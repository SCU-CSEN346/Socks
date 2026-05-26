import json
PARAMS = json.load(open("params.json", "r"))

LM = PARAMS["LM"]
ESSAY_SET = PARAMS["ESSAY_SET"]

from pathlib import Path

for directory in ["pred/", "pred/cosine/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer(LM)

def classification_model(q, a):

    sentences1 = [q]
    sentences2 = [a]

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    return cosine_scores[0].squeeze().cpu().tolist()

def prep(text):
        return str(text).replace("\n", " ").strip()

df = pd.read_csv(f"../../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()
df = df[df["split"] == "test"]

df_test = df.copy()[["essay_id"]]

with open('../../data/question.json', 'r') as f:
    meta_type = json.load(f)

q = meta_type[str(ESSAY_SET)]['question']
q = prep(q) if str(q) != "nan" else ""

df_test["pred"] = df.apply(lambda row:
                        classification_model(
                            q,
                            prep(row["essay"]) if str(row["essay"]) != "nan" else ""),
                            axis=1)

df_test.to_csv(f"pred/cosine/test.csv")