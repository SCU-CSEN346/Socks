import json
PARAMS = json.load(open("params.json", "r"))

LM = PARAMS["LM"]

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

df = pd.read_csv(f"../../data/test.csv", index_col=0).reset_index()

df_test = df.copy()[["index"]]

df_test["pred"] = df.apply(lambda row:
                        classification_model(
                            prep(row["Q"]) if str(row["Q"]) != "nan" else "",
                            prep(row["A"]) if str(row["A"]) != "nan" else ""),
                            axis=1)

df_test.to_csv(f"pred/cosine/test.csv")