import json
PARAMS = json.load(open("params.json", "r"))

ESSAY_SET = PARAMS["ESSAY_SET"]

from pathlib import Path

for directory in ["pred/", "pred/jaccard/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial import distance

df = pd.read_csv(f"../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
df = df[df["split"] == "test"]
indexs = df.index
df = df.reset_index()

with open('../../data/question.json', 'r') as f:
    meta_type = json.load(f)

A_corpus = df.apply(lambda x: f"Answer: {x['EssayText']}", axis=1).values
Q_corpus = df.apply(lambda x: f"Question: {meta_type[str(ESSAY_SET)]['question']}", axis=1).values

vectorizer = CountVectorizer(lowercase=True, binary=True, analyzer='word', min_df=2, strip_accents=None)
vectorizer.fit(list(A_corpus)+list(Q_corpus))

A_bow = vectorizer.transform(A_corpus)
Q_bow = vectorizer.transform(Q_corpus)

df_test = df.copy()[["Id"]]

df_test["pred"] = [
    1 - distance.jaccard(A_bow.toarray()[i], Q_bow.toarray()[i])
    for i in range(A_bow.shape[0])
    ]

df_test.to_csv(f"pred/jaccard/test.csv")
