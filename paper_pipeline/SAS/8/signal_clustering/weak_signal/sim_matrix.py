from pathlib import Path

import json
PARAMS = json.load(open("../params.json", "r"))

ESSAY_SET = PARAMS["ESSAY_SET"]

for directory in ["data/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

import pandas as pd
import numpy as np

df = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()
df = df[df["split"] == "train"]

corpus = df.apply(lambda x: f"Answer: {x['EssayText']}", axis=1)

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

vectorizer = CountVectorizer(lowercase=True, binary=True, analyzer='word', min_df=5, strip_accents=None)

bow = vectorizer.fit_transform(corpus)

bool_bow = csr_matrix((bow.toarray()>0).astype(np.uint8))

from scipy.spatial.distance import cdist

jaccard_distance = cdist(bool_bow.toarray(), bool_bow.toarray(), metric="jaccard")

jaccard_sim = 1 - jaccard_distance

np.save(open(f"data/sim_matrix.npy", "wb"), jaccard_sim)
