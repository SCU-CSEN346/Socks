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

df = pd.read_csv(f"../../data/test.csv", index_col=0).reset_index()

A_corpus = df.apply(lambda x: f"Respuesta: {x['A']}", axis=1).values
Q_corpus = df.apply(lambda x: f"Pregunta: {x['Q']}", axis=1).values

vectorizer = CountVectorizer(lowercase=True, binary=True, analyzer='word', min_df=5, strip_accents=None)
vectorizer.fit(list(A_corpus)+list(Q_corpus))

A_bow = vectorizer.transform(A_corpus)
Q_bow = vectorizer.transform(Q_corpus)

df_test = df.copy()[["index"]]

df_test["pred"] = [
    1 - distance.jaccard(A_bow.toarray()[i], Q_bow.toarray()[i]) 
    for i in range(A_bow.shape[0])
    ]

df_test.to_csv(f"pred/jaccard/test.csv")