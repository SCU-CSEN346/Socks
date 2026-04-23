from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

import json
import pandas as pd 
import numpy as np

output_dir = Path("../data/aes_signal")
output_dir.mkdir(parents=True, exist_ok=True)

for ESSAY_SET in range(1, 9):
    print(f"Processing essay set {ESSAY_SET}...")

    df = pd.read_csv(f"../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()
    df = df[df["split"] == "train"]

    corpus = df.apply(lambda x: f"Essay: {x['essay']}", axis=1)

    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        analyzer='word',
        min_df=5,
        strip_accents=None
    )

    bow = vectorizer.fit_transform(corpus)

    bool_bow = csr_matrix((bow.toarray() > 0).astype(np.uint8))

    jaccard_distance = cdist(bool_bow.toarray(), bool_bow.toarray(), metric="jaccard")
    jaccard_sim = 1 - jaccard_distance

    np.save(output_dir / f"sim_matrix_{ESSAY_SET}_aes.npy", jaccard_sim)
    print(f"Saved sim_matrix_{ESSAY_SET}.npy")