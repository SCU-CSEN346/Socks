import pandas as pd 
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

output_dir = Path("../data/sas_signal")
output_dir.mkdir(parents=True, exist_ok=True)

for ESSAY_SET in range(1, 11):
    print(f"Processing essay set {ESSAY_SET}...")

    df = pd.read_csv(f"../data/sas/essay_set_{ESSAY_SET}_sas.csv")

    df_train = df[df['split'] == 'train'].copy()

    corpus = df_train.apply(lambda x: f"Answer: {x['essaytext']}", axis=1)

    vectorizer = CountVectorizer(lowercase=True, binary=True, min_df=5)
    bow = vectorizer.fit_transform(corpus)
    bool_bow = bow.toarray()

    jaccard_distance = cdist(bool_bow, bool_bow, metric="jaccard")
    jaccard_sim = 1 - jaccard_distance

    np.save(output_dir / f"sim_matrix_{ESSAY_SET}_sas.npy", jaccard_sim)
    print(f"Matrix saved! Shape: {jaccard_sim.shape}")