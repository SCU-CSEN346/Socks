import json
PARAMS = json.load(open("../params.json", "r"))

NUM_CLUSTERS = PARAMS["NUM_CLUSTERS"]
MAX_LEN = PARAMS["MAX_LEN"]
MIN_CLUSTER_SIZE = PARAMS["MIN_CLUSTER_SIZE"]

import pandas as pd
import json

sample = pd.read_csv("output/sample.csv", index_col=0)

raw_primitive_bsqs = {ix: sample.loc[ix]["llm"] for ix in sample.index}

with open('bsq_filter.json', 'r') as f:
    filter = json.load(f)

primitive_bsqs = {}
for k in raw_primitive_bsqs.keys():
    o = raw_primitive_bsqs[k]

    pivots = [o.index(f"{i+1}.") for i in range(5)]

    raw_list = [o[pivots[i]: pivots[i+1]].strip() if i < 5-1 else o[pivots[i]: ].strip() for i in range(5)]
    
    raw_clean_list = [r[r.index("¿"): r.index("?")+1] for r in raw_list]
    
    clean_list = []
    for q in raw_clean_list:
        q_mod = q

        if ("¿La respuesta" in q_mod) and (q_mod not in filter["drop_bsqs"]):
            if "(" in q_mod:
                q_mod = q_mod[:q_mod.index("(")-1] + q_mod[q_mod.index(")")+1:]

            for a, b in filter["remove_context"].items():
                q_mod = q_mod.replace(a, b)

            for a, b in filter["to_masculine"].items():  
                q_mod = q_mod.replace(a, b)

            for a, b in filter["remove_string"].items():  
                q_mod = q_mod.replace(a, b)
        
        if ("¿La respuesta" in q_mod) and (q_mod not in filter["drop_bsqs"]) and (len(q_mod.split()) < MAX_LEN):
            clean_list.append(q_mod)

    primitive_bsqs[k] = clean_list

corpus_bsqs = [b for v in primitive_bsqs.values() for b in v]

print("Corpus size:", len(corpus_bsqs))

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SpanishStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stemmer = SpanishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = CountVectorizer(analyzer=stemmed_words)
stem_vectorizer.fit(corpus_bsqs)

from scipy.spatial.distance import cdist
import numpy as np

bow = np.array([stem_vectorizer.transform([b]).toarray()[0] for b in corpus_bsqs])
dis = cdist(bow, bow, metric="jaccard")

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(metric='precomputed', n_clusters=NUM_CLUSTERS, linkage='average')
model.fit(dis)

raw_is_centroid = []
for c in set(model.labels_):

    clusters = model.labels_
    mask = (clusters == c)

    subdis = dis[mask, :][:, mask]

    raw_centroid_ix = subdis.mean(axis=0).argmin()

    centroid_ix = np.where(mask == True)[0][raw_centroid_ix]

    if mask.sum() > MIN_CLUSTER_SIZE:
        raw_is_centroid.append(centroid_ix)

is_centroid = [i in raw_is_centroid for i in range(len(corpus_bsqs))]

clustered_bsqs = pd.DataFrame({"bsq": corpus_bsqs, "cluster": model.labels_, "is_centroid": is_centroid})

for c in clustered_bsqs.cluster.unique():
    o = clustered_bsqs[clustered_bsqs["cluster"] == c]
    o_bsqs = o.bsq.values
    o_centroid = o[o["is_centroid"]]["bsq"].values
    print(f"bsq {c}:", o_centroid)
    print(o_bsqs, "\n")

print("Total of BSQ:", clustered_bsqs["is_centroid"].sum())

clustered_bsqs.to_csv("output/clustered_bsqs.csv", index=0)

print("BSQ:", clustered_bsqs[clustered_bsqs["is_centroid"]].bsq.values)