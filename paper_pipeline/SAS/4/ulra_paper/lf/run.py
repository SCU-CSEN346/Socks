import json
PARAMS = json.load(open("../params.json", "r"))

ESSAY_SET = PARAMS["ESSAY_SET"]

from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

from feature_extractor import ULRAFeatureExtractor, pd

import json

basic_vocab = json.load(open("basic_vocab.json", "r"))["list"]

data = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)

A = data.copy().reset_index()

preproc = ULRAFeatureExtractor(basic_vocab)
df_features = preproc.get_preprocessing(A, text_name="EssayText")

df_features["index"] = A["index"]
df_features["split"] = A["split"]

df_features.to_csv(f"data/data.csv")
