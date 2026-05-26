import json
PARAMS = json.load(open("params.json", "r"))

SEED = PARAMS["SEED"]
S_MAX = PARAMS["S_MAX"]
S_MIN = PARAMS["S_MIN"]
ESSAY_SET = PARAMS["ESSAY_SET"]

from pathlib import Path

for directory in ["pred/", "pred/random/", "pred/length"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

import pandas as pd
import numpy as np

df = pd.read_csv(f"../../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()
df = df[df["split"] == "test"]

N = df.shape[0]
dummy = np.random.RandomState(SEED).uniform(S_MIN, S_MAX, N) 

df_test = df.copy()[["index"]]
df_test["pred"] = dummy

df_test.to_csv(f"pred/random/test.csv")

df_test["pred"] = df["essay"].apply(lambda x: len(str(x)) if str(x) != "nan" else 0)

df_test.to_csv(f"pred/length/test.csv")