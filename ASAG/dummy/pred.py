import json
PARAMS = json.load(open("params.json", "r"))

SEED = PARAMS["SEED"]

from pathlib import Path

for directory in ["pred/", "pred/random/", "pred/length"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

import pandas as pd
import numpy as np

df = pd.read_csv(f"../data/test.csv", index_col=0).reset_index()

N = df.shape[0]
dummy = np.random.RandomState(SEED).uniform(-1, 1, N) 

df_test = df.copy()[["index"]]
df_test["pred"] = dummy

df_test.to_csv(f"pred/random/test.csv")

df_test["pred"] = df["A"].apply(lambda x: len(str(x)) if str(x) != "nan" else 0)

df_test.to_csv(f"pred/length/test.csv")