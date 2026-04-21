import json
PARAMS = json.load(open("../params.json", "r"))

S_MAX = PARAMS["S_MAX"]

import pandas as pd

df_0 = pd.read_csv("data/cuda_0.csv", index_col=0)
df_1 = pd.read_csv("data/cuda_1.csv", index_col=0)

df = pd.concat([df_0, df_1])

df["pred"] = df["llm"].apply(lambda x: int(str(x).replace("Nota:", "").strip()[0]) if "Nota:" in str(x) and any(str(i+1) in str(x) for i in range(S_MAX)) else (1 + S_MAX) // 2)

df = df.reset_index()

df_ = df.copy()["index split pred".split()]

df_.to_csv("data/train_val.csv")
