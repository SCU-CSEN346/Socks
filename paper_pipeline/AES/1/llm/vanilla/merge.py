import json
PARAMS = json.load(open("../params.json", "r"))

S_MAX = PARAMS["S_MAX"]
S_MIN = PARAMS["S_MIN"]

import pandas as pd

df_0 = pd.read_csv(f"data/test_cuda_0.csv", index_col=0)
df_1 = pd.read_csv(f"data/test_cuda_1.csv", index_col=0)

df = pd.concat([df_0, df_1])

df["pred"] = df["llm"].apply(lambda x: int(str(x).lower().replace("score:", "").strip()[:2]) if "score:" in str(x).lower() and any(str(i+1) in str(x) for i in range(S_MIN-1, S_MAX)) else (S_MIN + S_MAX) // 2)
print(df["pred"].describe().to_dict())

df_test = df.copy()["essay_id pred".split()]

df_test.to_csv(f"data/test.csv")
