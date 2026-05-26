import json
import re
PARAMS = json.load(open("../params.json", "r"))

S_MAX = PARAMS["S_MAX"]
S_MIN = PARAMS["S_MIN"]

import pandas as pd

df_0 = pd.read_csv(f"data/train_cuda_0.csv", index_col=0)
df_1 = pd.read_csv(f"data/train_cuda_1.csv", index_col=0)

df = pd.concat([df_0, df_1])

def parse_score(x, s_min, s_max):
    s = str(x).lower()
    if "score:" in s:
        after = s.split("score:")[-1]
        m = re.search(r"\d+", after)
        if m:
            val = int(m.group())
            if s_min <= val <= s_max:
                return val
    return (s_min + s_max) // 2

df["pred"] = df["llm"].apply(lambda x: parse_score(x, S_MIN, S_MAX))
print(df["pred"].describe().to_dict())

df_test = df.copy()["Id pred".split()]

df_test.to_csv(f"data/train.csv")
