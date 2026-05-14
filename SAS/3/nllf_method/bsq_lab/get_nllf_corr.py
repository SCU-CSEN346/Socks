import pandas as pd
import json
from pathlib import Path

Path("output/").mkdir(parents=True, exist_ok=True)

dataset_nllfg = pd.read_csv("dataset_nllfg.csv", index_col=0)
bsq_data_train = pd.read_csv("data/bsq_data_train.csv", index_col=0)
ix_to_score = bsq_data_train["Score1"].to_dict()
dataset_nllfg["score"] = dataset_nllfg["index"].apply(lambda x: ix_to_score[x])

bsq_corr = dataset_nllfg.groupby("b").apply(lambda x: x["label"].corr(x["score"])).to_dict()
ci_corr = {}
for ci, v in bsq_corr.items():
    v = v if str(v) != "nan" else 0
    for a in "YN":
        c = f"{ci} ({a})"
        ci_corr[c] = v if a == "Y" else -v

print(dataset_nllfg.groupby("b").apply(lambda x: x["bsq"].iloc[0]).to_dict())
print(bsq_corr)

with open("output/corr.json", "w") as json_file:
    json.dump(ci_corr, json_file, indent=4)

print("corr.json saved to output/corr.json")
