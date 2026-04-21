import json
PARAMS = json.load(open("params.json", "r"))

FRAC_TRAIN = PARAMS["DATA_FRAC_TRAIN"]
FRAC_VAL = PARAMS["DATA_FRAC_VAL"]
SEED = PARAMS["SEED"]

from pathlib import Path

for directory in ["data/", "data/ef/", "data/lf/"]:

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

import pandas as pd
import numpy as np

dataset = pd.read_csv(f"../data/train_val.csv", index_col=0).set_index("index")

data_train = dataset[dataset["split"] == "train"]
data_val = dataset[dataset["split"] == "val"]

for data_name in "train val".split():
        
    for feature_name in "lf ef".split():

        data = {"train": data_train, "val": data_val}[data_name].copy()

        if feature_name == "lf":
            o = pd.read_csv(f"/data/{data_name}.csv", index_col=0)
        else:
            o = pd.read_csv(f"../nllf_method/ef/data/{data_name}.csv", index_col=0)

        o = o.set_index("index")
        o = o.rename(columns={c: f"{data_name}_{c}" for c in o.columns})
        data = pd.concat([data, o], axis=1)

        if feature_name == "lf":
            features_names = [c for c in data.columns if "lf_" in c]
        else:
            features_names = [c for c in data.columns if "ef_traditional" in c and c != "ef_traditional<&>A.is(nan)"]

        FRAC = {"train": FRAC_TRAIN, "val": FRAC_VAL}[data_name]
        support_ix = data.sample(frac=FRAC, random_state=SEED).index

        raw_dataset = []
        for i, ix in enumerate(np.sort(support_ix)):
            row_i = data.loc[ix]
            print(f"{ix}, {100*(i+1)/(support_ix.shape[0]):.0f}%")
            for j, jx in enumerate(np.sort(support_ix)):
                row_j = data.loc[jx]
                if ix < jx:
                    o = {**{
                        "index1": ix,
                        "Q1": row_i["Q"],
                        "A1": row_i["A"],
                        "index2": jx,
                        "Q2": row_j["Q"],
                        "A2": row_j["A"]},
                        **{c: int(row_i[c] > row_j[c]) for c in features_names}
                        }
                    raw_dataset.append(o)               

        df_sample_task = pd.DataFrame(raw_dataset)
        df_sample_task.to_csv(f"data/{feature_name}/{data_name}.csv")

        print("Size:", df_sample_task.shape)