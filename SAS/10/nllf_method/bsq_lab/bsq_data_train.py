import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
FRAC = PARAMS["FRAC"]
ESSAY_SET = PARAMS["ESSAY_SET"]

import pandas as pd
from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

raw_dataset = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0)
raw_dataset = raw_dataset[raw_dataset["split"] == "train"]

print("Dataset size:", raw_dataset.shape[0])

bsq_data_train = raw_dataset.sample(frac=FRAC, random_state=SEED)
bsq_data_train = bsq_data_train.reset_index()

bsq_data_train.to_csv("data/bsq_data_train.csv", index=0)

print("Subdataset size:", bsq_data_train.shape[0])

print("Subset generated successfully!")
