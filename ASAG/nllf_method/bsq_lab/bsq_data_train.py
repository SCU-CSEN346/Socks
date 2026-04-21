import json
PARAMS = json.load(open("../params.json", "r"))

SEED = PARAMS["SEED"]
FRAC = PARAMS["FRAC"]

import pandas as pd
from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

raw_dataset = pd.read_csv("../../data/train_val.csv", index_col=0)

print("Dataset size:", raw_dataset.shape[0])

users = raw_dataset[raw_dataset["split"] == "train"]["usuario_id"].sample(frac=1, random_state=SEED).unique()
selected_users = users[:int(FRAC * len(users))]

bsq_data_train = raw_dataset[raw_dataset["usuario_id"].isin(selected_users)]
bsq_data_train = bsq_data_train.reset_index()

bsq_data_train.to_csv("data/bsq_data_train.csv", index=0)

print("Subdataset size:", bsq_data_train.shape[0])

print("Subset generated successfully!")