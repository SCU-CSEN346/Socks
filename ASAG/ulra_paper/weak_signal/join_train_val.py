import pandas as pd

train = pd.read_csv("data/train.csv", index_col=0)
val = pd.read_csv("data/val.csv", index_col=0)

pd.concat([train, val], ignore_index=True).to_csv("data/train_val.csv")