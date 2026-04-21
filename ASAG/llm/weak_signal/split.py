import pandas as pd

df = pd.read_csv("data/train_val.csv", index_col=0)
df[df["split"] == "train"].to_csv("data/train.csv")
df[df["split"] == "val"].to_csv("data/val.csv")