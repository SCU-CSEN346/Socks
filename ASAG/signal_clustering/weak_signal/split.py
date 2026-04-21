import pandas as pd

df = pd.read_csv("pred/train_val.csv", index_col=0)
df[df["split"] == "train"].to_csv("pred/train.csv")
df[df["split"] == "test"].to_csv("pred/val.csv")