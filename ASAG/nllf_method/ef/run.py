from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

from feature_extractor import Preprocessing, pd

data_train_val = pd.read_csv("../../data/train_val.csv", index_col=0).reset_index()
data_train = data_train_val[data_train_val["split"] == "train"]
data_val = data_train_val[data_train_val["split"] == "val"]
data_test = pd.read_csv("../../data/test.csv", index_col=0).reset_index()


for SET_NAME in "train val test".split():

    data = {
        "train": data_train,
        "val": data_val,
        "test": data_test    
    }[SET_NAME]

    A = data.copy()

    preproc = Preprocessing()
    df_features = preproc.get_preprocessing(A)

    df_features["index"] = A["index"]

    df_features.to_csv(f"data/{SET_NAME}.csv")