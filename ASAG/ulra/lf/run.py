from pathlib import Path

directory = "data/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory data/ created successfully!")

from feature_extractor import ULRAFeatureExtractor, pd


for SET_NAME in "train val test".split():

    data = pd.read_csv(f"../../data/{SET_NAME}.csv", index_col=0)

    A = data.copy().reset_index()

    preproc = ULRAFeatureExtractor()
    df_features = preproc.get_preprocessing(A)

    df_features["index"] = A["index"]

    df_features.to_csv(f"data/{SET_NAME}.csv")