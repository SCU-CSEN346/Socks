import pandas as pd
import os
import glob

import json
PARAMS = json.load(open("../params.json", "r"))

ESSAY_SET = PARAMS["ESSAY_SET"]

data = pd.read_csv(f"../../../data/essay_set_{ESSAY_SET}.csv", index_col=0).reset_index()

for set, data in {
    "data": data[["index", "split", "Id"]]
    }.items():

    bsq_folders = glob.glob(f"labels/{set}/*")

    for bsq_filename in bsq_folders:

        bsq = bsq_filename.split("_")[-1].replace(".csv","")

        df_bsq = pd.read_csv(bsq_filename, index_col=0)
        df_bsq = df_bsq.rename(columns={"N": f"{bsq} (N)", "Y": f"{bsq} (Y)"})
        df_bsq = df_bsq.loc[data["index"].values].reset_index().drop(columns="index")

        data = pd.concat([data, df_bsq], axis=1)

    data.to_csv(f"labels/{set}_nllf.csv")
