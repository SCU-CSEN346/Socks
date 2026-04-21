from pathlib import Path

directory = "labels/"

path = Path(directory)
path.mkdir(parents=True, exist_ok=True)

print("Directory labels/ created successfully!")

import pandas as pd
import os
import glob
import torch

for set in ["data"]:

    directory = f"labels/{set}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    folder = f"output/{set}"
    bsq_folders = glob.glob(f"{folder}/*")

    for bsq_folder in bsq_folders:
        
        bsq = bsq_folder.split("/")[-1]
        print(f"[{set}] [bsq: {bsq}]")

        raw_df = {"index": [], "N": [], "Y": []}

        idxs_folder = bsq_folder+"/idxs"
        logits_folder = bsq_folder+"/logits"

        idxs_filenames = glob.glob(f"{idxs_folder}/*")
        for idxs_filename in idxs_filenames:
            
            it = int(idxs_filename.split("_")[-1].replace(".pt", ""))

            logits_filename = logits_folder + f"/it_{it}.pt"

            idxs = torch.load(idxs_filename)

            logits = torch.load(logits_filename)
            probs = torch.sigmoid(logits)

            N_probs = probs[:, 0]
            Y_probs = probs[:, 1]

            raw_df["index"] += idxs.tolist()
            raw_df["N"] += N_probs.tolist()
            raw_df["Y"] += Y_probs.tolist()
            
        nllf = pd.DataFrame(raw_df).sort_values("index").set_index("index")
        nllf.to_csv(directory+f"/nllf_{bsq}.csv")