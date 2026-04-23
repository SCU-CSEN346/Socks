import pandas as pd
import numpy as np
import json
from scipy import stats
from pathlib import Path

def get_z_values(S0, sim, max_iter=100, eps=1e-5):
    S = S0
    S_std = S.std()
    if S_std == 0:
        S_std = 1

    Z = np.array([(S[k] - np.concatenate([S[:k], S[k+1:]]).mean()) / S_std
            for k in range(S.shape[0])])

    corr = 0
    corr_per_iter = [corr]

    for _ in range(max_iter):
        S = sim @ Z

        if S.std() == 0:
            break
        
        else:
            Z1 = np.array([(S[k] - np.concatenate([S[:k], S[k+1:]]).mean()) / S.std()
                for k in range(S.shape[0])])
                
            corr = abs(stats.pearsonr(Z, Z1)[0])
            corr_per_iter.append(corr)
            Z = Z1

            if corr > 1 - eps:
                break
    return Z

input_dir = Path("../data/aes")
output_dir = Path("../data/aes_signal")
output_dir.mkdir(parents=True, exist_ok=True)

for ESSAY_SET in range(1, 9):
    print(f"Processing essay set {ESSAY_SET}...")

    df = pd.read_csv(f"../data/essay_set_{ESSAY_SET}.csv", index_col=0)
    df = df[df["split"] == "train"].reset_index(drop=True)

    indexs = df.index.to_numpy()

    sim = np.load(f"../data/sim_matrix_{ESSAY_SET}_aes.npy", allow_pickle=True)

    y_guess = df["essay"].apply(
        lambda x: len(str(x)) if str(x) != "nan" else 0
    ).to_numpy()

    N = sim.shape[0]
    support_indexs = np.arange(N)

    df_test = df[["essay_id"]].copy()

    partition_p = support_indexs

    sim_p = sim[np.ix_(partition_p, partition_p)]
    np.fill_diagonal(sim_p, 0)

    y_guess_p = y_guess[partition_p]

    z_values_p = get_z_values(
        S0=y_guess_p,
        sim=sim_p,
        max_iter=500,
        eps=1e-6
    )

    corr = stats.pearsonr(y_guess_p, z_values_p)[0]
    z_sign = np.sign(corr)
    z_sign = z_sign if not np.isnan(z_sign) else 1

    z_values = z_sign * z_values_p

    subfolder = Path("data/signal_clustering")
    subfolder.mkdir(parents=True, exist_ok=True)
    print("Directory data/signal_clustering created successfully!")

    df_test["pred"] = z_values
    df_test.to_csv(f"../data/signal_clustering/train_{ESSAY_SET}_ws.csv", index=False)