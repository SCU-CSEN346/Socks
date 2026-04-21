from pathlib import Path

for directory in ["pred/"]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

print("Directory pred/ created successfully!")

import pandas as pd
import numpy as np
from scipy import stats

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

df = pd.read_csv(f"../../test.csv", index_col=0).reset_index()

sim = np.load(f"data/sim_matrix.npy", allow_pickle=True)

y_guess = pd.read_csv(f"../../dummy/pred/length/test.csv", index_col=0)["pred"].values

N = sim.shape[0]
support_indexs = np.array(range(N))

df_test = df.copy()[["index"]]

partition_p = support_indexs

sim_p = sim[partition_p, :][:, partition_p]

sim_p[range(sim_p.shape[0]), range(sim_p.shape[1])] = 0

y_guess_p = y_guess[partition_p]

z_values_p = get_z_values(S0=y_guess_p, sim=sim_p, max_iter=500, eps=1e-6)

z_sign = np.sign(stats.pearsonr(y_guess_p, z_values_p)[0])
z_sign = z_sign if str(z_sign) != "nan" else 1

z_values = z_sign*z_values_p
df_test["pred"] = z_values
        
df_test.to_csv(f"pred/test.csv")