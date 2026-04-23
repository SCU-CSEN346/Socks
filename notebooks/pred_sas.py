import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def get_z_values(S0, sim, max_iter=100, eps=1e-5):
    S = S0
    S_std = S.std() if S.std() != 0 else 1
    
    Z = np.array([(S[k] - np.concatenate([S[:k], S[k+1:]]).mean()) / S_std
            for k in range(S.shape[0])])

    for _ in range(max_iter):
        S = sim @ Z 
        if S.std() == 0: break
        
        Z1 = np.array([(S[k] - np.concatenate([S[:k], S[k+1:]]).mean()) / S.std()
                for k in range(S.shape[0])])
                
        corr = abs(stats.pearsonr(Z, Z1)[0])
        Z = Z1
        if corr > 1 - eps: break
    return Z


input_dir = Path("../data/sas")
output_dir = Path("../data/sas_signal")
output_dir.mkdir(parents=True, exist_ok=True)

for ESSAY_SET in range(1, 11):
    data_path = input_dir/f"essay_set_{ESSAY_SET}_sas.csv"
    sim_path = output_dir/f"sim_matrix_{ESSAY_SET}_sas.npy"
    
    print(f"Processing Essay Set {ESSAY_SET}...")

    df_full = pd.read_csv(data_path)
    sim = np.load(sim_path)

    df = df_full[df_full['split'] == 'train'].copy().reset_index(drop=True)
    #issue where df and matrix have different dimension, currently just a quick fix, to be resolved later
    N_df = len(df)
    N_sim = sim.shape[0]
    np.fill_diagonal(sim, 0)
    
    y_guess = df["essaytext"].apply(lambda x: len(str(x)) if str(x) != "nan" else 0).values

    z_values_refined = get_z_values(S0=y_guess, sim=sim)

    z_sign = np.sign(stats.pearsonr(y_guess, z_values_refined)[0])
    if np.isnan(z_sign): z_sign = 1
    
    df_results = df[["id", "split"]].copy()
    df_results["pred"] = z_sign * z_values_refined
    
    save_path = output_dir/f"weak_signal_{ESSAY_SET}_sas.csv"
    df_results.to_csv(save_path, index=False)
    print(f"Successfully saved results to {save_path}")