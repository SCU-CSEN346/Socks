"""AES weak-signal helpers preserved from teammate signal clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT_DIR = Path("../data")
DEFAULT_SIGNAL_DIR = Path("../data/signal_clustering")


def get_z_values(S0, sim, max_iter=100, eps=1e-5):
    S = np.asarray(S0, dtype=float)
    S_std = S.std()
    if S_std == 0:
        S_std = 1

    Z = np.array(
        [
            (S[k] - np.concatenate([S[:k], S[k + 1 :]]).mean()) / S_std
            for k in range(S.shape[0])
        ]
    )

    corr = 0
    corr_per_iter = [corr]

    for _ in range(max_iter):
        S = sim @ Z

        if S.std() == 0:
            break

        else:
            Z1 = np.array(
                [
                    (S[k] - np.concatenate([S[:k], S[k + 1 :]]).mean()) / S.std()
                    for k in range(S.shape[0])
                ]
            )

            corr = abs(stats.pearsonr(Z, Z1)[0])
            corr_per_iter.append(corr)
            Z = Z1

            if corr > 1 - eps:
                break
    return Z


def generate_length_based_weak_signal(
    essay_df: pd.DataFrame,
    sim: np.ndarray,
    max_iter: int = 500,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    if "essay" not in essay_df.columns:
        raise ValueError("Expected an essay column to generate the AES weak signal.")

    y_guess = essay_df["essay"].apply(
        lambda x: len(str(x)) if str(x) != "nan" else 0
    ).to_numpy(dtype=float)

    sim_local = np.asarray(sim, dtype=float).copy()
    np.fill_diagonal(sim_local, 0)

    z_values = get_z_values(
        S0=y_guess,
        sim=sim_local,
        max_iter=max_iter,
        eps=eps,
    )

    corr = stats.pearsonr(y_guess, z_values)[0]
    z_sign = np.sign(corr)
    z_sign = z_sign if not np.isnan(z_sign) and z_sign != 0 else 1
    return z_sign * z_values, corr


def main() -> None:
    output_dir = DEFAULT_SIGNAL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for essay_set in range(1, 9):
        print(f"Processing essay set {essay_set}...")

        df = pd.read_csv(DEFAULT_INPUT_DIR / f"essay_set_{essay_set}.csv", index_col=0)
        df = df[df["split"] == "train"].reset_index(drop=True)

        sim = np.load(DEFAULT_INPUT_DIR / f"sim_matrix_{essay_set}_aes.npy", allow_pickle=True)
        z_values, _ = generate_length_based_weak_signal(
            df,
            sim=sim,
            max_iter=500,
            eps=1e-6,
        )

        df_test = df[["essay_id"]].copy()
        df_test["pred"] = z_values
        df_test.to_csv(output_dir / f"train_{essay_set}_ws.csv", index=False)


if __name__ == "__main__":
    main()
