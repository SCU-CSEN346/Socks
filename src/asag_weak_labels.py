"""Signal-clustering weak labels for the canonical ASAP-SAS baseline.

This module adapts the teammate logic from `notebooks/matrix_sas.py` and
`notebooks/pred_sas.py` into a reusable train-only weak-label generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm, rankdata

from .data_loading import PROJECT_ROOT


DEFAULT_INPUT_PATH = PROJECT_ROOT / "results" / "processed" / "asap-sas" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "weak_labels" / "asap-sas"
METHOD_NAME = "signal_clustering_jaccard_length"

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train-only ASAP-SAS weak labels with signal clustering."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels, diagnostics = generate_asag_weak_labels(
        input_path=args.input_path,
        min_df=args.min_df,
        max_iter=args.max_iter,
        eps=args.eps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_path = args.output_dir / "train_signal_clustering.csv"
    diagnostic_path = args.output_dir / "train_signal_clustering_diagnostics.csv"
    labels.to_csv(label_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)

    print("ASAP-SAS weak-label generation complete.")
    print(f"- weak labels: {label_path}")
    print(f"- diagnostics: {diagnostic_path}")


def generate_asag_weak_labels(
    input_path: Path = DEFAULT_INPUT_PATH,
    min_df: int = 5,
    max_iter: int = 100,
    eps: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_processed_train(input_path)

    label_frames = []
    diagnostic_rows = []
    for question_id, group in train_df.groupby("question_id", sort=True):
        labels, diagnostics = weak_labels_for_question(
            group=group,
            question_id=int(question_id),
            min_df=min_df,
            max_iter=max_iter,
            eps=eps,
        )
        label_frames.append(labels)
        diagnostic_rows.append(diagnostics)

    return (
        pd.concat(label_frames, ignore_index=True),
        pd.DataFrame(diagnostic_rows).sort_values("question_id").reset_index(drop=True),
    )


def load_processed_train(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing processed ASAP-SAS train file: {input_path}. "
            "Run `python3 -m src.preprocess_asap_sas` first."
        )

    df = pd.read_csv(input_path)
    required = ["sample_id", "question_id", "student_answer"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")
    return df[required].copy()


def weak_labels_for_question(
    group: pd.DataFrame,
    question_id: int,
    min_df: int,
    max_iter: int,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group = group.sort_values("sample_id").reset_index(drop=True)
    texts = group["student_answer"].fillna("").astype(str)
    
    length_signal = zscore(np.log1p(texts.map(len).to_numpy()))

    if len(group) < 3:
        labels = _create_label_df(group, question_id, length_signal, length_signal, 1.0)
        diagnostics = _generate_diag(question_id, group, labels, 1.0, "insufficient_samples")
        return labels, diagnostics

    try:
        emb = MODEL.encode(texts.tolist(), show_progress_bar=False)
        sim = cosine_similarity(emb)
        np.fill_diagonal(sim, 0.0)

        top_k_sims = np.sort(sim, axis=1)[:, -5:].mean(axis=1)
        density_signal = zscore(top_k_sims)
        
        y_guess = (0.75 * length_signal) + (0.25 * density_signal)
        z_values = get_z_values(y_guess, sim, max_iter=5, eps=eps)
        
        corr = stats.pearsonr(length_signal, z_values)[0]
        if not np.isnan(corr) and corr < 0:
            z_values = -z_values
            corr = -corr

        fallback_msg = ""

    except Exception as e:
        z_values = length_signal
        corr = 1.0
        fallback_msg = f"error_fallback: {str(e)}"

    labels = _create_label_df(group, question_id, z_values, length_signal, corr)
    diagnostics = _generate_diag(question_id, group, labels, corr, fallback_msg)
    
    return labels, diagnostics


def get_z_values(S0: np.ndarray, sim: np.ndarray, max_iter: int = 5, eps: float = 1e-5) -> np.ndarray:
    k = min(5, len(S0) - 1)
    mask = np.zeros_like(sim, dtype=bool)
    for i in range(len(sim)):
        idx = np.argpartition(sim[i], -k)[-k:]
        mask[i, idx] = True
    
    W_raw = np.where(mask, sim, 0)
    row_sums = W_raw.sum(axis=1, keepdims=True)
    W = np.divide(W_raw, row_sums, out=np.zeros_like(W_raw), where=row_sums != 0)

    z = S0.copy()
    for _ in range(max_iter):
        prev_z = z.copy()
        z = zscore((W @ z) + S0)
        
        if np.linalg.norm(z - prev_z) < eps:
            break
    z = zscore(z) 
    z = 0.8 * z 
    return z

def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(values.std())
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if low == high:
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)

def _create_label_df(group, qid, z_vals, len_sig, corr) -> pd.DataFrame:
    ranks = rankdata(z_vals)
    weak_label_normalized = (ranks - 0.5) / len(ranks)
    
    return pd.DataFrame({
        "sample_id": group["sample_id"].astype(int),
        "question_id": int(qid),
        "weak_label_raw": z_vals,
        "weak_label_normalized": weak_label_normalized,
        "method": METHOD_NAME,
        "initial_length_signal": len_sig,
        "answer_char_count": group["student_answer"].fillna("").str.len(),
        "pearson_with_length": float(corr),
    })


def _generate_diag(qid, group, labels, corr, fallback) -> dict:
    return {
        "question_id": int(qid),
        "rows": int(len(group)),
        "weak_mean": float(labels["weak_label_normalized"].mean()),
        "weak_std": float(labels["weak_label_normalized"].std(ddof=0)),
        "pearson_with_length": float(corr) if not np.isnan(corr) else 0.0,
        "vocabulary_size": 0,
        "fallback": fallback,
    }

if __name__ == "__main__":
    main()
