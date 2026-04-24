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
from sklearn.feature_extraction.text import CountVectorizer

from .data_loading import PROJECT_ROOT


DEFAULT_INPUT_PATH = PROJECT_ROOT / "results" / "processed" / "asap-sas" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "weak_labels" / "asap-sas"
METHOD_NAME = "signal_clustering_jaccard_length"


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
    y_guess = texts.map(lambda text: len(str(text)) if str(text) != "nan" else 0).to_numpy(dtype=float)

    fallback = ""
    if len(group) < 2:
        z_values = np.zeros(len(group), dtype=float)
        corr = np.nan
        vocabulary_size = 0
        fallback = "single_row"
    else:
        try:
            sim, vocabulary_size = build_jaccard_similarity(texts, min_df=min_df)
            np.fill_diagonal(sim, 0.0)
            z_values = get_z_values(y_guess, sim, max_iter=max_iter, eps=eps)
            corr = stats.pearsonr(y_guess, z_values)[0]
            z_sign = np.sign(corr) if not np.isnan(corr) else 1.0
            z_values = z_sign * z_values
        except ValueError:
            z_values = zscore(np.log1p(y_guess))
            corr = np.nan
            vocabulary_size = 0
            fallback = "length_only"

    weak_label_normalized = minmax_scale(z_values)
    labels = pd.DataFrame(
        {
            "sample_id": group["sample_id"].astype(int),
            "question_id": int(question_id),
            "weak_label_raw": z_values,
            "weak_label_normalized": weak_label_normalized,
            "method": METHOD_NAME,
            "initial_length_signal": zscore(np.log1p(y_guess)),
            "answer_char_count": y_guess.astype(int),
            "fallback": fallback,
        }
    )
    diagnostics = {
        "question_id": int(question_id),
        "rows": int(len(group)),
        "weak_mean": float(weak_label_normalized.mean()),
        "weak_std": float(weak_label_normalized.std(ddof=0)),
        "pearson_with_length": float(corr) if not np.isnan(corr) else np.nan,
        "vocabulary_size": int(vocabulary_size),
        "fallback": fallback,
    }
    return labels, diagnostics


def build_jaccard_similarity(texts: pd.Series, min_df: int) -> tuple[np.ndarray, int]:
    vectorizer = CountVectorizer(lowercase=True, binary=True, min_df=min_df)
    bow = vectorizer.fit_transform(texts)
    if bow.shape[1] == 0:
        raise ValueError("Empty vocabulary for this question.")
    bool_bow = bow.toarray().astype(np.uint8)
    jaccard_distance = cdist(bool_bow, bool_bow, metric="jaccard")
    return 1.0 - jaccard_distance, len(vectorizer.vocabulary_)


def get_z_values(S0: np.ndarray, sim: np.ndarray, max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
    S = S0.astype(float)
    S_std = float(S.std()) if float(S.std()) != 0 else 1.0
    Z = np.array(
        [
            (S[k] - np.concatenate([S[:k], S[k + 1 :]]).mean()) / S_std
            for k in range(S.shape[0])
        ]
    )

    for _ in range(max_iter):
        S = sim @ Z
        S_std = float(S.std())
        if S_std == 0:
            break

        Z1 = np.array(
            [
                (S[k] - np.concatenate([S[:k], S[k + 1 :]]).mean()) / S_std
                for k in range(S.shape[0])
            ]
        )

        corr = abs(stats.pearsonr(Z, Z1)[0])
        Z = Z1
        if corr > 1.0 - eps:
            break
    return Z


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


if __name__ == "__main__":
    main()
