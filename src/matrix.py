"""AES similarity-matrix helpers preserved from teammate signal clustering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_INPUT_DIR = Path("../data")
DEFAULT_OUTPUT_DIR = Path("../data/aes_signal")


def build_jaccard_similarity_matrix(
    essays: pd.Series,
    min_df: int = 5,
) -> tuple[np.ndarray, int]:
    corpus = essays.fillna("").astype(str).map(lambda essay: f"Essay: {essay}")
    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        analyzer="word",
        min_df=min_df,
        strip_accents=None,
    )
    bow = vectorizer.fit_transform(corpus)
    bool_bow = (bow.toarray() > 0).astype(np.uint8)
    jaccard_distance = cdist(bool_bow, bool_bow, metric="jaccard")
    jaccard_sim = 1.0 - np.nan_to_num(jaccard_distance, nan=1.0)
    return jaccard_sim, len(vectorizer.vocabulary_)


def build_similarity_for_essay_set(
    essay_df: pd.DataFrame,
    min_df: int = 5,
) -> tuple[np.ndarray, int]:
    if "essay" not in essay_df.columns:
        raise ValueError("Expected an essay column to build the AES similarity matrix.")
    return build_jaccard_similarity_matrix(essay_df["essay"], min_df=min_df)


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for essay_set in range(1, 9):
        print(f"Processing essay set {essay_set}...")

        df = pd.read_csv(DEFAULT_INPUT_DIR / f"essay_set_{essay_set}.csv", index_col=0).reset_index()
        df = df[df["split"] == "train"]

        similarity, _ = build_similarity_for_essay_set(df, min_df=5)
        np.save(output_dir / f"sim_matrix_{essay_set}_aes.npy", similarity)
        print(f"Saved sim_matrix_{essay_set}.npy")


if __name__ == "__main__":
    main()
