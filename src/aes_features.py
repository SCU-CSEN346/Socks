"""Interpretable feature extraction for the canonical ASAP-AES baseline."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data_loading import PROJECT_ROOT


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "results" / "processed" / "asap-aes"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "asap-aes"

WORD_RE = re.compile(r"\b\w+\b")
SENTENCE_RE = re.compile(r"[.!?]+")
PUNCT_RE = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d")

FEATURE_COLUMNS = [
    "word_count",
    "character_count",
    "sentence_count",
    "average_word_length",
    "unique_word_count",
    "type_token_ratio",
    "long_word_count",
    "punctuation_count",
    "digit_count",
    "paragraph_count",
]


def load_processed_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        path = processed_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed ASAP-AES split: {path}. "
                "Run `python3 -m src.run_aes_baseline` to regenerate it."
            )
        splits[split_name] = pd.read_csv(path)
    return splits


def extract_feature_splits(processed_splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        split_name: extract_feature_frame(split_df)
        for split_name, split_df in processed_splits.items()
    }


def extract_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        text = str(row.essay or "")
        tokens = WORD_RE.findall(text)
        lowered_tokens = [token.lower() for token in tokens]
        word_count = len(tokens)
        sentence_count = len(SENTENCE_RE.findall(text))
        if text.strip() and sentence_count == 0:
            sentence_count = 1

        unique_word_count = len(set(lowered_tokens))
        average_word_length = (
            sum(len(token) for token in tokens) / word_count if word_count else 0.0
        )
        paragraph_count = len([chunk for chunk in text.splitlines() if chunk.strip()])
        if text.strip() and paragraph_count == 0:
            paragraph_count = 1

        rows.append(
            {
                "essay_id": int(row.essay_id),
                "essay_set": int(row.essay_set),
                "split": row.split,
                "word_count": word_count,
                "character_count": len(text),
                "sentence_count": sentence_count,
                "average_word_length": average_word_length,
                "unique_word_count": unique_word_count,
                "type_token_ratio": (unique_word_count / word_count) if word_count else 0.0,
                "long_word_count": sum(1 for token in tokens if len(token) >= 7),
                "punctuation_count": len(PUNCT_RE.findall(text)),
                "digit_count": len(DIGIT_RE.findall(text)),
                "paragraph_count": paragraph_count,
            }
        )

    feature_df = pd.DataFrame(rows)
    return feature_df[
        ["essay_id", "essay_set", "split", *FEATURE_COLUMNS]
    ].sort_values(["essay_set", "essay_id"]).reset_index(drop=True)


def write_feature_splits(
    feature_splits: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_FEATURE_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, feature_df in feature_splits.items():
        path = output_dir / f"{split_name}_features.csv"
        feature_df.to_csv(path, index=False)
        paths[split_name] = path
    return paths
