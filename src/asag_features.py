"""Lightweight interpretable features for ASAP-SAS answers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data_loading import PROJECT_ROOT


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "results" / "processed" / "asap-sas"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "asap-sas"

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?]+")
PUNCT_RE = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d")
UPPER_RE = re.compile(r"[A-Z]")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "more",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

FEATURE_COLUMNS = [
    "character_count",
    "word_count",
    "sentence_count",
    "average_word_length",
    "average_sentence_length",
    "unique_word_count",
    "type_token_ratio",
    "long_word_count",
    "digit_count",
    "punctuation_count",
    "uppercase_count",
    "stopword_ratio",
    "short_answer_bin",
    "medium_answer_bin",
    "long_answer_bin",
]


def load_processed_splits(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    split_names = ["train", "val", "public_test", "private_test"]
    splits: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        path = processed_dir / f"{split_name}.csv"
        if path.exists():
            splits[split_name] = pd.read_csv(path)
    if not splits:
        raise FileNotFoundError(
            f"No processed ASAP-SAS split files found under {processed_dir}. "
            "Run `python3 -m src.preprocess_asap_sas` first."
        )
    return splits


def extract_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        text = str(row.student_answer)
        words = WORD_RE.findall(text)
        word_count = len(words)
        unique_word_count = len({word.lower() for word in words})
        sentence_count = count_sentences(text, word_count)
        stopword_count = sum(1 for word in words if word.lower() in STOPWORDS)
        short_bin, medium_bin, long_bin = answer_length_bins(word_count)
        rows.append(
            {
                "sample_id": int(row.sample_id),
                "question_id": int(row.question_id),
                "split": row.split,
                "character_count": len(text),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_word_length": sum(len(word) for word in words) / word_count if word_count else 0.0,
                "average_sentence_length": word_count / sentence_count if sentence_count else 0.0,
                "unique_word_count": unique_word_count,
                "type_token_ratio": unique_word_count / word_count if word_count else 0.0,
                "long_word_count": sum(1 for word in words if len(word) >= 7),
                "digit_count": len(DIGIT_RE.findall(text)),
                "punctuation_count": len(PUNCT_RE.findall(text)),
                "uppercase_count": len(UPPER_RE.findall(text)),
                "stopword_ratio": stopword_count / word_count if word_count else 0.0,
                "short_answer_bin": short_bin,
                "medium_answer_bin": medium_bin,
                "long_answer_bin": long_bin,
            }
        )
    return pd.DataFrame(rows)


def extract_feature_splits(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {split_name: extract_feature_frame(df) for split_name, df in splits.items()}


def write_feature_splits(
    feature_splits: dict[str, pd.DataFrame],
    output_dir: Path = DEFAULT_FEATURE_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, df in feature_splits.items():
        path = output_dir / f"{split_name}_features.csv"
        df.to_csv(path, index=False)
        paths[split_name] = path
    return paths


def count_sentences(text: str, word_count: int) -> int:
    count = len(SENTENCE_RE.findall(text))
    return max(count, 1 if word_count else 0)


def answer_length_bins(word_count: int) -> tuple[int, int, int]:
    if word_count <= 20:
        return 1, 0, 0
    if word_count <= 60:
        return 0, 1, 0
    return 0, 0, 1
