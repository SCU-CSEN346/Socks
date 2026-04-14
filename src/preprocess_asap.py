"""Minimal preprocessing CLI for ASAP-AES baseline experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loading import DEFAULT_DATA_DIR, PROJECT_ROOT, SPLIT_FILENAMES, load_asap_split


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "processed"


def normalize_essay_text(series: pd.Series) -> pd.Series:
    """Collapse repeated whitespace and trim essay text."""
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def preprocess_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return a clean dataframe suitable for simple baseline experiments."""
    if "essay" not in df.columns:
        raise ValueError("Expected an essay column in the ASAP-AES dataframe.")

    clean = df.copy()
    clean["essay"] = normalize_essay_text(clean["essay"])
    clean = clean[clean["essay"] != ""].copy()

    # Rater-level columns reveal scoring details that should not be model inputs.
    rater_columns = [col for col in clean.columns if col.startswith("rater")]
    clean = clean.drop(columns=rater_columns, errors="ignore")

    # The train split has domain1_score; valid/test do not. Keep unlabeled splits
    # unlabeled instead of assuming a score column exists everywhere.
    if "domain1_score" in clean.columns:
        clean = clean.rename(columns={"domain1_score": "score"})

    clean["split"] = split
    clean = clean.dropna(axis=1, how="all")

    return clean[_ordered_columns(clean)]


def default_output_path(
    output_dir: Path,
    split: str,
    essay_set: Optional[int],
) -> Path:
    """Build a predictable output path for the processed CSV."""
    if essay_set is None:
        filename = f"asap_{split}.csv"
    else:
        filename = f"asap_{split}_set_{essay_set}.csv"

    return output_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess an ASAP-AES TSV split for baseline experiments."
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_FILENAMES),
        default="train",
        help="Dataset split to preprocess.",
    )
    parser.add_argument(
        "--essay-set",
        type=int,
        default=None,
        help="Optional essay set number to keep, for example 1.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the extracted ASAP-AES TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the processed CSV should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = load_asap_split(
        split=args.split,
        essay_set=args.essay_set,
        data_dir=args.data_dir,
    )
    processed = preprocess_dataframe(raw, split=args.split)

    output_path = default_output_path(args.output_dir, args.split, args.essay_set)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    label_status = "labeled" if "score" in processed.columns else "unlabeled"
    print(f"Wrote {len(processed)} {label_status} rows to {output_path}")


def _ordered_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "split",
        "essay_id",
        "essay_set",
        "essay",
        "score",
        "domain2_score",
        "domain1_predictionid",
        "domain2_predictionid",
    ]
    leading = [col for col in preferred if col in df.columns]
    trailing = [col for col in df.columns if col not in leading]
    return leading + trailing


if __name__ == "__main__":
    main()
