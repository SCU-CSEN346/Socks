"""Reusable loading helpers for the local ASAP-AES TSV files."""

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "asap-aes"

SPLIT_FILENAMES = {
    "train": "training_set_rel3.tsv",
    "valid": "valid_set.tsv",
    "test": "test_set.tsv",
}


def get_split_path(split: str, data_dir: Optional[Path] = None) -> Path:
    """Return the local TSV path for a known ASAP-AES split."""
    split = split.lower()
    if split not in SPLIT_FILENAMES:
        expected = ", ".join(sorted(SPLIT_FILENAMES))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {expected}.")

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return root / SPLIT_FILENAMES[split]


def load_asap_split(
    split: str,
    essay_set: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load an ASAP-AES TSV split, optionally filtered to one essay set.

    The raw files use Latin-1 compatible text and contain long essay lines, so
    this wrapper keeps the pandas call consistent across scripts and notebooks.
    """
    path = get_split_path(split, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find ASAP-AES split at {path}. "
            "Place the extracted dataset under data/asap-aes/."
        )

    df = pd.read_csv(path, sep="\t", encoding="latin1")

    if essay_set is not None:
        if "essay_set" not in df.columns:
            raise ValueError(f"Split '{split}' does not include an essay_set column.")

        df = df[df["essay_set"] == essay_set].copy()
        if df.empty:
            raise ValueError(f"No rows found for essay_set={essay_set} in split '{split}'.")

    return df.reset_index(drop=True)
