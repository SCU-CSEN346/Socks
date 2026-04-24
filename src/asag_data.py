"""Loading and preprocessing helpers for the ASAP-SAS ASAG baseline.

This canonical path adapts the teammate SAS helper logic from
`notebooks/read_data_sas.py` into a reusable module with stable `results/`
outputs and prompt-level score ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loading import PROJECT_ROOT


DEFAULT_ASAP_SAS_DIR = PROJECT_ROOT / "data" / "asap-sas"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "processed" / "asap-sas"
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.20

TRAIN_FILE = "train_rel_2.tsv"
PUBLIC_FILE = "public_leaderboard_rel_2.tsv"
PUBLIC_SOLUTION_FILE = "public_leaderboard_solution.csv"
PRIVATE_FILE = "private_leaderboard.tsv"
PRIVATE_META_FILE = "test.csv"

SCHEMA_COLUMNS = [
    "sample_id",
    "question_id",
    "student_answer_raw",
    "student_answer",
    "score_raw",
    "score_secondary",
    "score_min",
    "score_max",
    "score_normalized",
    "split",
    "source_dataset",
    "essay_weight",
]


def build_asap_sas_splits(
    data_dir: Path = DEFAULT_ASAP_SAS_DIR,
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create canonical ASAP-SAS train/val/public/private splits."""
    raw_train = load_train(data_dir)
    score_ranges = derive_score_ranges(raw_train)

    train_all = standardize_train(raw_train, score_ranges=score_ranges)
    train_split, val_split = split_train_val(train_all, val_size=val_size, seed=seed)

    public_test = standardize_public_test(
        load_public_with_solution(data_dir),
        score_ranges=score_ranges,
    )
    private_test = standardize_private_test(
        load_private_with_metadata(data_dir),
        score_ranges=score_ranges,
    )

    splits = {
        "train": train_split,
        "val": val_split,
        "public_test": public_test,
        "private_test": private_test,
    }
    validate_all_splits(splits)
    return splits, score_ranges


def load_train(data_dir: Path = DEFAULT_ASAP_SAS_DIR) -> pd.DataFrame:
    path = data_dir / TRAIN_FILE
    df = read_tsv(path)
    require_columns(df, ["Id", "EssaySet", "Score1", "Score2", "EssayText"], path)
    return df


def load_public_with_solution(data_dir: Path = DEFAULT_ASAP_SAS_DIR) -> pd.DataFrame:
    public_path = data_dir / PUBLIC_FILE
    solution_path = data_dir / PUBLIC_SOLUTION_FILE
    public = read_tsv(public_path)
    solution = read_csv(solution_path)
    require_columns(public, ["Id", "EssaySet", "EssayText"], public_path)
    require_columns(solution, ["id", "essay_set", "essay_weight", "essay_score"], solution_path)

    return public.merge(solution, left_on="Id", right_on="id", how="inner", validate="one_to_one")


def load_private_with_metadata(data_dir: Path = DEFAULT_ASAP_SAS_DIR) -> pd.DataFrame:
    private_path = data_dir / PRIVATE_FILE
    metadata_path = data_dir / PRIVATE_META_FILE
    private = read_tsv(private_path)
    metadata = read_csv(metadata_path)
    require_columns(private, ["Id", "EssaySet", "EssayText"], private_path)
    require_columns(metadata, ["id", "essay_set", "essay_weight"], metadata_path)

    return private.merge(metadata, left_on="Id", right_on="id", how="inner", validate="one_to_one")


def standardize_train(raw_train: pd.DataFrame, score_ranges: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": raw_train["Id"].astype(int),
            "question_id": raw_train["EssaySet"].astype(int),
            "student_answer_raw": raw_train["EssayText"],
            "score_raw": pd.to_numeric(raw_train["Score1"], errors="coerce"),
            "score_secondary": pd.to_numeric(raw_train["Score2"], errors="coerce"),
            "split": "train_source",
            "source_dataset": "asap-sas",
            "essay_weight": np.nan,
        }
    )
    return finalize_schema(df, score_ranges=score_ranges)


def standardize_public_test(raw_public: pd.DataFrame, score_ranges: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": raw_public["Id"].astype(int),
            "question_id": raw_public["EssaySet"].astype(int),
            "student_answer_raw": raw_public["EssayText"],
            "score_raw": pd.to_numeric(raw_public["essay_score"], errors="coerce"),
            "score_secondary": np.nan,
            "split": "public_test",
            "source_dataset": "asap-sas",
            "essay_weight": pd.to_numeric(raw_public["essay_weight"], errors="coerce"),
        }
    )
    return finalize_schema(df, score_ranges=score_ranges)


def standardize_private_test(raw_private: pd.DataFrame, score_ranges: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": raw_private["Id"].astype(int),
            "question_id": raw_private["EssaySet"].astype(int),
            "student_answer_raw": raw_private["EssayText"],
            "score_raw": np.nan,
            "score_secondary": np.nan,
            "split": "private_test",
            "source_dataset": "asap-sas",
            "essay_weight": pd.to_numeric(raw_private["essay_weight"], errors="coerce"),
        }
    )
    return finalize_schema(df, score_ranges=score_ranges)


def split_train_val(
    train_all: pd.DataFrame,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = choose_stratification(train_all, val_size=val_size)
    train_df, val_df = train_test_split(
        train_all,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    return (
        train_df.sort_values(["question_id", "sample_id"]).reset_index(drop=True),
        val_df.sort_values(["question_id", "sample_id"]).reset_index(drop=True),
    )


def choose_stratification(df: pd.DataFrame, val_size: float) -> Optional[pd.Series]:
    question_and_score = df["question_id"].astype(str) + "__" + df["score_raw"].astype(str)
    if can_stratify(question_and_score, val_size):
        return question_and_score
    if can_stratify(df["question_id"], val_size):
        return df["question_id"]
    return None


def can_stratify(labels: pd.Series, val_size: float) -> bool:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return False

    n_classes = len(counts)
    val_n = int(round(len(labels) * val_size))
    train_n = len(labels) - val_n
    return val_n >= n_classes and train_n >= n_classes


def derive_score_ranges(raw_train: pd.DataFrame) -> pd.DataFrame:
    score_df = pd.DataFrame(
        {
            "question_id": raw_train["EssaySet"].astype(int),
            "score_raw": pd.to_numeric(raw_train["Score1"], errors="coerce"),
        }
    ).dropna(subset=["score_raw"])
    return (
        score_df.groupby("question_id", as_index=False)
        .agg(score_min=("score_raw", "min"), score_max=("score_raw", "max"))
        .sort_values("question_id")
        .reset_index(drop=True)
    )


def finalize_schema(df: pd.DataFrame, score_ranges: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["student_answer_raw"] = df["student_answer_raw"].fillna("").astype(str)
    df["student_answer"] = normalize_whitespace(df["student_answer_raw"])
    df = df.merge(score_ranges, on="question_id", how="left", validate="many_to_one")

    denominator = df["score_max"] - df["score_min"]
    df["score_normalized"] = (df["score_raw"] - df["score_min"]) / denominator
    df.loc[df["score_raw"].isna(), "score_normalized"] = np.nan
    return df[SCHEMA_COLUMNS].sort_values(["question_id", "sample_id"]).reset_index(drop=True)


def validate_all_splits(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, df in splits.items():
        missing = [column for column in SCHEMA_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{split_name} is missing required columns: {missing}")
        if df["sample_id"].duplicated().any():
            raise ValueError(f"{split_name} contains duplicate sample_id values.")
        empty_text = df["student_answer"].fillna("").astype(str).str.strip().eq("")
        if empty_text.any():
            raise ValueError(f"{split_name} contains empty student_answer values.")


def write_processed_splits(
    splits: dict[str, pd.DataFrame],
    score_ranges: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, df in splits.items():
        path = output_dir / f"{split_name}.csv"
        df.to_csv(path, index=False)
        paths[split_name] = path

    score_ranges_path = output_dir / "score_ranges.csv"
    score_ranges.to_csv(score_ranges_path, index=False)
    paths["score_ranges"] = score_ranges_path
    return paths


def normalize_whitespace(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ASAP-SAS file: {path}")
    return pd.read_csv(path, sep="\t", encoding="latin1")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing ASAP-SAS file: {path}")
    return pd.read_csv(path, encoding="latin1")


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
