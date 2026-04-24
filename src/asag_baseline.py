"""Train and evaluate the canonical ASAP-SAS ASAG baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .asag_features import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_PROCESSED_DIR,
    FEATURE_COLUMNS,
    extract_feature_splits,
    load_processed_splits,
    write_feature_splits,
)
from .data_loading import PROJECT_ROOT


DEFAULT_WEAK_LABEL_PATH = PROJECT_ROOT / "results" / "weak_labels" / "asap-sas" / "train_signal_clustering.csv"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions" / "asap-sas"
DEFAULT_METRIC_DIR = PROJECT_ROOT / "results" / "metrics" / "asap-sas"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "asap-sas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical ASAP-SAS ASAG baseline.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--weak-label-path", type=Path, default=DEFAULT_WEAK_LABEL_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        processed_dir=args.processed_dir,
        weak_label_path=args.weak_label_path,
        feature_dir=args.feature_dir,
        prediction_dir=args.prediction_dir,
        metric_dir=args.metric_dir,
        model_dir=args.model_dir,
    )

    print("ASAP-SAS ASAG baseline complete.")
    print("\nMacro metrics:")
    print(outputs["macro_metrics"].to_string(index=False))


def run_pipeline(
    processed_dir: Path,
    weak_label_path: Path,
    feature_dir: Path,
    prediction_dir: Path,
    metric_dir: Path,
    model_dir: Path,
) -> dict[str, object]:
    processed_splits = load_processed_splits(processed_dir)
    weak_labels = load_weak_labels(weak_label_path)
    feature_splits = extract_feature_splits(processed_splits)
    feature_paths = write_feature_splits(feature_splits, output_dir=feature_dir)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions, metrics, coefficients = train_predict_evaluate(
        processed_splits=processed_splits,
        feature_splits=feature_splits,
        weak_labels=weak_labels,
    )

    prediction_paths = write_prediction_outputs(predictions, prediction_dir)
    metric_paths = write_metric_outputs(metrics, metric_dir)
    coefficient_path = model_dir / "positive_linear_coefficients.csv"
    coefficients.to_csv(coefficient_path, index=False)

    macro_metrics = metrics[metrics["question_id"] == "macro"].copy()
    paths = {
        **{f"features_{name}": path for name, path in feature_paths.items()},
        **prediction_paths,
        **metric_paths,
        "model_coefficients": coefficient_path,
    }
    return {"macro_metrics": macro_metrics, "paths": paths}


def load_weak_labels(weak_label_path: Path) -> pd.DataFrame:
    if not weak_label_path.exists():
        raise FileNotFoundError(
            f"Missing ASAP-SAS weak-label file: {weak_label_path}. "
            "Run `python3 -m src.asag_weak_labels` first."
        )
    weak_labels = pd.read_csv(weak_label_path)
    required = ["sample_id", "question_id", "weak_label_normalized"]
    missing = [column for column in required if column not in weak_labels.columns]
    if missing:
        raise ValueError(f"Weak-label file is missing required columns: {missing}")
    return weak_labels[required].copy()


def train_predict_evaluate(
    processed_splits: dict[str, pd.DataFrame],
    feature_splits: dict[str, pd.DataFrame],
    weak_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = feature_splits["train"].merge(
        weak_labels,
        on=["sample_id", "question_id"],
        how="inner",
        validate="one_to_one",
    )

    prediction_frames = []
    metric_rows = []
    coefficient_rows = []
    prediction_splits = [name for name in ["val", "public_test", "private_test"] if name in processed_splits]

    for question_id, train_group in train_features.groupby("question_id", sort=True):
        model = make_pipeline(StandardScaler(), LinearRegression(positive=True))
        model.fit(train_group[FEATURE_COLUMNS], train_group["weak_label_normalized"])

        scaler = model.named_steps["standardscaler"]
        regressor = model.named_steps["linearregression"]
        coefficient_rows.extend(
            {
                "question_id": int(question_id),
                "feature": feature_name,
                "coefficient": float(coefficient),
                "scaler_mean": float(mean),
                "scaler_scale": float(scale),
                "intercept": float(regressor.intercept_),
            }
            for feature_name, coefficient, mean, scale in zip(
                FEATURE_COLUMNS,
                regressor.coef_,
                scaler.mean_,
                scaler.scale_,
            )
        )

        for split_name in prediction_splits:
            eval_df = build_prediction_frame(
                split_name=split_name,
                question_id=int(question_id),
                processed_df=processed_splits[split_name],
                feature_df=feature_splits[split_name],
                model=model,
            )
            if eval_df.empty:
                continue

            prediction_frames.append(eval_df)
            if split_name in {"val", "public_test"}:
                metric_rows.append(compute_metric_row(eval_df))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["split", "question_id"]).reset_index(drop=True)
    metrics = append_macro_rows(metrics)
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, metrics, coefficients


def build_prediction_frame(
    split_name: str,
    question_id: int,
    processed_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    split_processed = processed_df[processed_df["question_id"] == question_id].copy()
    if split_processed.empty:
        return pd.DataFrame()

    split_features = feature_df[feature_df["question_id"] == question_id].copy()
    frame = split_processed.merge(
        split_features,
        on=["sample_id", "question_id", "split"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        return frame

    weak_prediction = model.predict(frame[FEATURE_COLUMNS])
    weak_prediction_clipped = np.clip(weak_prediction, 0.0, 1.0)
    pred_score = rescale_predictions(
        weak_prediction_clipped,
        score_min=frame["score_min"].to_numpy(dtype=float),
        score_max=frame["score_max"].to_numpy(dtype=float),
    )
    pred_score_rounded = np.rint(pred_score).clip(
        frame["score_min"].to_numpy(dtype=float),
        frame["score_max"].to_numpy(dtype=float),
    )

    frame["weak_prediction"] = weak_prediction
    frame["weak_prediction_clipped"] = weak_prediction_clipped
    frame["pred_score"] = pred_score
    frame["pred_score_rounded"] = pred_score_rounded
    return frame[
        [
            "split",
            "sample_id",
            "question_id",
            "score_raw",
            "score_min",
            "score_max",
            "weak_prediction",
            "weak_prediction_clipped",
            "pred_score",
            "pred_score_rounded",
        ]
    ].sort_values(["question_id", "sample_id"])


def write_prediction_outputs(predictions: pd.DataFrame, prediction_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split_name, split_df in predictions.groupby("split", sort=False):
        path = prediction_dir / f"{split_name}_predictions.csv"
        split_df.to_csv(path, index=False)
        paths[f"predictions_{split_name}"] = path
    return paths


def write_metric_outputs(metrics: pd.DataFrame, metric_dir: Path) -> dict[str, Path]:
    per_prompt = metrics[metrics["question_id"] != "macro"].copy()
    macro = metrics[metrics["question_id"] == "macro"].copy()

    per_prompt_path = metric_dir / "metrics.csv"
    macro_json_path = metric_dir / "metrics.json"
    summary_path = metric_dir / "summary_table.txt"

    per_prompt.to_csv(per_prompt_path, index=False)
    with macro_json_path.open("w", encoding="utf-8") as handle:
        json.dump(macro.to_dict(orient="records"), handle, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(metrics.round(4).to_string(index=False))
        handle.write("\n")

    return {
        "metrics_csv": per_prompt_path,
        "metrics_json": macro_json_path,
        "summary_table": summary_path,
    }


def compute_metric_row(eval_df: pd.DataFrame) -> dict[str, float | int | str]:
    gold = eval_df["score_raw"].to_numpy(dtype=float)
    pred_score = eval_df["pred_score"].to_numpy(dtype=float)
    pred_score_rounded = eval_df["pred_score_rounded"].to_numpy(dtype=float)
    score_min = int(eval_df["score_min"].iloc[0])
    score_max = int(eval_df["score_max"].iloc[0])
    rounded_gold = np.rint(gold).clip(score_min, score_max).astype(int)

    return {
        "split": str(eval_df["split"].iloc[0]),
        "question_id": int(eval_df["question_id"].iloc[0]),
        "n": int(len(eval_df)),
        "qwk": quadratic_weighted_kappa(
            rounded_gold,
            pred_score_rounded.astype(int),
            min_rating=score_min,
            max_rating=score_max,
        ),
        "mae": float(np.mean(np.abs(gold - pred_score))),
        "pearson": pearson_corr(gold, pred_score),
    }


def append_macro_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    macro_rows = (
        metrics.groupby("split", as_index=False)
        .agg(n=("n", "sum"), qwk=("qwk", "mean"), mae=("mae", "mean"), pearson=("pearson", "mean"))
        .assign(question_id="macro")
    )
    return pd.concat(
        [metrics, macro_rows[["split", "question_id", "n", "qwk", "mae", "pearson"]]],
        ignore_index=True,
    )


def rescale_predictions(weak_prediction: np.ndarray, score_min: np.ndarray, score_max: np.ndarray) -> np.ndarray:
    score_min = np.asarray(score_min, dtype=float)
    score_max = np.asarray(score_max, dtype=float)
    scaled = score_min + weak_prediction * (score_max - score_min)
    return np.clip(scaled, score_min, score_max)


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: int,
    max_rating: int,
) -> float:
    ratings = np.arange(min_rating, max_rating + 1)
    n_ratings = len(ratings)
    if n_ratings <= 1:
        return 1.0

    observed = np.zeros((n_ratings, n_ratings), dtype=float)
    for actual, predicted in zip(y_true - min_rating, y_pred - min_rating):
        observed[int(actual), int(predicted)] += 1

    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / max(observed.sum(), 1.0)
    weights = np.zeros((n_ratings, n_ratings), dtype=float)
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    numerator = float((weights * observed).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - numerator / denominator


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    main()
