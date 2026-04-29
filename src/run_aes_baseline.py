"""Canonical paper-style ASAP-AES baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aes_features import (
    DEFAULT_FEATURE_DIR,
    FEATURE_COLUMNS,
    extract_feature_splits,
    write_feature_splits,
)
from .data_loading import DEFAULT_DATA_DIR, PROJECT_ROOT, load_asap_split
from .preprocess_asap import preprocess_dataframe


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "results" / "processed" / "asap-aes"
DEFAULT_WEAK_LABEL_DIR = PROJECT_ROOT / "results" / "weak_labels" / "asap-aes"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions" / "asap-aes"
DEFAULT_METRIC_DIR = PROJECT_ROOT / "results" / "metrics" / "asap-aes"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "asap-aes"
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.20
DEFAULT_TEST_SIZE = 0.20
DEFAULT_MIN_DF = 5
DEFAULT_MAX_ITER = 100
DEFAULT_EPS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical paper-style ASAP-AES baseline from preprocessing through evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--weak-label-dir", type=Path, default=DEFAULT_WEAK_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-df", type=int, default=DEFAULT_MIN_DF)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        weak_label_dir=args.weak_label_dir,
        feature_dir=args.feature_dir,
        prediction_dir=args.prediction_dir,
        metric_dir=args.metric_dir,
        model_dir=args.model_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        min_df=args.min_df,
        max_iter=args.max_iter,
        eps=args.eps,
    )

    print("ASAP-AES baseline complete.")
    print("\nMacro metrics:")
    print(outputs["macro_metrics"].to_string(index=False))
    print("\nOutputs:")
    for name, path in outputs["paths"].items():
        print(f"- {name}: {path}")


def run_pipeline(
    data_dir: Path,
    processed_dir: Path,
    weak_label_dir: Path,
    feature_dir: Path,
    prediction_dir: Path,
    metric_dir: Path,
    model_dir: Path,
    val_size: float,
    test_size: float,
    seed: int,
    min_df: int,
    max_iter: int,
    eps: float,
) -> dict[str, object]:
    processed_splits, split_summary = build_aes_splits(
        data_dir=data_dir,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    processed_paths = write_processed_splits(processed_splits, split_summary, output_dir=processed_dir)

    weak_labels, weak_diagnostics = generate_weak_labels(
        processed_splits["train"],
        min_df=min_df,
        max_iter=max_iter,
        eps=eps,
    )
    weak_paths = write_weak_label_outputs(weak_labels, weak_diagnostics, weak_label_dir)

    feature_splits = extract_feature_splits(processed_splits)
    feature_paths = write_feature_splits(feature_splits, output_dir=feature_dir)

    predictions, metrics, coefficients = train_predict_evaluate(
        processed_splits=processed_splits,
        feature_splits=feature_splits,
        weak_labels=weak_labels,
    )
    prediction_paths = write_prediction_outputs(predictions, prediction_dir)
    metric_paths = write_metric_outputs(metrics, metric_dir)
    model_paths = write_model_outputs(coefficients, model_dir)

    paths = {
        **{f"processed_{name}": path for name, path in processed_paths.items()},
        **{f"weak_{name}": path for name, path in weak_paths.items()},
        **{f"features_{name}": path for name, path in feature_paths.items()},
        **prediction_paths,
        **metric_paths,
        **model_paths,
    }
    return {
        "macro_metrics": metrics[metrics["essay_set"] == "macro"].copy(),
        "paths": paths,
    }


def build_aes_splits(
    data_dir: Path,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    raw_train = load_asap_split("train", data_dir=data_dir)
    processed = preprocess_dataframe(raw_train, split="train")
    if "score" not in processed.columns:
        raise ValueError("Processed ASAP-AES train split is missing the gold score column.")

    processed = processed.rename(columns={"score": "gold_score"})
    processed["essay_set"] = processed["essay_set"].astype(int)
    processed["essay_id"] = processed["essay_id"].astype(int)
    processed["gold_score"] = processed["gold_score"].astype(float)

    split_frames = {"train": [], "val": [], "test": []}
    summary_rows = []
    temp_size = val_size + test_size

    for essay_set, essay_df in processed.groupby("essay_set", sort=True):
        train_df, temp_df = safe_train_test_split(
            essay_df,
            test_size=temp_size,
            seed=seed,
        )
        relative_test_size = test_size / temp_size
        val_df, test_df = safe_train_test_split(
            temp_df,
            test_size=relative_test_size,
            seed=seed + int(essay_set),
        )

        score_min = float(essay_df["gold_score"].min())
        score_max = float(essay_df["gold_score"].max())

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_copy = split_df.copy()
            split_copy["split"] = split_name
            split_copy["score_min"] = score_min
            split_copy["score_max"] = score_max
            split_frames[split_name].append(split_copy)
            summary_rows.append(
                {
                    "essay_set": int(essay_set),
                    "split": split_name,
                    "n": int(len(split_copy)),
                    "score_min": score_min,
                    "score_max": score_max,
                }
            )

    splits = {
        name: pd.concat(parts, ignore_index=True)
        .sort_values(["essay_set", "essay_id"])
        .reset_index(drop=True)
        for name, parts in split_frames.items()
    }
    split_summary = pd.DataFrame(summary_rows).sort_values(["essay_set", "split"]).reset_index(drop=True)
    return splits, split_summary


def safe_train_test_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if len(df) < 2:
        return df.copy(), df.iloc[0:0].copy()

    stratify = build_stratify_labels(df["gold_score"], test_size=test_size)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.copy(), test_df.copy()


def build_stratify_labels(scores: pd.Series, test_size: float) -> pd.Series | None:
    value_counts = scores.value_counts()
    if value_counts.empty or value_counts.min() < 2:
        return None

    n_classes = int(value_counts.shape[0])
    test_n = int(round(len(scores) * test_size))
    train_n = len(scores) - test_n
    if test_n < n_classes or train_n < n_classes:
        return None
    return scores


def write_processed_splits(
    processed_splits: dict[str, pd.DataFrame],
    split_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_df in processed_splits.items():
        path = output_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        paths[split_name] = path
    summary_path = output_dir / "split_summary.csv"
    split_summary.to_csv(summary_path, index=False)
    paths["split_summary"] = summary_path
    return paths


def generate_weak_labels(
    train_df: pd.DataFrame,
    min_df: int,
    max_iter: int,
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weak_rows = []
    diagnostic_rows = []

    for essay_set, group in train_df.groupby("essay_set", sort=True):
        texts = group["essay"].fillna("").astype(str)
        y_guess = texts.map(lambda text: len(text)).to_numpy(dtype=float)

        sim, vocabulary_size = build_jaccard_similarity(texts, min_df=min_df)
        z_values = get_z_values(y_guess, sim, max_iter=max_iter, eps=eps)

        corr = pearson_corr(y_guess, z_values)
        z_sign = np.sign(corr) if not np.isnan(corr) else 1.0
        if z_sign == 0:
            z_sign = 1.0
        z_values = z_sign * z_values
        weak_label_normalized = minmax_scale(z_values)

        weak_rows.append(
            pd.DataFrame(
                {
                    "essay_id": group["essay_id"].to_numpy(dtype=int),
                    "essay_set": int(essay_set),
                    "weak_label_raw": z_values,
                    "weak_label_normalized": weak_label_normalized,
                }
            )
        )
        diagnostic_rows.append(
            {
                "essay_set": int(essay_set),
                "n": int(len(group)),
                "vocabulary_size": int(vocabulary_size),
                "init_length_mean": float(y_guess.mean()),
                "weak_mean": float(weak_label_normalized.mean()),
                "weak_std": float(weak_label_normalized.std(ddof=0)),
                "length_weak_pearson": float(corr) if not np.isnan(corr) else None,
            }
        )

    weak_labels = pd.concat(weak_rows, ignore_index=True).sort_values(["essay_set", "essay_id"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values("essay_set").reset_index(drop=True)
    return weak_labels, diagnostics


def build_jaccard_similarity(texts: pd.Series, min_df: int) -> tuple[np.ndarray, int]:
    corpus = texts.fillna("").astype(str).map(lambda essay: f"Essay: {essay}")
    vectorizer = CountVectorizer(lowercase=True, binary=True, analyzer="word", min_df=min_df)
    bow = vectorizer.fit_transform(corpus)
    bool_bow = (bow.toarray() > 0).astype(np.uint8)
    jaccard_distance = cdist(bool_bow, bool_bow, metric="jaccard")
    sim = 1.0 - np.nan_to_num(jaccard_distance, nan=1.0)
    return sim, len(vectorizer.vocabulary_)


def get_z_values(S0: np.ndarray, sim: np.ndarray, max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
    signal = np.asarray(S0, dtype=float)
    signal_std = signal.std()
    if signal_std == 0:
        signal_std = 1.0

    z_values = np.array(
        [
            (signal[index] - np.concatenate([signal[:index], signal[index + 1 :]]).mean()) / signal_std
            for index in range(signal.shape[0])
        ],
        dtype=float,
    )

    for _ in range(max_iter):
        propagated = sim @ z_values
        propagated_std = propagated.std()
        if propagated_std == 0:
            break

        updated = np.array(
            [
                (propagated[index] - np.concatenate([propagated[:index], propagated[index + 1 :]]).mean())
                / propagated_std
                for index in range(propagated.shape[0])
            ],
            dtype=float,
        )

        corr = pearson_corr(z_values, updated)
        z_values = updated
        if not np.isnan(corr) and abs(corr) > 1 - eps:
            break
    return z_values


def write_weak_label_outputs(
    weak_labels: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weak_path = output_dir / "train_weak_labels.csv"
    diagnostics_path = output_dir / "train_weak_label_diagnostics.csv"
    weak_labels.to_csv(weak_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    return {"train": weak_path, "diagnostics": diagnostics_path}


def train_predict_evaluate(
    processed_splits: dict[str, pd.DataFrame],
    feature_splits: dict[str, pd.DataFrame],
    weak_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = feature_splits["train"].merge(
        weak_labels,
        on=["essay_id", "essay_set"],
        how="inner",
        validate="one_to_one",
    )

    prediction_frames = []
    metric_rows = []
    coefficient_rows = []

    for essay_set, train_group in train_features.groupby("essay_set", sort=True):
        model = make_pipeline(StandardScaler(), LinearRegression(positive=True))
        model.fit(train_group[FEATURE_COLUMNS], train_group["weak_label_normalized"])

        scaler = model.named_steps["standardscaler"]
        regressor = model.named_steps["linearregression"]
        coefficient_rows.extend(
            {
                "essay_set": int(essay_set),
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

        for split_name in ["val", "test"]:
            prediction_df = build_prediction_frame(
                split_name=split_name,
                essay_set=int(essay_set),
                processed_df=processed_splits[split_name],
                feature_df=feature_splits[split_name],
                model=model,
            )
            if prediction_df.empty:
                continue
            prediction_frames.append(prediction_df)
            metric_rows.append(compute_metric_row(prediction_df))

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["split", "essay_set", "essay_id"])
    metrics = pd.DataFrame(metric_rows).sort_values(["split", "essay_set"]).reset_index(drop=True)
    metrics = append_macro_rows(metrics)
    coefficients = pd.DataFrame(coefficient_rows).sort_values(["essay_set", "feature"]).reset_index(drop=True)
    return predictions, metrics, coefficients


def build_prediction_frame(
    split_name: str,
    essay_set: int,
    processed_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    split_processed = processed_df[processed_df["essay_set"] == essay_set].copy()
    if split_processed.empty:
        return pd.DataFrame()

    split_features = feature_df[feature_df["essay_set"] == essay_set].copy()
    frame = split_processed.merge(
        split_features,
        on=["essay_id", "essay_set", "split"],
        how="inner",
        validate="one_to_one",
    )
    if frame.empty:
        return frame

    weak_prediction = model.predict(frame[FEATURE_COLUMNS])
    weak_prediction_clipped = np.clip(weak_prediction, 0.0, 1.0)
    pred_score = rescale_predictions(
        weak_prediction_clipped,
        frame["score_min"].to_numpy(dtype=float),
        frame["score_max"].to_numpy(dtype=float),
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
            "essay_id",
            "essay_set",
            "gold_score",
            "score_min",
            "score_max",
            "weak_prediction",
            "weak_prediction_clipped",
            "pred_score",
            "pred_score_rounded",
        ]
    ].sort_values(["essay_set", "essay_id"])


def write_prediction_outputs(predictions: pd.DataFrame, prediction_dir: Path) -> dict[str, Path]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    combined_path = prediction_dir / "val_test_predictions.csv"
    predictions.to_csv(combined_path, index=False)
    paths["predictions_val_test"] = combined_path
    for split_name, split_df in predictions.groupby("split", sort=False):
        path = prediction_dir / f"{split_name}_predictions.csv"
        split_df.to_csv(path, index=False)
        paths[f"predictions_{split_name}"] = path
    return paths


def write_metric_outputs(metrics: pd.DataFrame, metric_dir: Path) -> dict[str, Path]:
    metric_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metric_dir / "metrics.csv"
    metrics_json_path = metric_dir / "metrics.json"
    summary_path = metric_dir / "summary_table.txt"

    metrics.to_csv(metrics_path, index=False)
    macro_rows = metrics[metrics["essay_set"] == "macro"].copy()
    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(macro_rows.to_dict(orient="records"), handle, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(metrics.round(4).to_string(index=False))
        handle.write("\n")
    return {
        "metrics_csv": metrics_path,
        "metrics_json": metrics_json_path,
        "summary_table": summary_path,
    }


def write_model_outputs(coefficients: pd.DataFrame, model_dir: Path) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    coefficient_path = model_dir / "positive_linear_coefficients.csv"
    coefficients.to_csv(coefficient_path, index=False)
    return {"model_coefficients": coefficient_path}


def compute_metric_row(prediction_df: pd.DataFrame) -> dict[str, float | int | str]:
    gold = prediction_df["gold_score"].to_numpy(dtype=float)
    pred = prediction_df["pred_score"].to_numpy(dtype=float)
    pred_rounded = prediction_df["pred_score_rounded"].to_numpy(dtype=float)
    score_min = int(np.min(prediction_df["score_min"].to_numpy(dtype=float)))
    score_max = int(np.max(prediction_df["score_max"].to_numpy(dtype=float)))

    return {
        "split": prediction_df["split"].iloc[0],
        "essay_set": int(prediction_df["essay_set"].iloc[0]),
        "n": int(len(prediction_df)),
        "qwk": quadratic_weighted_kappa(gold, pred_rounded, min_rating=score_min, max_rating=score_max),
        "mae": float(np.mean(np.abs(gold - pred))),
        "pearson": float(pearson_corr(gold, pred)),
    }


def append_macro_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    macro_rows = (
        metrics.groupby("split", as_index=False)
        .agg({"n": "sum", "qwk": "mean", "mae": "mean", "pearson": "mean"})
        .assign(essay_set="macro")
    )
    return pd.concat([metrics, macro_rows], ignore_index=True)


def rescale_predictions(
    weak_prediction: np.ndarray,
    score_min: np.ndarray,
    score_max: np.ndarray,
) -> np.ndarray:
    return score_min + weak_prediction * (score_max - score_min)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value == 0:
        return np.full_like(values, 0.5, dtype=float)
    return (values - min_value) / (max_value - min_value)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: int,
    max_rating: int,
) -> float:
    y_true = np.rint(np.asarray(y_true, dtype=float)).astype(int)
    y_pred = np.rint(np.asarray(y_pred, dtype=float)).astype(int)
    ratings = np.arange(min_rating, max_rating + 1)
    n_ratings = len(ratings)
    if n_ratings <= 1:
        return 1.0

    label_to_index = {label: index for index, label in enumerate(ratings)}
    observed = np.zeros((n_ratings, n_ratings), dtype=float)
    for truth, pred in zip(y_true, y_pred):
        observed[label_to_index[truth], label_to_index[pred]] += 1.0

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    total = observed.sum()
    if total == 0:
        return float("nan")

    expected = np.outer(hist_true, hist_pred) / total
    weights = np.zeros((n_ratings, n_ratings), dtype=float)
    denominator = float((n_ratings - 1) ** 2)
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i, j] = ((i - j) ** 2) / denominator

    observed_score = np.sum(weights * observed) / total
    expected_score = np.sum(weights * expected) / total
    if expected_score == 0:
        return 1.0
    return 1.0 - (observed_score / expected_score)


if __name__ == "__main__":
    main()
