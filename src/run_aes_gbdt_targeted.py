"""Targeted AES GBDT experiment driven by validation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aes_features import FEATURE_COLUMNS
from .run_aes_baseline import append_macro_rows, build_prediction_frame, compute_metric_row


DEFAULT_CLEAN_DIR = Path("results/clean_results/aes_gbdt_targeted")
GBDT_CANDIDATES = ["hist_gbdt_all_sets", "sklearn_gbdt_all_sets"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted AES GBDT selection experiments.")
    parser.add_argument("--processed-dir", type=Path, default=Path("results/processed/asap-aes"))
    parser.add_argument("--feature-dir", type=Path, default=Path("results/features/asap-aes"))
    parser.add_argument("--weak-label-file", type=Path, default=Path("results/weak_labels/asap-aes/train_weak_labels.csv"))
    parser.add_argument("--baseline-metrics-json", type=Path, default=Path("results/metrics/asap-aes/metrics.json"))
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.clean_dir.mkdir(parents=True, exist_ok=True)
    args.clean_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)

    processed = load_processed_splits(args.processed_dir)
    features = load_feature_splits(args.feature_dir)
    weak = pd.read_csv(args.weak_label_file)
    baseline_reference = pd.DataFrame(json.loads(args.baseline_metrics_json.read_text()))
    baseline_reference.to_csv(args.clean_dir / "baseline_metrics.csv", index=False)

    trained = {
        "baseline_positive_linear": train_models("positive_linear", features, weak, args.seed),
        "ridge_alpha_10": train_models("ridge", features, weak, args.seed),
        "hist_gbdt_all_sets": train_models("hist_gbdt", features, weak, args.seed),
        "sklearn_gbdt_all_sets": train_models("sklearn_gbdt", features, weak, args.seed),
    }

    predictions_by_variant = {}
    metrics_by_variant = {}
    for variant_name, models in trained.items():
        predictions, metrics = evaluate_models(variant_name, models, processed, features)
        predictions_by_variant[variant_name] = predictions
        metrics_by_variant[variant_name] = metrics

    best_gbdt_by_set = choose_best_model_by_set(metrics_by_variant, GBDT_CANDIDATES)
    best_gbdt_predictions = mix_predictions(
        "best_gbdt_by_set",
        predictions_by_variant,
        {essay_set: best_gbdt_by_set[essay_set] for essay_set in best_gbdt_by_set},
        default_variant="baseline_positive_linear",
    )
    calibrated_best_gbdt_predictions = calibrate_selected_sets(
        best_gbdt_predictions,
        weak,
        selected_sets=set(best_gbdt_by_set),
        variant_name="best_gbdt_by_set_plus_shrinked_calibration",
        lambda_value=0.25,
    )
    metrics_by_variant["best_gbdt_by_set"] = metrics_from_predictions("best_gbdt_by_set", best_gbdt_predictions)
    metrics_by_variant["best_gbdt_by_set_plus_shrinked_calibration"] = metrics_from_predictions(
        "best_gbdt_by_set_plus_shrinked_calibration",
        calibrated_best_gbdt_predictions,
    )

    diagnostics = build_per_set_diagnostics(metrics_by_variant, best_gbdt_by_set)
    selected_sets = select_sets(diagnostics)

    candidate_predictions = {
        "baseline_positive_linear": predictions_by_variant["baseline_positive_linear"],
        "gbdt_selected_qwk_only": build_targeted_prediction(
            "gbdt_selected_qwk_only", predictions_by_variant, best_gbdt_by_set, selected_sets["qwk_only_selection"]
        ),
        "gbdt_selected_balanced": build_targeted_prediction(
            "gbdt_selected_balanced", predictions_by_variant, best_gbdt_by_set, selected_sets["balanced_selection"]
        ),
        "gbdt_selected_conservative": build_targeted_prediction(
            "gbdt_selected_conservative", predictions_by_variant, best_gbdt_by_set, selected_sets["conservative_selection"]
        ),
        "gbdt_manual_safe_group": build_targeted_prediction(
            "gbdt_manual_safe_group", predictions_by_variant, best_gbdt_by_set, selected_sets["manual_safe_group"]
        ),
        "set6_gbdt_only": build_targeted_prediction(
            "set6_gbdt_only", predictions_by_variant, best_gbdt_by_set, {6}
        ),
    }

    calibration_sets = select_calibration_sets(diagnostics, selected_sets["balanced_selection"])
    balanced_base = candidate_predictions["gbdt_selected_balanced"]
    candidate_predictions["gbdt_selected_balanced_plus_shrinked_calibration"] = calibrate_selected_sets(
        balanced_base,
        weak,
        selected_sets=calibration_sets,
        variant_name="gbdt_selected_balanced_plus_shrinked_calibration",
        lambda_value=0.25,
    )
    candidate_predictions["set6_gbdt_plus_shrinked_calibration"] = calibrate_selected_sets(
        candidate_predictions["set6_gbdt_only"],
        weak,
        selected_sets={6},
        variant_name="set6_gbdt_plus_shrinked_calibration",
        lambda_value=0.25,
    )
    manual_calibration_sets = set(
        diagnostics.loc[
            diagnostics["essay_set"].isin(selected_sets["manual_safe_group"])
            & diagnostics["recommendation"].eq("use_gbdt_plus_calibration"),
            "essay_set",
        ].astype(int)
    )
    candidate_predictions["gbdt_manual_safe_group_plus_recommended_calibration"] = calibrate_selected_sets(
        candidate_predictions["gbdt_manual_safe_group"],
        weak,
        selected_sets=manual_calibration_sets,
        variant_name="gbdt_manual_safe_group_plus_recommended_calibration",
        lambda_value=0.25,
    )
    candidate_predictions["set6_gbdt_set8_baseline_or_calibration"] = build_set6_plus_optional_set8_calibration(
        predictions_by_variant,
        best_gbdt_by_set,
        weak,
        diagnostics,
    )
    val_selected_models = choose_best_model_by_set(
        metrics_by_variant,
        ["baseline_positive_linear", "ridge_alpha_10", "hist_gbdt_all_sets", "sklearn_gbdt_all_sets"],
    )
    candidate_predictions["val_selected_per_essay_set_model"] = mix_predictions(
        "val_selected_per_essay_set_model",
        predictions_by_variant,
        val_selected_models,
        default_variant="baseline_positive_linear",
    )

    candidate_metrics = pd.concat(
        [metrics_from_predictions(name, frame)[metrics_from_predictions(name, frame)["essay_set"].astype(str).eq("macro")]
         for name, frame in candidate_predictions.items()],
        ignore_index=True,
    )
    per_set_metrics = pd.concat(
        [metrics_from_predictions(name, frame)[metrics_from_predictions(name, frame)["essay_set"].astype(str).ne("macro")]
         for name, frame in candidate_predictions.items()],
        ignore_index=True,
    )
    candidate_metrics = add_deltas(candidate_metrics)
    per_set_delta = add_per_set_deltas(per_set_metrics)

    selected_sets_df = selected_sets_to_frame(
        selected_sets,
        best_gbdt_by_set,
        calibration_sets,
        manual_calibration_sets,
        val_selected_models,
    )
    feature_importance = compute_feature_importances(trained, processed, features, args.seed)
    mistral_subset = build_mistral_subset_note()

    diagnostics.to_csv(args.clean_dir / "per_set_gbdt_diagnostics.csv", index=False)
    selected_sets_df.to_csv(args.clean_dir / "selected_sets_by_rule.csv", index=False)
    candidate_metrics.to_csv(args.clean_dir / "targeted_candidate_metrics.csv", index=False)
    per_set_metrics.to_csv(args.clean_dir / "per_essay_set_metrics.csv", index=False)
    per_set_delta.to_csv(args.clean_dir / "per_essay_set_delta.csv", index=False)
    feature_importance.to_csv(args.clean_dir / "feature_importance_by_set.csv", index=False)
    mistral_subset.to_csv(args.clean_dir / "mistral_subset_ablation.csv", index=False)
    write_summary(args.clean_dir, baseline_reference, candidate_metrics, diagnostics, selected_sets_df, per_set_delta, feature_importance, mistral_subset)
    write_readme(args.clean_dir)
    write_graph(args.clean_dir, candidate_metrics)

    print("AES targeted GBDT experiment complete.")
    print(candidate_metrics.sort_values(["split", "qwk"], ascending=[True, False]).to_string(index=False))


def load_processed_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(processed_dir / f"{split}.csv") for split in ["train", "val", "test"]}


def load_feature_splits(feature_dir: Path) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(feature_dir / f"{split}_features.csv") for split in ["train", "val", "test"]}


def train_models(model_type: str, features: dict[str, pd.DataFrame], weak: pd.DataFrame, seed: int) -> dict[int, object]:
    train_features = features["train"].merge(weak, on=["essay_id", "essay_set"], how="inner", validate="one_to_one")
    models = {}
    for essay_set, group in train_features.groupby("essay_set", sort=True):
        model = make_model(model_type, seed)
        model.fit(group[FEATURE_COLUMNS], group["weak_label_normalized"])
        models[int(essay_set)] = model
    return models


def make_model(model_type: str, seed: int):
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if model_type == "hist_gbdt":
        return HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=seed,
        )
    if model_type == "sklearn_gbdt":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )
    return make_pipeline(StandardScaler(), LinearRegression(positive=True))


def evaluate_models(
    variant_name: str,
    models: dict[int, object],
    processed: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    for essay_set, model in sorted(models.items()):
        for split in ["val", "test"]:
            frame = build_prediction_frame(split, essay_set, processed[split], features[split], model)
            if not frame.empty:
                frame.insert(0, "source_variant", variant_name)
                prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return predictions, metrics_from_predictions(variant_name, predictions)


def metrics_from_predictions(variant_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(["split", "essay_set"], sort=True):
        rows.append(compute_metric_row(group))
    metrics = pd.DataFrame(rows).sort_values(["split", "essay_set"]).reset_index(drop=True)
    metrics = append_macro_rows(metrics)
    metrics.insert(0, "variant_name", variant_name)
    return metrics


def choose_best_model_by_set(metrics_by_variant: dict[str, pd.DataFrame], candidates: list[str]) -> dict[int, str]:
    frames = []
    for name in candidates:
        frame = metrics_by_variant[name]
        frames.append(frame[frame["split"].eq("val") & frame["essay_set"].astype(str).ne("macro")].copy())
    val = pd.concat(frames, ignore_index=True)
    choices = {}
    for essay_set, group in val.groupby("essay_set", sort=True):
        choices[int(essay_set)] = str(group.sort_values(["qwk", "mae"], ascending=[False, True]).iloc[0]["variant_name"])
    return choices


def mix_predictions(
    variant_name: str,
    predictions_by_variant: dict[str, pd.DataFrame],
    choices: dict[int, str],
    default_variant: str,
) -> pd.DataFrame:
    pieces = []
    essay_sets = sorted(predictions_by_variant[default_variant]["essay_set"].astype(int).unique())
    for essay_set in essay_sets:
        source = choices.get(int(essay_set), default_variant)
        group = predictions_by_variant[source][predictions_by_variant[source]["essay_set"].astype(int).eq(int(essay_set))].copy()
        group["source_variant"] = source
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True)
    out["variant_name"] = variant_name
    return out.sort_values(["split", "essay_set", "essay_id"]).reset_index(drop=True)


def build_targeted_prediction(
    variant_name: str,
    predictions_by_variant: dict[str, pd.DataFrame],
    best_gbdt_by_set: dict[int, str],
    selected_sets: set[int],
) -> pd.DataFrame:
    choices = {essay_set: best_gbdt_by_set[essay_set] for essay_set in selected_sets if essay_set in best_gbdt_by_set}
    return mix_predictions(variant_name, predictions_by_variant, choices, default_variant="baseline_positive_linear")


def build_per_set_diagnostics(metrics_by_variant: dict[str, pd.DataFrame], best_gbdt_by_set: dict[int, str]) -> pd.DataFrame:
    rows = []
    baseline = metrics_by_variant["baseline_positive_linear"]
    calibrated = metrics_by_variant["best_gbdt_by_set_plus_shrinked_calibration"]
    for essay_set in sorted(best_gbdt_by_set):
        gbdt_name = best_gbdt_by_set[essay_set]
        gbdt = metrics_by_variant[gbdt_name]
        row = {
            "essay_set": essay_set,
            "best_gbdt_variant": gbdt_name,
        }
        for split in ["val", "test"]:
            base_row = one_metric(baseline, split, essay_set)
            gbdt_row = one_metric(gbdt, split, essay_set)
            cal_row = one_metric(calibrated, split, essay_set)
            for metric in ["qwk", "mae", "pearson"]:
                row[f"baseline_{split}_{metric}"] = float(base_row[metric])
                row[f"gbdt_{split}_{metric}"] = float(gbdt_row[metric])
                row[f"gbdt_calibrated_{split}_{metric}"] = float(cal_row[metric])
            row[f"gbdt_{split}_qwk_delta"] = row[f"gbdt_{split}_qwk"] - row[f"baseline_{split}_qwk"]
            row[f"gbdt_{split}_mae_delta"] = row[f"gbdt_{split}_mae"] - row[f"baseline_{split}_mae"]
            row[f"gbdt_{split}_pearson_delta"] = row[f"gbdt_{split}_pearson"] - row[f"baseline_{split}_pearson"]
            row[f"gbdt_calibrated_{split}_qwk_delta"] = row[f"gbdt_calibrated_{split}_qwk"] - row[f"baseline_{split}_qwk"]
            row[f"gbdt_calibrated_{split}_mae_delta"] = row[f"gbdt_calibrated_{split}_mae"] - row[f"baseline_{split}_mae"]
            row[f"gbdt_calibrated_{split}_pearson_delta"] = row[f"gbdt_calibrated_{split}_pearson"] - row[f"baseline_{split}_pearson"]
        row["recommendation"] = recommend_set(row)
        rows.append(row)
    return pd.DataFrame(rows)


def one_metric(metrics: pd.DataFrame, split: str, essay_set: int) -> pd.Series:
    per_set = metrics[metrics["essay_set"].astype(str).ne("macro")].copy()
    return per_set[per_set["split"].eq(split) & per_set["essay_set"].astype(int).eq(essay_set)].iloc[0]


def recommend_set(row: dict) -> str:
    if row["gbdt_calibrated_val_qwk_delta"] >= 0.005 and row["gbdt_calibrated_val_mae_delta"] <= 0.05:
        return "use_gbdt_plus_calibration"
    if row["gbdt_val_qwk_delta"] >= 0.005 and row["gbdt_val_mae_delta"] <= 0.05 and row["gbdt_val_pearson_delta"] >= -0.02:
        return "use_gbdt"
    if row["gbdt_val_mae_delta"] > 0.25 or row["gbdt_calibrated_val_mae_delta"] > 0.25:
        return "avoid"
    return "use_baseline"


def select_sets(diagnostics: pd.DataFrame) -> dict[str, set[int]]:
    qwk_only = set(diagnostics.loc[diagnostics["gbdt_val_qwk_delta"] >= 0.005, "essay_set"].astype(int))
    balanced = set(
        diagnostics.loc[
            (diagnostics["gbdt_val_qwk_delta"] >= 0.005)
            & (diagnostics["gbdt_val_mae_delta"] <= 0.05)
            & (diagnostics["gbdt_val_pearson_delta"] >= -0.02),
            "essay_set",
        ].astype(int)
    )
    conservative = set(
        diagnostics.loc[
            (diagnostics["gbdt_val_qwk_delta"] >= 0.01)
            & (diagnostics["gbdt_val_mae_delta"] <= 0.01)
            & (diagnostics["gbdt_val_pearson_delta"] >= 0.0),
            "essay_set",
        ].astype(int)
    )
    bad_mae = set(diagnostics.loc[diagnostics["gbdt_val_mae_delta"] > 0.25, "essay_set"].astype(int))
    manual = ({6} | balanced) - bad_mae
    if 8 in diagnostics["essay_set"].to_numpy() and int(
        diagnostics.loc[diagnostics["essay_set"].eq(8), "gbdt_val_mae_delta"].iloc[0] > 0.25
    ):
        manual.discard(8)
    return {
        "qwk_only_selection": qwk_only,
        "balanced_selection": balanced,
        "conservative_selection": conservative,
        "manual_safe_group": manual,
    }


def select_calibration_sets(diagnostics: pd.DataFrame, selected_sets: set[int]) -> set[int]:
    mask = (
        diagnostics["essay_set"].isin(selected_sets)
        & (diagnostics["gbdt_calibrated_val_qwk"] > diagnostics["gbdt_val_qwk"])
        & (diagnostics["gbdt_calibrated_val_qwk_delta"] >= 0.005)
        & (diagnostics["gbdt_calibrated_val_mae_delta"] <= 0.05)
    )
    return set(diagnostics.loc[mask, "essay_set"].astype(int))


def calibrate_selected_sets(
    predictions: pd.DataFrame,
    weak: pd.DataFrame,
    selected_sets: set[int],
    variant_name: str,
    lambda_value: float,
) -> pd.DataFrame:
    pieces = []
    for essay_set, group in predictions.groupby("essay_set", sort=True):
        group = group.copy()
        if int(essay_set) in selected_sets:
            reference = weak[weak["essay_set"].astype(int).eq(int(essay_set))]["weak_label_normalized"].to_numpy(dtype=float)
            mapped = weak_quantile_map_to_score(group["weak_prediction"].to_numpy(dtype=float), reference, group)
            group["pred_score"] = (1.0 - lambda_value) * group["pred_score"].to_numpy(dtype=float) + lambda_value * mapped
            score_min = group["score_min"].to_numpy(dtype=float)
            score_max = group["score_max"].to_numpy(dtype=float)
            group["pred_score"] = np.clip(group["pred_score"], score_min, score_max)
            group["pred_score_rounded"] = np.rint(group["pred_score"]).clip(score_min, score_max)
        group["variant_name"] = variant_name
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["split", "essay_set", "essay_id"]).reset_index(drop=True)


def weak_quantile_map_to_score(values: np.ndarray, reference: np.ndarray, group: pd.DataFrame) -> np.ndarray:
    levels = np.arange(int(group["score_min"].iloc[0]), int(group["score_max"].iloc[0]) + 1)
    if len(reference) == 0:
        return np.full(len(values), float(levels[len(levels) // 2]))
    sorted_ref = np.sort(reference)
    pct = np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)
    indices = np.clip(np.floor(pct * len(levels)).astype(int), 0, len(levels) - 1)
    return levels[indices].astype(float)


def build_set6_plus_optional_set8_calibration(
    predictions_by_variant: dict[str, pd.DataFrame],
    best_gbdt_by_set: dict[int, str],
    weak: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    base = build_targeted_prediction("set6_gbdt_set8_baseline_or_calibration", predictions_by_variant, best_gbdt_by_set, {6})
    set8 = diagnostics[diagnostics["essay_set"].eq(8)]
    calibration_sets = set()
    if not set8.empty:
        row = set8.iloc[0]
        if row["gbdt_calibrated_val_qwk_delta"] >= 0.005 and row["gbdt_calibrated_val_mae_delta"] <= 0.05:
            calibration_sets.add(8)
    return calibrate_selected_sets(base, weak, calibration_sets, "set6_gbdt_set8_baseline_or_calibration", 0.25)


def compute_feature_importances(
    trained: dict[str, dict[int, object]],
    processed: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for variant_name in GBDT_CANDIDATES:
        for essay_set, model in sorted(trained[variant_name].items()):
            val = features["val"][features["val"]["essay_set"].astype(int).eq(essay_set)].copy()
            target = processed["val"][processed["val"]["essay_set"].astype(int).eq(essay_set)][
                ["essay_id", "essay_set", "gold_score", "score_min", "score_max"]
            ]
            val = val.merge(target, on=["essay_id", "essay_set"], how="inner", validate="one_to_one")
            normalized_gold = (val["gold_score"] - val["score_min"]) / (val["score_max"] - val["score_min"])
            if hasattr(model, "feature_importances_"):
                scores = np.asarray(model.feature_importances_, dtype=float)
                method = "native"
            else:
                result = permutation_importance(
                    model,
                    val[FEATURE_COLUMNS],
                    normalized_gold,
                    n_repeats=5,
                    random_state=seed,
                    scoring="neg_mean_absolute_error",
                )
                scores = result.importances_mean
                method = "validation_permutation"
            for feature, importance in zip(FEATURE_COLUMNS, scores):
                rows.append(
                    {
                        "variant_name": variant_name,
                        "essay_set": int(essay_set),
                        "feature": feature,
                        "importance": float(importance),
                        "importance_method": method,
                    }
                )
    return pd.DataFrame(rows).sort_values(["variant_name", "essay_set", "importance"], ascending=[True, True, False])


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics[metrics["variant_name"].eq("baseline_positive_linear")][
        ["split", "qwk", "mae", "pearson"]
    ].rename(columns={"qwk": "baseline_qwk", "mae": "baseline_mae", "pearson": "baseline_pearson"})
    out = metrics.merge(baseline, on="split", how="left")
    out["delta_qwk_vs_baseline"] = out["qwk"] - out["baseline_qwk"]
    out["delta_mae_vs_baseline"] = out["mae"] - out["baseline_mae"]
    out["delta_pearson_vs_baseline"] = out["pearson"] - out["baseline_pearson"]
    return out.sort_values(["split", "qwk"], ascending=[True, False]).reset_index(drop=True)


def add_per_set_deltas(per_set_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = per_set_metrics[per_set_metrics["variant_name"].eq("baseline_positive_linear")][
        ["split", "essay_set", "qwk", "mae", "pearson"]
    ].rename(columns={"qwk": "baseline_qwk", "mae": "baseline_mae", "pearson": "baseline_pearson"})
    out = per_set_metrics.merge(baseline, on=["split", "essay_set"], how="left")
    out["delta_qwk_vs_baseline"] = out["qwk"] - out["baseline_qwk"]
    out["delta_mae_vs_baseline"] = out["mae"] - out["baseline_mae"]
    out["delta_pearson_vs_baseline"] = out["pearson"] - out["baseline_pearson"]
    return out.sort_values(["split", "essay_set", "delta_qwk_vs_baseline"], ascending=[True, True, False])


def selected_sets_to_frame(
    selected_sets: dict[str, set[int]],
    best_gbdt_by_set: dict[int, str],
    calibration_sets: set[int],
    manual_calibration_sets: set[int],
    val_selected_models: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for rule, sets in selected_sets.items():
        for essay_set in sorted(sets):
            rows.append(
                {
                    "selection_rule": rule,
                    "essay_set": essay_set,
                    "selected_model": best_gbdt_by_set.get(essay_set, "baseline_positive_linear"),
                    "uses_shrinked_calibration": essay_set in calibration_sets,
                }
            )
    for essay_set, model_name in sorted(val_selected_models.items()):
        rows.append(
            {
                "selection_rule": "val_selected_per_essay_set_model",
                "essay_set": essay_set,
                "selected_model": model_name,
                "uses_shrinked_calibration": False,
            }
        )
    for essay_set in sorted(manual_calibration_sets):
        rows.append(
            {
                "selection_rule": "manual_safe_group_recommended_calibration",
                "essay_set": essay_set,
                "selected_model": best_gbdt_by_set.get(essay_set, "baseline_positive_linear"),
                "uses_shrinked_calibration": True,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["selection_rule", "essay_set", "selected_model", "uses_shrinked_calibration"])
    return pd.DataFrame(rows)


def build_mistral_subset_note() -> pd.DataFrame:
    possible_paths = [
        Path("results/experiments/mistral_aes_hybrid_25pilot/labels/aes_prompt_understanding_v2_25_per_set_labels.jsonl"),
        Path("results/experiments/mistral_prompt_ablation/labels/aes_prompt_ablation_labels.jsonl"),
    ]
    found = [str(path) for path in possible_paths if path.exists()]
    if not found:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "No cached AES Mistral label file was found in this worktree; no API calls were made.",
                    "labels_file": "",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "status": "skipped",
                "reason": "Cached AES Mistral labels exist, but this runner keeps the targeted GBDT test on full current weak labels only.",
                "labels_file": ";".join(found),
            }
        ]
    )


def write_summary(
    clean_dir: Path,
    baseline_reference: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_sets: pd.DataFrame,
    per_set_delta: pd.DataFrame,
    feature_importance: pd.DataFrame,
    mistral_subset: pd.DataFrame,
) -> None:
    non_baseline = metrics[~metrics["variant_name"].eq("baseline_positive_linear")]
    clean = non_baseline[~non_baseline["variant_name"].eq("val_selected_per_essay_set_model")]
    best_clean_val = clean[clean["split"].eq("val")].sort_values(["qwk", "mae"], ascending=[False, True]).iloc[0]
    best_clean_test = clean[clean["split"].eq("test")].sort_values(["qwk", "mae"], ascending=[False, True]).iloc[0]
    set6 = metrics[(metrics["split"].eq("test")) & (metrics["variant_name"].eq("set6_gbdt_only"))].iloc[0]
    val_selected = metrics[(metrics["split"].eq("test")) & (metrics["variant_name"].eq("val_selected_per_essay_set_model"))].iloc[0]
    top_features = feature_importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(6)
    lines = [
        "# AES Targeted GBDT Experiment",
        "",
        "This AES-only experiment selects GBDT essay sets using validation metrics, then reports test metrics after selection.",
        "",
        "## Baseline",
        "",
    ]
    for row in baseline_reference.itertuples(index=False):
        lines.append(f"- {row.split}: QWK {row.qwk:.6f}, MAE {row.mae:.6f}, Pearson {row.pearson:.6f}.")
    lines.extend(["", "## Selected Sets", ""])
    for rule, group in selected_sets.groupby("selection_rule", sort=False):
        chosen = ", ".join(str(int(value)) for value in sorted(group["essay_set"].unique()))
        lines.append(f"- `{rule}`: {chosen if chosen else 'none'}.")
    lines.extend(
        [
            "",
            "## Best Results",
            "",
            f"- Best clean validation candidate: `{best_clean_val.variant_name}` with val QWK {best_clean_val.qwk:.6f}, "
            f"MAE {best_clean_val.mae:.6f}, Pearson {best_clean_val.pearson:.6f}.",
            f"- Best clean test candidate: `{best_clean_test.variant_name}` with test QWK {best_clean_test.qwk:.6f}, "
            f"MAE {best_clean_test.mae:.6f}, Pearson {best_clean_test.pearson:.6f}.",
            f"- Set-6-only candidate: test QWK {set6.qwk:.6f}, MAE {set6.mae:.6f}, Pearson {set6.pearson:.6f}.",
            f"- Validation-selected candidate: test QWK {val_selected.qwk:.6f}, MAE {val_selected.mae:.6f}, Pearson {val_selected.pearson:.6f}.",
            "",
            "## Per-Set Diagnostics",
            "",
        ]
    )
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"- Set {int(row.essay_set)}: best GBDT `{row.best_gbdt_variant}`, "
            f"val delta QWK {row.gbdt_val_qwk_delta:+.6f}, val delta MAE {row.gbdt_val_mae_delta:+.6f}, "
            f"recommendation `{row.recommendation}`."
        )
    lines.extend(["", "## Feature Importance", ""])
    for row in top_features.itertuples(index=False):
        lines.append(f"- `{row.feature}` mean importance {row.importance:.6f}.")
    lines.extend(["", "## Mistral Subset Ablation", ""])
    for row in mistral_subset.itertuples(index=False):
        lines.append(f"- {row.status}: {row.reason}")
    lines.extend(["", "## Recommendation", ""])
    if best_clean_val.delta_qwk_vs_baseline > 0 and best_clean_test.delta_qwk_vs_baseline > 0 and best_clean_test.delta_mae_vs_baseline <= 0.05:
        lines.append("- A targeted GBDT candidate is clean enough to consider extracting.")
    elif best_clean_val.delta_qwk_vs_baseline > 0 and best_clean_test.delta_qwk_vs_baseline > 0:
        lines.append("- Targeted GBDT improves QWK but has a meaningful MAE tradeoff; keep it as an ablation unless QWK is the priority.")
    else:
        lines.append("- Do not keep targeted GBDT as a clean candidate; gains are not validation/test consistent.")
    lines.append("- Do not apply GBDT globally.")
    clean_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(clean_dir: Path) -> None:
    clean_dir.joinpath("README.md").write_text(
        "\n".join(
            [
                "# AES Targeted GBDT Quick Study",
                "",
                "Command:",
                "",
                "```bash",
                "python3 -m src.run_aes_gbdt_targeted",
                "```",
                "",
                "The runner uses existing ASAP-AES processed splits, weak labels, and handcrafted features. It does not change the default AES baseline command.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_graph(clean_dir: Path, metrics: pd.DataFrame) -> None:
    names = [
        "baseline_positive_linear",
        "set6_gbdt_only",
        "set6_gbdt_plus_shrinked_calibration",
        "gbdt_selected_balanced",
        "gbdt_manual_safe_group_plus_recommended_calibration",
        "val_selected_per_essay_set_model",
    ]
    test = metrics[metrics["split"].eq("test")].set_index("variant_name")
    present = [name for name in names if name in test.index]
    labels_by_name = {
        "baseline_positive_linear": "Baseline",
        "set6_gbdt_only": "Set 6 GBDT",
        "set6_gbdt_plus_shrinked_calibration": "Set 6 + cal",
        "gbdt_selected_balanced": "Balanced",
        "gbdt_manual_safe_group_plus_recommended_calibration": "Manual + cal",
        "val_selected_per_essay_set_model": "Val selected",
    }
    labels = [labels_by_name[name] for name in present]
    values = test.loc[present, "qwk"].to_numpy(dtype=float)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=["#6b7280", "#2563eb", "#059669", "#d97706", "#7c3aed"][: len(present)])
    plt.ylabel("Test QWK")
    plt.title("AES Targeted GBDT Test QWK")
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(clean_dir / "figures" / "aes_targeted_gbdt_qwk.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
