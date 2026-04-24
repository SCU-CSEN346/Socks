"""Canonical end-to-end runner for the ASAP-SAS ASAG baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import asag_baseline
from .asag_data import (
    DEFAULT_ASAP_SAS_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_PROCESSED_DIR,
    DEFAULT_SEED,
    DEFAULT_VAL_SIZE,
    build_asap_sas_splits,
    write_processed_splits,
)
from .asag_weak_labels import DEFAULT_OUTPUT_DIR as DEFAULT_WEAK_LABEL_DIR, generate_asag_weak_labels
from .data_loading import PROJECT_ROOT


DEFAULT_FEATURE_DIR = PROJECT_ROOT / "results" / "features" / "asap-sas"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions" / "asap-sas"
DEFAULT_METRIC_DIR = PROJECT_ROOT / "results" / "metrics" / "asap-sas"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results" / "models" / "asap-sas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical ASAP-SAS ASAG baseline from preprocessing through evaluation."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ASAP_SAS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--weak-label-dir", type=Path, default=DEFAULT_WEAK_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
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
        seed=args.seed,
        min_df=args.min_df,
        max_iter=args.max_iter,
        eps=args.eps,
    )

    print("ASAG baseline complete.")
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
    seed: int,
    min_df: int,
    max_iter: int,
    eps: float,
) -> dict[str, object]:
    splits, score_ranges = build_asap_sas_splits(data_dir=data_dir, val_size=val_size, seed=seed)
    processed_paths = write_processed_splits(splits, score_ranges, output_dir=processed_dir)

    weak_labels, diagnostics = generate_asag_weak_labels(
        input_path=processed_dir / "train.csv",
        min_df=min_df,
        max_iter=max_iter,
        eps=eps,
    )
    weak_label_dir.mkdir(parents=True, exist_ok=True)
    weak_label_path = weak_label_dir / "train_signal_clustering.csv"
    diagnostics_path = weak_label_dir / "train_signal_clustering_diagnostics.csv"
    weak_labels.to_csv(weak_label_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    baseline_outputs = asag_baseline.run_pipeline(
        processed_dir=processed_dir,
        weak_label_path=weak_label_path,
        feature_dir=feature_dir,
        prediction_dir=prediction_dir,
        metric_dir=metric_dir,
        model_dir=model_dir,
    )

    paths = {
        **{f"processed_{name}": path for name, path in processed_paths.items()},
        "weak_labels_train": weak_label_path,
        "weak_label_diagnostics": diagnostics_path,
        **baseline_outputs["paths"],
    }
    return {"macro_metrics": baseline_outputs["macro_metrics"], "paths": paths}


if __name__ == "__main__":
    main()
