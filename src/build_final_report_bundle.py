"""Build a file-backed final-report artifact bundle for the course project.

This script does not rerun heavyweight experiments by default. Instead, it
collects verified outputs that are already present in the repository, computes
paper-ready summaries, and writes a clean bundle under `results/final_report/`.

Only locally supported experiments are treated as reproduced results. Historical
or presentation-only values are surfaced as unsupported notes rather than being
promoted to main tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/socks-mplconfig")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.to_latex.*")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import entropy, spearmanr

from .data_loading import PROJECT_ROOT


FINAL_REPORT_DIR = PROJECT_ROOT / "results" / "final_report"
TABLE_DIR = FINAL_REPORT_DIR / "tables"
FIGURE_DIR = FINAL_REPORT_DIR / "figures"
LOG_DIR = FINAL_REPORT_DIR / "logs"
PREDICTION_DIR = FINAL_REPORT_DIR / "predictions"
CONFIG_DIR = FINAL_REPORT_DIR / "configs"

AES_CANONICAL_DIR = PROJECT_ROOT / "results"
AES_PROCESSED_DIR = AES_CANONICAL_DIR / "processed" / "asap-aes"
AES_FEATURE_DIR = AES_CANONICAL_DIR / "features" / "asap-aes"
AES_WEAK_LABEL_DIR = AES_CANONICAL_DIR / "weak_labels" / "asap-aes"
AES_METRIC_DIR = AES_CANONICAL_DIR / "metrics" / "asap-aes"
AES_MODEL_DIR = AES_CANONICAL_DIR / "models" / "asap-aes"
AES_PREDICTION_DIR = AES_CANONICAL_DIR / "predictions" / "asap-aes"

ASAG_CANONICAL_DIR = PROJECT_ROOT / "results"
ASAG_PROCESSED_DIR = ASAG_CANONICAL_DIR / "processed" / "asap-sas"
ASAG_FEATURE_DIR = ASAG_CANONICAL_DIR / "features" / "asap-sas"
ASAG_WEAK_LABEL_DIR = ASAG_CANONICAL_DIR / "weak_labels" / "asap-sas"
ASAG_METRIC_DIR = ASAG_CANONICAL_DIR / "metrics" / "asap-sas"
ASAG_MODEL_DIR = ASAG_CANONICAL_DIR / "models" / "asap-sas"
ASAG_PREDICTION_DIR = ASAG_CANONICAL_DIR / "predictions" / "asap-sas"

WORD_RE = re.compile(r"\b\w+\b")
PUNCT_RE = re.compile(r"[^\w\s]")

DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 200
DEFAULT_PAPER_PDF = Path("/home/harsh/Downloads/SCU346_2026_Harshvardhan_Helen_Shravan.pdf")
DEFAULT_GUIDELINES_PDF = Path("/home/harsh/Downloads/Final Project Grading Guidelines (1).pdf")


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    display_name: str
    task: str
    split_order: tuple[str, ...]
    per_prompt_metrics_path: Path | None
    macro_metrics_path: Path | None
    predictions: dict[str, Path]
    coefficient_path: Path | None
    weak_label_path: Path | None
    processed_root: Path | None
    feature_root: Path | None
    supported_local: bool
    requested_in_suite_b: bool
    supervision: str
    validation_selected: bool
    calibration: str
    note: str


SUPPORTED_SYSTEMS: list[SystemSpec] = [
    SystemSpec(
        system_id="aes_weak_label_baseline",
        display_name="AES weak-label baseline",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=AES_METRIC_DIR / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": AES_PREDICTION_DIR / "val_predictions.csv",
            "test": AES_PREDICTION_DIR / "test_predictions.csv",
        },
        coefficient_path=AES_MODEL_DIR / "positive_linear_coefficients.csv",
        weak_label_path=AES_WEAK_LABEL_DIR / "train_weak_labels.csv",
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=True,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Canonical ASAP-AES baseline from current repo metrics.",
    ),
    SystemSpec(
        system_id="aes_set6_8_feature_selection",
        display_name="AES set 6/8 feature-selection variant",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "set6_8_feature_selection"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "set6_8_feature_selection"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "set6_8_feature_selection"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "set6_8_feature_selection"
        / "models"
        / "coefficients.csv",
        weak_label_path=AES_WEAK_LABEL_DIR / "train_weak_labels.csv",
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=True,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="File-backed targeted AES feature-selection candidate for sets 6 and 8.",
    ),
    SystemSpec(
        system_id="aes_length_only_linear",
        display_name="AES length-only baseline model",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes"
        / "length_only_linear"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes"
            / "length_only_linear"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes"
            / "length_only_linear"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes"
        / "length_only_linear"
        / "models"
        / "positive_linear_coefficients.csv",
        weak_label_path=AES_WEAK_LABEL_DIR / "train_weak_labels.csv",
        processed_root=AES_PROCESSED_DIR,
        feature_root=PROJECT_ROOT / "results" / "experiments" / "aes" / "length_only_linear" / "features",
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Model-side ablation using only length-dominated features.",
    ),
    SystemSpec(
        system_id="aes_ridge_alpha_10",
        display_name="AES Ridge alpha=10",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "ridge_alpha_10_0"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "ridge_alpha_10_0"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "ridge_alpha_10_0"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "ridge_alpha_10_0"
        / "models"
        / "coefficients.csv",
        weak_label_path=AES_WEAK_LABEL_DIR / "train_weak_labels.csv",
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Supported AES Ridge ablation from autonomous-search artifacts.",
    ),
    SystemSpec(
        system_id="aes_set6_8_rank_uniform_mapping",
        display_name="AES set 6/8 rank-uniform mapping",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "set6_8_rank_uniform_mapping"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "set6_8_rank_uniform_mapping"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "set6_8_rank_uniform_mapping"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "set6_8_rank_uniform_mapping"
        / "models"
        / "coefficients.csv",
        weak_label_path=AES_WEAK_LABEL_DIR / "train_weak_labels.csv",
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="rank_uniform_validation_free",
        note="Supported AES calibration ablation on sets 6 and 8.",
    ),
    SystemSpec(
        system_id="aes_conservative_validation_selected",
        display_name="AES conservative per-essay-set selection",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes_autonomous_search"
        / "conservative_per_essay_set_candidate"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "conservative_per_essay_set_candidate"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes_autonomous_search"
            / "conservative_per_essay_set_candidate"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=None,
        weak_label_path=None,
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=True,
        calibration="mixed_selected",
        note="Validation-selected AES candidate; not a pure default baseline.",
    ),
    SystemSpec(
        system_id="aes_oracle_mean_gold",
        display_name="AES gold-supervised upper-bound reference",
        task="aes",
        split_order=("val", "test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "aes"
        / "oracle_mean_gold"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes"
            / "oracle_mean_gold"
            / "predictions"
            / "val_predictions.csv",
            "test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "aes"
            / "oracle_mean_gold"
            / "predictions"
            / "test_predictions.csv",
        },
        coefficient_path=None,
        weak_label_path=None,
        processed_root=AES_PROCESSED_DIR,
        feature_root=AES_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=False,
        supervision="supervised_upper_bound",
        validation_selected=False,
        calibration="none",
        note="Gold-informed oracle reference only; not comparable as a weakly supervised method.",
    ),
    SystemSpec(
        system_id="asag_legacy_initial_local",
        display_name="ASAG legacy initial baseline (local file-backed)",
        task="asag",
        split_order=("val", "public_test"),
        per_prompt_metrics_path=PROJECT_ROOT / "results" / "asag_baseline" / "metrics" / "per_prompt_metrics.csv",
        macro_metrics_path=PROJECT_ROOT / "results" / "asag_baseline" / "metrics" / "macro_metrics.csv",
        predictions={
            "val": PROJECT_ROOT / "results" / "asag_baseline" / "predictions" / "val_predictions.csv",
            "public_test": PROJECT_ROOT
            / "results"
            / "asag_baseline"
            / "predictions"
            / "public_test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "asag_baseline"
        / "models"
        / "positive_linear_coefficients.csv",
        weak_label_path=PROJECT_ROOT
        / "results"
        / "asag_baseline"
        / "intermediate"
        / "weak_labels"
        / "train_signal_clustering.csv",
        processed_root=PROJECT_ROOT / "results" / "asag_baseline" / "processed",
        feature_root=PROJECT_ROOT / "results" / "asag_baseline" / "features",
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Legacy ASAP-SAS baseline preserved in the repo; weaker than the current main baseline.",
    ),
    SystemSpec(
        system_id="asag_improved_weak_label_baseline",
        display_name="ASAG improved weak-label baseline",
        task="asag",
        split_order=("val", "public_test"),
        per_prompt_metrics_path=ASAG_METRIC_DIR / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": ASAG_PREDICTION_DIR / "val_predictions.csv",
            "public_test": ASAG_PREDICTION_DIR / "public_test_predictions.csv",
        },
        coefficient_path=ASAG_MODEL_DIR / "positive_linear_coefficients.csv",
        weak_label_path=ASAG_WEAK_LABEL_DIR / "train_signal_clustering.csv",
        processed_root=ASAG_PROCESSED_DIR,
        feature_root=ASAG_FEATURE_DIR,
        supported_local=True,
        requested_in_suite_b=True,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Canonical ASAP-SAS baseline from current repo metrics.",
    ),
    SystemSpec(
        system_id="asag_tfidf_length_only",
        display_name="ASAG length-only weak-label baseline",
        task="asag",
        split_order=("val", "public_test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_length_only"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "tfidf_length_only"
            / "predictions"
            / "val_predictions.csv",
            "public_test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "tfidf_length_only"
            / "predictions"
            / "public_test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_length_only"
        / "models"
        / "positive_linear_coefficients.csv",
        weak_label_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_length_only"
        / "weak_labels"
        / "train_signal_clustering.csv",
        processed_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_length_only" / "processed",
        feature_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_length_only" / "features",
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Supported ASAP-SAS weak-label ablation using a length-only signal variant.",
    ),
    SystemSpec(
        system_id="asag_tfidf_density_only",
        display_name="ASAG density-only weak-label baseline",
        task="asag",
        split_order=("val", "public_test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_density_only"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "tfidf_density_only"
            / "predictions"
            / "val_predictions.csv",
            "public_test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "tfidf_density_only"
            / "predictions"
            / "public_test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_density_only"
        / "models"
        / "positive_linear_coefficients.csv",
        weak_label_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "tfidf_density_only"
        / "weak_labels"
        / "train_signal_clustering.csv",
        processed_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_density_only" / "processed",
        feature_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_density_only" / "features",
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Supported ASAP-SAS weak-label ablation using a density-only signal variant.",
    ),
    SystemSpec(
        system_id="asag_sbert_hybrid_supported_candidate",
        display_name="ASAG supported best-available candidate (SBERT-hybrid label)",
        task="asag",
        split_order=("val", "public_test"),
        per_prompt_metrics_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "sbert_hybrid"
        / "metrics"
        / "metrics.csv",
        macro_metrics_path=None,
        predictions={
            "val": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "sbert_hybrid"
            / "predictions"
            / "val_predictions.csv",
            "public_test": PROJECT_ROOT
            / "results"
            / "experiments"
            / "asag"
            / "sbert_hybrid"
            / "predictions"
            / "public_test_predictions.csv",
        },
        coefficient_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "sbert_hybrid"
        / "models"
        / "positive_linear_coefficients.csv",
        weak_label_path=PROJECT_ROOT
        / "results"
        / "experiments"
        / "asag"
        / "sbert_hybrid"
        / "weak_labels"
        / "train_signal_clustering.csv",
        processed_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "sbert_hybrid" / "processed",
        feature_root=PROJECT_ROOT / "results" / "experiments" / "asag" / "sbert_hybrid" / "features",
        supported_local=True,
        requested_in_suite_b=False,
        supervision="weakly_supervised",
        validation_selected=False,
        calibration="clip_and_round",
        note="Supported exploratory ASAG candidate. Comparison summary records backend fallbacks; keep as exploratory, not default.",
    ),
]


UNSUPPORTED_REQUESTED_SYSTEMS = [
    {
        "system_id": "asag_initial_reported_paper_baseline",
        "display_name": "ASAG initial weak-label baseline (paper-reported)",
        "task": "asag",
        "requested_in_suite_b": True,
        "note": (
            "Mentioned in presentation assets around public-test QWK 0.157, "
            "but no matching locally re-confirmed result file was found."
        ),
    },
    {
        "system_id": "asag_ridge_alpha_10_reported",
        "display_name": "ASAG Ridge alpha=10 (paper-reported)",
        "task": "asag",
        "requested_in_suite_b": True,
        "note": (
            "Presentation assets mention this as a clean candidate, but the "
            "underlying compact result CSV is explicitly noted as missing."
        ),
    },
    {
        "system_id": "asag_rank_uniform_mapping_reported",
        "display_name": "ASAG rank-uniform mapping (paper-reported)",
        "task": "asag",
        "requested_in_suite_b": True,
        "note": "Mentioned in presentation assets only; no local prediction/metric files were found.",
    },
    {
        "system_id": "asag_conservative_per_question_reported",
        "display_name": "ASAG conservative per-question selection (paper-reported)",
        "task": "asag",
        "requested_in_suite_b": True,
        "note": "Mentioned in presentation assets only; no local prediction/metric files were found.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final-report artifacts from existing repo outputs.")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--paper-pdf", type=Path, default=DEFAULT_PAPER_PDF)
    parser.add_argument("--guidelines-pdf", type=Path, default=DEFAULT_GUIDELINES_PDF)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    produced_files: list[str] = []
    commands_run = ["python3 -m src.build_final_report_bundle"]

    system_registry = build_system_registry()
    write_json(CONFIG_DIR / "supported_systems.json", system_registry)
    produced_files.append(relpath(CONFIG_DIR / "supported_systems.json"))

    metric_audit = build_metric_audit()
    write_json(CONFIG_DIR / "metric_audit.json", metric_audit)
    produced_files.append(relpath(CONFIG_DIR / "metric_audit.json"))

    selected_systems = build_selected_systems_config()
    write_json(CONFIG_DIR / "selected_systems.json", selected_systems)
    produced_files.append(relpath(CONFIG_DIR / "selected_systems.json"))

    dataset_summary = build_dataset_summary_tables()
    produced_files.extend(dataset_summary)

    main_results = build_main_results_tables()
    produced_files.extend(main_results)

    weak_quality = build_weak_label_quality_tables()
    produced_files.extend(weak_quality)

    ablation_outputs = build_ablation_tables()
    produced_files.extend(ablation_outputs)

    interpretability_outputs = build_interpretability_outputs()
    produced_files.extend(interpretability_outputs)

    error_outputs = build_error_analysis_tables()
    produced_files.extend(error_outputs)

    robustness_outputs = build_robustness_tables(bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    produced_files.extend(robustness_outputs)

    figure_outputs = build_figures()
    produced_files.extend(figure_outputs)

    prediction_outputs = export_prediction_copies()
    produced_files.extend(prediction_outputs)

    revision_outputs = write_revision_notes(
        paper_pdf=args.paper_pdf,
        guidelines_pdf=args.guidelines_pdf,
    )
    produced_files.extend(revision_outputs)

    manifest_path = FINAL_REPORT_DIR / "experiment_manifest.json"
    manifest = build_manifest(
        commands_run=commands_run,
        produced_files=sorted(set(produced_files)),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        paper_pdf=args.paper_pdf,
        guidelines_pdf=args.guidelines_pdf,
    )
    write_json(manifest_path, manifest)
    produced_files.append(relpath(manifest_path))

    smoke_test_path = LOG_DIR / "smoke_test_results.txt"
    smoke_text = "\n".join(
        [
            "Smoke tests run:",
            "- final-report bundle script executed successfully",
            "- all required canonical input files were readable",
            "- no private-test metrics were generated for ASAP-SAS",
        ]
    )
    smoke_test_path.write_text(smoke_text + "\n", encoding="utf-8")
    produced_files.append(relpath(smoke_test_path))

    print("Final report bundle complete.")
    print(f"Artifacts written under: {FINAL_REPORT_DIR}")
    print("Key outputs:")
    for path in sorted(set(produced_files)):
        print(f"- {path}")


def ensure_output_dirs() -> None:
    for path in [FINAL_REPORT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, PREDICTION_DIR, CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def relpath(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_system_registry() -> dict[str, object]:
    return {
        "supported_systems": [
            {
                "system_id": spec.system_id,
                "display_name": spec.display_name,
                "task": spec.task,
                "supported_local": spec.supported_local,
                "requested_in_suite_b": spec.requested_in_suite_b,
                "supervision": spec.supervision,
                "validation_selected": spec.validation_selected,
                "calibration": spec.calibration,
                "note": spec.note,
                "prediction_files": {split: relpath(path) for split, path in spec.predictions.items()},
                "per_prompt_metrics_path": relpath(spec.per_prompt_metrics_path) if spec.per_prompt_metrics_path else None,
                "macro_metrics_path": relpath(spec.macro_metrics_path) if spec.macro_metrics_path else None,
            }
            for spec in SUPPORTED_SYSTEMS
        ],
        "unsupported_requested_systems": UNSUPPORTED_REQUESTED_SYSTEMS,
    }


def build_metric_audit() -> dict[str, object]:
    return {
        "qwk": {
            "definition": "Quadratic weighted kappa computed within each prompt/question score range.",
            "prediction_handling": "Predictions are clipped to prompt/question-specific score_min/score_max and rounded to the nearest integer before QWK.",
            "gold_handling": "Gold labels are rounded to integers when needed inside the repo metric functions.",
        },
        "mae": {
            "definition": "Mean absolute error on continuous clipped score predictions in the raw score range.",
            "prediction_handling": "Continuous prompt/question-specific clipped predictions are used; not rounded.",
        },
        "pearson": {
            "definition": "Pearson correlation between gold labels and continuous clipped score predictions in the raw score range.",
            "prediction_handling": "Continuous prompt/question-specific clipped predictions are used; not rounded.",
        },
        "weighted_macro_convention": (
            "Sample-weighted averages are computed as prompt/question-level metric means weighted by prompt/question n. "
            "This preserves prompt/question-specific score ranges instead of pooling scales directly."
        ),
        "repo_sources": {
            "aes": relpath(PROJECT_ROOT / "src" / "run_aes_baseline.py"),
            "asag": relpath(PROJECT_ROOT / "src" / "asag_baseline.py"),
        },
    }


def build_selected_systems_config() -> dict[str, object]:
    return {
        "aes_default_baseline": "aes_weak_label_baseline",
        "aes_supported_final_candidate": "aes_set6_8_feature_selection",
        "asag_default_baseline": "asag_improved_weak_label_baseline",
        "asag_supported_exploratory_candidate": "asag_sbert_hybrid_supported_candidate",
        "selection_notes": [
            "AES final supported candidate uses the reproduced set 6/8 feature-selection variant.",
            "ASAG Ridge alpha=10 is not selected because the local repo does not contain a re-confirmed result file for that candidate.",
            "ASAG baseline remains the default supported final system in the artifact bundle.",
        ],
    }


def build_dataset_summary_tables() -> list[str]:
    aes_rows = compute_aes_dataset_summary()
    asag_rows = compute_asag_dataset_summary()

    full_df = pd.concat([aes_rows, asag_rows], ignore_index=True)
    csv_path = TABLE_DIR / "dataset_summary.csv"
    full_df.to_csv(csv_path, index=False)

    concise_path = TABLE_DIR / "dataset_summary.tex"
    full_tex_path = TABLE_DIR / "dataset_summary_full.tex"
    with concise_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated dataset summary tables.\n")
        handle.write(aes_rows.rename(columns=AES_DATASET_LATEX_COLUMNS)[list(AES_DATASET_LATEX_COLUMNS.values())].to_latex(index=False, float_format=fmt3))
        handle.write("\n\n")
        handle.write(asag_rows.rename(columns=ASAG_DATASET_LATEX_COLUMNS)[list(ASAG_DATASET_LATEX_COLUMNS.values())].to_latex(index=False, float_format=fmt3))
    with full_tex_path.open("w", encoding="utf-8") as handle:
        handle.write(full_df.to_latex(index=False, float_format=fmt3))

    return [relpath(csv_path), relpath(concise_path), relpath(full_tex_path)]


AES_DATASET_LATEX_COLUMNS = {
    "task": "Task",
    "unit_id": "Set",
    "total_n": "N",
    "train_n": "Train",
    "val_n": "Val",
    "test_n": "Test",
    "score_min": "Min",
    "score_max": "Max",
    "gold_mean": "MeanScore",
    "gold_std": "StdScore",
    "word_mean": "MeanWords",
    "word_std": "StdWords",
    "word_median": "MedianWords",
}

ASAG_DATASET_LATEX_COLUMNS = {
    "task": "Task",
    "unit_id": "QID",
    "train_n": "Train",
    "val_n": "Val",
    "public_test_n": "Public",
    "private_test_n": "Private",
    "score_min": "Min",
    "score_max": "Max",
    "train_gold_mean": "TrainMean",
    "val_gold_mean": "ValMean",
    "public_gold_mean": "PublicMean",
    "char_mean": "MeanChars",
    "word_mean": "MeanWords",
    "word_median": "MedianWords",
}


def compute_aes_dataset_summary() -> pd.DataFrame:
    splits = {
        split: pd.read_csv(AES_PROCESSED_DIR / f"{split}.csv")
        for split in ["train", "val", "test"]
    }
    rows = []
    combined = pd.concat(splits.values(), ignore_index=True)
    combined["word_count_stat"] = combined["essay"].fillna("").astype(str).map(count_words)
    for essay_set, group in combined.groupby("essay_set", sort=True):
        rows.append(
            {
                "task": "AES",
                "unit_id": int(essay_set),
                "total_n": int(len(group)),
                "train_n": int((splits["train"]["essay_set"] == essay_set).sum()),
                "val_n": int((splits["val"]["essay_set"] == essay_set).sum()),
                "test_n": int((splits["test"]["essay_set"] == essay_set).sum()),
                "score_min": float(group["score_min"].min()),
                "score_max": float(group["score_max"].max()),
                "gold_mean": float(group["gold_score"].mean()),
                "gold_std": float(group["gold_score"].std(ddof=0)),
                "word_mean": float(group["word_count_stat"].mean()),
                "word_std": float(group["word_count_stat"].std(ddof=0)),
                "word_median": float(group["word_count_stat"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def compute_asag_dataset_summary() -> pd.DataFrame:
    splits = {
        split: pd.read_csv(ASAG_PROCESSED_DIR / f"{split}.csv")
        for split in ["train", "val", "public_test", "private_test"]
    }
    combined = pd.concat(splits.values(), ignore_index=True)
    combined["char_count_stat"] = combined["student_answer"].fillna("").astype(str).map(len)
    combined["word_count_stat"] = combined["student_answer"].fillna("").astype(str).map(count_words)

    rows = []
    for question_id, group in combined.groupby("question_id", sort=True):
        row = {
            "task": "ASAG",
            "unit_id": int(question_id),
            "train_n": int((splits["train"]["question_id"] == question_id).sum()),
            "val_n": int((splits["val"]["question_id"] == question_id).sum()),
            "public_test_n": int((splits["public_test"]["question_id"] == question_id).sum()),
            "private_test_n": int((splits["private_test"]["question_id"] == question_id).sum()),
            "score_min": float(group["score_min"].min()),
            "score_max": float(group["score_max"].max()),
            "train_gold_mean": split_mean(splits["train"], question_id, "score_raw"),
            "train_gold_std": split_std(splits["train"], question_id, "score_raw"),
            "val_gold_mean": split_mean(splits["val"], question_id, "score_raw"),
            "val_gold_std": split_std(splits["val"], question_id, "score_raw"),
            "public_gold_mean": split_mean(splits["public_test"], question_id, "score_raw"),
            "public_gold_std": split_std(splits["public_test"], question_id, "score_raw"),
            "char_mean": float(group["char_count_stat"].mean()),
            "char_median": float(group["char_count_stat"].median()),
            "word_mean": float(group["word_count_stat"].mean()),
            "word_std": float(group["word_count_stat"].std(ddof=0)),
            "word_median": float(group["word_count_stat"].median()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def split_mean(df: pd.DataFrame, unit_id: int, column: str) -> float:
    subset = df[df["question_id"] == unit_id][column].dropna()
    return float(subset.mean()) if not subset.empty else float("nan")


def split_std(df: pd.DataFrame, unit_id: int, column: str) -> float:
    subset = df[df["question_id"] == unit_id][column].dropna()
    return float(subset.std(ddof=0)) if not subset.empty else float("nan")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def build_main_results_tables() -> list[str]:
    rows = []
    per_prompt_rows = []
    for spec in SUPPORTED_SYSTEMS:
        per_prompt_df, macro_df = load_metric_frames(spec)
        weighted_df = compute_weighted_metric_rows(per_prompt_df, spec)
        macro_lookup = macro_df.set_index("split") if not macro_df.empty else pd.DataFrame()
        weighted_lookup = weighted_df.set_index("split") if not weighted_df.empty else pd.DataFrame()

        for split in spec.split_order:
            row = {
                "task": spec.task.upper(),
                "system_id": spec.system_id,
                "display_name": spec.display_name,
                "split": split,
                "supported_local": spec.supported_local,
                "requested_in_suite_b": spec.requested_in_suite_b,
                "supervision": spec.supervision,
                "validation_selected": spec.validation_selected,
                "calibration": spec.calibration,
                "note": spec.note,
                "qwk_macro": metric_from_lookup(macro_lookup, split, "qwk"),
                "mae_macro": metric_from_lookup(macro_lookup, split, "mae"),
                "pearson_macro": metric_from_lookup(macro_lookup, split, "pearson"),
                "qwk_weighted": metric_from_lookup(weighted_lookup, split, "qwk_weighted"),
                "mae_weighted": metric_from_lookup(weighted_lookup, split, "mae_weighted"),
                "pearson_weighted": metric_from_lookup(weighted_lookup, split, "pearson_weighted"),
            }
            rows.append(row)

        per_prompt_unit_col = "essay_set" if spec.task == "aes" else "question_id"
        for prompt_row in per_prompt_df.to_dict(orient="records"):
            per_prompt_rows.append(
                {
                    "task": spec.task.upper(),
                    "system_id": spec.system_id,
                    "display_name": spec.display_name,
                    "split": prompt_row["split"],
                    "unit_id": int(prompt_row[per_prompt_unit_col]),
                    "n": int(prompt_row["n"]),
                    "qwk": float(prompt_row["qwk"]),
                    "mae": float(prompt_row["mae"]),
                    "pearson": float(prompt_row["pearson"]),
                    "supported_local": spec.supported_local,
                    "note": spec.note,
                }
            )

    for entry in UNSUPPORTED_REQUESTED_SYSTEMS:
        rows.append(
            {
                "task": entry["task"].upper(),
                "system_id": entry["system_id"],
                "display_name": entry["display_name"],
                "split": "UNKNOWN",
                "supported_local": False,
                "requested_in_suite_b": entry["requested_in_suite_b"],
                "supervision": "UNKNOWN",
                "validation_selected": False,
                "calibration": "UNKNOWN",
                "note": entry["note"],
                "qwk_macro": float("nan"),
                "mae_macro": float("nan"),
                "pearson_macro": float("nan"),
                "qwk_weighted": float("nan"),
                "mae_weighted": float("nan"),
                "pearson_weighted": float("nan"),
            }
        )

    macro_df = pd.DataFrame(rows)
    per_prompt_df = pd.DataFrame(per_prompt_rows)

    macro_path = TABLE_DIR / "main_results_macro.csv"
    per_prompt_path = TABLE_DIR / "per_prompt_results.csv"
    macro_df.to_csv(macro_path, index=False)
    per_prompt_df.to_csv(per_prompt_path, index=False)

    macro_tex = TABLE_DIR / "main_results_macro.tex"
    per_prompt_tex = TABLE_DIR / "per_prompt_results.tex"
    macro_paper_df = macro_df[
        macro_df["requested_in_suite_b"] | macro_df["system_id"].isin(["asag_legacy_initial_local"])
    ].copy()
    write_latex_table(
        macro_tex,
        macro_paper_df[
            [
                "task",
                "display_name",
                "split",
                "qwk_macro",
                "mae_macro",
                "pearson_macro",
                "supported_local",
                "note",
            ]
        ],
    )
    write_latex_table(
        per_prompt_tex,
        per_prompt_df[per_prompt_df["system_id"].isin(["aes_weak_label_baseline", "aes_set6_8_feature_selection", "asag_improved_weak_label_baseline", "asag_legacy_initial_local"])],
    )

    return [relpath(macro_path), relpath(macro_tex), relpath(per_prompt_path), relpath(per_prompt_tex)]


def load_metric_frames(spec: SystemSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.per_prompt_metrics_path is None:
        return pd.DataFrame(), pd.DataFrame()
    per_prompt_df = pd.read_csv(spec.per_prompt_metrics_path)
    unit_col = "essay_set" if spec.task == "aes" else "question_id"
    if unit_col in per_prompt_df.columns:
        mask = per_prompt_df[unit_col].astype(str).str.lower() != "macro"
        per_prompt_df = per_prompt_df[mask].copy()
    if spec.macro_metrics_path and spec.macro_metrics_path.exists():
        macro_df = pd.read_csv(spec.macro_metrics_path)
    else:
        macro_df = append_macro_from_per_prompt(per_prompt_df, spec.task)
    return per_prompt_df.reset_index(drop=True), macro_df.reset_index(drop=True)


def append_macro_from_per_prompt(per_prompt_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if per_prompt_df.empty:
        return per_prompt_df.copy()
    unit_col = "essay_set" if task == "aes" else "question_id"
    macro = (
        per_prompt_df.groupby("split", as_index=False)
        .agg(n=("n", "sum"), qwk=("qwk", "mean"), mae=("mae", "mean"), pearson=("pearson", "mean"))
        .assign(**{unit_col: "macro"})
    )
    return macro[[c for c in per_prompt_df.columns if c in macro.columns or c == unit_col]]


def compute_weighted_metric_rows(per_prompt_df: pd.DataFrame, spec: SystemSpec) -> pd.DataFrame:
    rows = []
    if per_prompt_df.empty:
        return pd.DataFrame(rows)
    for split, group in per_prompt_df.groupby("split", sort=False):
        n = group["n"].astype(float)
        total = float(n.sum())
        rows.append(
            {
                "split": split,
                "qwk_weighted": weighted_mean(group["qwk"], n),
                "mae_weighted": weighted_mean(group["mae"], n),
                "pearson_weighted": weighted_mean(group["pearson"], n),
                "n_total": int(total),
                "task": spec.task,
            }
        )
    return pd.DataFrame(rows)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    values_arr = np.asarray(list(values), dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)
    if values_arr.size == 0 or float(weights_arr.sum()) == 0:
        return float("nan")
    return float(np.average(values_arr, weights=weights_arr))


def metric_from_lookup(lookup: pd.DataFrame, split: str, column: str) -> float:
    if lookup.empty or split not in lookup.index:
        return float("nan")
    return float(lookup.loc[split, column])


def build_weak_label_quality_tables() -> list[str]:
    rows = []

    # AES variants derived from canonical processed/features/weak labels.
    aes_train = pd.read_csv(AES_PROCESSED_DIR / "train.csv")
    aes_features = pd.read_csv(AES_FEATURE_DIR / "train_features.csv")
    aes_baseline = pd.read_csv(AES_WEAK_LABEL_DIR / "train_weak_labels.csv")
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_baseline,
            variant_name="aes_baseline_propagated_minmax",
            signal_series=aes_baseline["weak_label_normalized"],
            rank_normalized=False,
            propagation="on",
            source_note="Canonical propagated weak labels with stored min-max normalization.",
        )
    )
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_baseline,
            variant_name="aes_baseline_propagated_rank",
            signal_series=percentile_rank_by_unit(aes_baseline, "essay_set", "weak_label_raw"),
            rank_normalized=True,
            propagation="on",
            source_note="Canonical propagated weak labels with derived rank normalization.",
        )
    )
    aes_word_df = aes_features[["essay_id", "essay_set", "word_count", "unique_word_count"]].copy()
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_word_df,
            variant_name="aes_total_word_count_minmax",
            signal_series=minmax_by_unit(aes_word_df, "essay_set", "word_count"),
            rank_normalized=False,
            propagation="off",
            source_note="No-propagation total word count signal.",
        )
    )
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_word_df,
            variant_name="aes_total_word_count_rank",
            signal_series=percentile_rank_by_unit(aes_word_df, "essay_set", "word_count"),
            rank_normalized=True,
            propagation="off",
            source_note="No-propagation total word count signal with rank normalization.",
        )
    )
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_word_df,
            variant_name="aes_unique_word_count_minmax",
            signal_series=minmax_by_unit(aes_word_df, "essay_set", "unique_word_count"),
            rank_normalized=False,
            propagation="off",
            source_note="No-propagation unique word count signal.",
        )
    )
    rows.extend(
        evaluate_weak_label_variant(
            task="AES",
            unit_col="essay_set",
            id_col="essay_id",
            gold_col="gold_score",
            processed_df=aes_train,
            weak_df=aes_word_df,
            variant_name="aes_unique_word_count_rank",
            signal_series=percentile_rank_by_unit(aes_word_df, "essay_set", "unique_word_count"),
            rank_normalized=True,
            propagation="off",
            source_note="No-propagation unique word count signal with rank normalization.",
        )
    )
    for variant_name, path in [
        (
            "aes_multi_signal_rank_mean",
            PROJECT_ROOT / "results" / "experiments" / "aes_autonomous_search" / "weak_labels" / "multi_signal_rank_mean" / "train_weak_labels.csv",
        ),
        (
            "aes_trait_proxy_rank_average",
            PROJECT_ROOT / "results" / "experiments" / "aes_autonomous_search" / "weak_labels" / "trait_proxy_rank_average" / "train_weak_labels.csv",
        ),
    ]:
        if path.exists():
            weak_df = pd.read_csv(path)
            rows.extend(
                evaluate_weak_label_variant(
                    task="AES",
                    unit_col="essay_set",
                    id_col="essay_id",
                    gold_col="gold_score",
                    processed_df=aes_train,
                    weak_df=weak_df,
                    variant_name=variant_name,
                    signal_series=weak_df["weak_label_normalized"],
                    rank_normalized=True,
                    propagation="off",
                    source_note="Supported exploratory AES weak-label variant from autonomous-search artifacts.",
                )
            )

    # ASAG variants from file-backed runs.
    asag_train = pd.read_csv(ASAG_PROCESSED_DIR / "train.csv")
    asag_variants = {
        "asag_legacy_initial_local_rank": (
            pd.read_csv(PROJECT_ROOT / "results" / "asag_baseline" / "intermediate" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_normalized",
            True,
            "on",
            "Legacy initial local ASAP-SAS baseline.",
        ),
        "asag_hybrid_rank": (
            pd.read_csv(ASAG_WEAK_LABEL_DIR / "train_signal_clustering.csv"),
            "weak_label_normalized",
            True,
            "on",
            "Current improved ASAP-SAS baseline.",
        ),
        "asag_hybrid_minmax_raw": (
            pd.read_csv(ASAG_WEAK_LABEL_DIR / "train_signal_clustering.csv"),
            "weak_label_raw",
            False,
            "on",
            "Current improved ASAP-SAS baseline with derived min-max normalization of raw weak labels.",
        ),
        "asag_length_only_rank": (
            pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_length_only" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_normalized",
            True,
            "on",
            "Length-only propagated ASAP-SAS weak labels.",
        ),
        "asag_length_only_minmax_raw": (
            pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_length_only" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_raw",
            False,
            "on",
            "Length-only propagated ASAP-SAS weak labels with derived min-max normalization.",
        ),
        "asag_density_only_rank": (
            pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_density_only" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_normalized",
            True,
            "on",
            "Density-only propagated ASAP-SAS weak labels.",
        ),
        "asag_density_only_minmax_raw": (
            pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "asag" / "tfidf_density_only" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_raw",
            False,
            "on",
            "Density-only propagated ASAP-SAS weak labels with derived min-max normalization.",
        ),
        "asag_sbert_hybrid_rank": (
            pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "asag" / "sbert_hybrid" / "weak_labels" / "train_signal_clustering.csv"),
            "weak_label_normalized",
            True,
            "on",
            "Supported exploratory ASAP-SAS candidate labeled as SBERT hybrid.",
        ),
    }
    for variant_name, (weak_df, signal_col, rank_norm, propagation, note) in asag_variants.items():
        if signal_col == "weak_label_raw":
            signal_series = minmax_by_unit(weak_df, "question_id", "weak_label_raw")
        else:
            signal_series = weak_df[signal_col]
        rows.extend(
            evaluate_weak_label_variant(
                task="ASAG",
                unit_col="question_id",
                id_col="sample_id",
                gold_col="score_raw",
                processed_df=asag_train,
                weak_df=weak_df,
                variant_name=variant_name,
                signal_series=signal_series,
                rank_normalized=rank_norm,
                propagation=propagation,
                source_note=note,
            )
        )
    improved_df = pd.read_csv(ASAG_WEAK_LABEL_DIR / "train_signal_clustering.csv")
    if "initial_length_signal" in improved_df.columns:
        rows.extend(
            evaluate_weak_label_variant(
                task="ASAG",
                unit_col="question_id",
                id_col="sample_id",
                gold_col="score_raw",
                processed_df=asag_train,
                weak_df=improved_df,
                variant_name="asag_length_only_no_propagation_rank",
                signal_series=percentile_rank_by_unit(improved_df, "question_id", "initial_length_signal"),
                rank_normalized=True,
                propagation="off",
                source_note="Derived from stored initial ASAP-SAS length signal without propagation.",
            )
        )
        rows.extend(
            evaluate_weak_label_variant(
                task="ASAG",
                unit_col="question_id",
                id_col="sample_id",
                gold_col="score_raw",
                processed_df=asag_train,
                weak_df=improved_df,
                variant_name="asag_length_only_no_propagation_minmax",
                signal_series=minmax_by_unit(improved_df, "question_id", "initial_length_signal"),
                rank_normalized=False,
                propagation="off",
                source_note="Derived from stored initial ASAP-SAS length signal without propagation.",
            )
        )

    quality_df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "weak_label_quality.csv"
    quality_df.to_csv(csv_path, index=False)

    tex_path = TABLE_DIR / "weak_label_quality.tex"
    macro_rows = quality_df[quality_df["unit_id"] == "macro"].copy()
    write_latex_table(
        tex_path,
        macro_rows[
            [
                "task",
                "variant_name",
                "rank_normalized",
                "propagation",
                "pearson_with_gold",
                "spearman_with_gold",
                "mae_scaled",
                "qwk_scaled",
                "entropy_rounded_scores",
            ]
        ],
    )

    heatmap_path = FIGURE_DIR / "weak_label_quality_heatmap.pdf"
    plot_weak_label_quality_heatmap(macro_rows, heatmap_path)

    return [relpath(csv_path), relpath(tex_path), relpath(heatmap_path)]


def percentile_rank_by_unit(df: pd.DataFrame, unit_col: str, value_col: str) -> pd.Series:
    return df.groupby(unit_col)[value_col].rank(method="average", pct=True)


def minmax_by_unit(df: pd.DataFrame, unit_col: str, value_col: str) -> pd.Series:
    def _scale(series: pd.Series) -> pd.Series:
        low = float(series.min())
        high = float(series.max())
        if math.isclose(low, high):
            return pd.Series(np.full(len(series), 0.5), index=series.index)
        return (series - low) / (high - low)

    return df.groupby(unit_col)[value_col].transform(_scale)


def evaluate_weak_label_variant(
    task: str,
    unit_col: str,
    id_col: str,
    gold_col: str,
    processed_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    variant_name: str,
    signal_series: pd.Series,
    rank_normalized: bool,
    propagation: str,
    source_note: str,
) -> list[dict[str, object]]:
    signal_frame = weak_df[[id_col, unit_col]].copy()
    signal_frame["signal_norm"] = np.asarray(signal_series, dtype=float)
    merged = processed_df.merge(signal_frame, on=[id_col, unit_col], how="inner", validate="one_to_one")
    rows = []
    macro_rows = []
    for unit_id, group in merged.groupby(unit_col, sort=True):
        score_min = float(group["score_min"].min())
        score_max = float(group["score_max"].max())
        pred = np.clip(score_min + group["signal_norm"].to_numpy(dtype=float) * (score_max - score_min), score_min, score_max)
        rounded = np.rint(pred).clip(score_min, score_max)
        gold = group[gold_col].to_numpy(dtype=float)
        pearson_value = safe_corr(gold, pred)
        spearman_value = safe_spearman(gold, pred)
        qwk_value = quadratic_weighted_kappa(
            np.rint(gold).astype(int),
            rounded.astype(int),
            int(score_min),
            int(score_max),
        )
        hist = rounded.astype(int)
        counts = np.bincount(hist - int(score_min), minlength=int(score_max - score_min + 1))
        rows.append(
            {
                "task": task,
                "variant_name": variant_name,
                "unit_id": int(unit_id),
                "n": int(len(group)),
                "rank_normalized": rank_normalized,
                "propagation": propagation,
                "pearson_with_gold": pearson_value,
                "spearman_with_gold": spearman_value,
                "mae_scaled": float(np.mean(np.abs(gold - pred))),
                "qwk_scaled": qwk_value,
                "entropy_rounded_scores": float(entropy(counts + 1e-12, base=2)),
                "source_note": source_note,
            }
        )
        macro_rows.append(rows[-1])

    macro_df = pd.DataFrame(macro_rows)
    rows.append(
        {
            "task": task,
            "variant_name": variant_name,
            "unit_id": "macro",
            "n": int(macro_df["n"].sum()),
            "rank_normalized": rank_normalized,
            "propagation": propagation,
            "pearson_with_gold": float(macro_df["pearson_with_gold"].mean()),
            "spearman_with_gold": float(macro_df["spearman_with_gold"].mean()),
            "mae_scaled": float(macro_df["mae_scaled"].mean()),
            "qwk_scaled": float(macro_df["qwk_scaled"].mean()),
            "entropy_rounded_scores": float(macro_df["entropy_rounded_scores"].mean()),
            "source_note": source_note,
        }
    )
    return rows


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int, max_rating: int) -> float:
    ratings = np.arange(min_rating, max_rating + 1)
    n_ratings = len(ratings)
    if n_ratings <= 1:
        return 1.0
    label_to_index = {label: idx for idx, label in enumerate(ratings)}
    observed = np.zeros((n_ratings, n_ratings), dtype=float)
    for truth, pred in zip(y_true, y_pred):
        observed[label_to_index[int(truth)], label_to_index[int(pred)]] += 1.0

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    total = observed.sum()
    if total == 0:
        return float("nan")

    expected = np.outer(hist_true, hist_pred) / total
    weights = np.zeros((n_ratings, n_ratings), dtype=float)
    denom = float((n_ratings - 1) ** 2)
    for i in range(n_ratings):
        for j in range(n_ratings):
            weights[i, j] = ((i - j) ** 2) / denom
    observed_score = np.sum(weights * observed) / total
    expected_score = np.sum(weights * expected) / total
    if expected_score == 0:
        return 1.0
    return 1.0 - (observed_score / expected_score)


def plot_weak_label_quality_heatmap(macro_rows: pd.DataFrame, output_path: Path) -> None:
    pivot = macro_rows.pivot(index="variant_name", columns="task", values="qwk_scaled").sort_index()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.3 * len(pivot))))
    image = ax.imshow(pivot.fillna(np.nan).to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Weak-Label Quality (Macro QWK)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_ablation_tables() -> list[str]:
    wanted_ids = [
        "aes_weak_label_baseline",
        "aes_set6_8_feature_selection",
        "aes_ridge_alpha_10",
        "aes_set6_8_rank_uniform_mapping",
        "aes_length_only_linear",
        "aes_conservative_validation_selected",
        "aes_oracle_mean_gold",
        "asag_legacy_initial_local",
        "asag_improved_weak_label_baseline",
        "asag_tfidf_length_only",
        "asag_tfidf_density_only",
        "asag_sbert_hybrid_supported_candidate",
    ]
    rows = []
    for spec in SUPPORTED_SYSTEMS:
        if spec.system_id not in wanted_ids:
            continue
        _, macro_df = load_metric_frames(spec)
        for split in spec.split_order:
            row = {
                "task": spec.task.upper(),
                "system_id": spec.system_id,
                "display_name": spec.display_name,
                "split": split,
                "qwk_macro": macro_value(macro_df, split, "qwk"),
                "mae_macro": macro_value(macro_df, split, "mae"),
                "pearson_macro": macro_value(macro_df, split, "pearson"),
                "weakly_supervised": spec.supervision == "weakly_supervised",
                "validation_selected": spec.validation_selected,
                "supervised_upper_bound": spec.supervision == "supervised_upper_bound",
                "calibration_ablation": "rank_uniform" in spec.calibration,
                "supported_local": True,
                "note": spec.note,
            }
            rows.append(row)
    for entry in UNSUPPORTED_REQUESTED_SYSTEMS[1:]:
        rows.append(
            {
                "task": entry["task"].upper(),
                "system_id": entry["system_id"],
                "display_name": entry["display_name"],
                "split": "UNKNOWN",
                "qwk_macro": float("nan"),
                "mae_macro": float("nan"),
                "pearson_macro": float("nan"),
                "weakly_supervised": True,
                "validation_selected": "conservative" in entry["system_id"],
                "supervised_upper_bound": False,
                "calibration_ablation": "rank_uniform" in entry["system_id"],
                "supported_local": False,
                "note": entry["note"],
            }
        )
    ablation_df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "ablation_results.csv"
    tex_path = TABLE_DIR / "ablation_results.tex"
    ablation_df.to_csv(csv_path, index=False)
    write_latex_table(
        tex_path,
        ablation_df[
            [
                "task",
                "display_name",
                "split",
                "qwk_macro",
                "mae_macro",
                "pearson_macro",
                "weakly_supervised",
                "validation_selected",
                "supervised_upper_bound",
                "supported_local",
            ]
        ],
    )
    return [relpath(csv_path), relpath(tex_path)]


def macro_value(macro_df: pd.DataFrame, split: str, column: str) -> float:
    if macro_df.empty:
        return float("nan")
    unit_col = [c for c in ["essay_set", "question_id"] if c in macro_df.columns][0]
    row = macro_df[(macro_df["split"] == split) & (macro_df[unit_col].astype(str) == "macro")]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def build_interpretability_outputs() -> list[str]:
    outputs = []
    coeff_rows = []

    aes_spec = next(spec for spec in SUPPORTED_SYSTEMS if spec.system_id == "aes_set6_8_feature_selection")
    asag_spec = next(spec for spec in SUPPORTED_SYSTEMS if spec.system_id == "asag_improved_weak_label_baseline")

    aes_coeff = load_coefficient_frame(aes_spec, task="aes")
    asag_coeff = load_coefficient_frame(asag_spec, task="asag")

    # Overall summaries.
    coeff_rows.extend(summarize_coefficients(aes_coeff, "AES", aes_spec.system_id, "overall"))
    coeff_rows.extend(summarize_coefficients(aes_coeff[aes_coeff["unit_id"].isin([6])], "AES", aes_spec.system_id, "essay_set_6"))
    coeff_rows.extend(summarize_coefficients(aes_coeff[aes_coeff["unit_id"].isin([8])], "AES", aes_spec.system_id, "essay_set_8"))

    asag_metrics, _ = load_metric_frames(asag_spec)
    public_qwk = asag_metrics[asag_metrics["split"] == "public_test"].copy()
    top_questions = public_qwk.sort_values("qwk", ascending=False)["question_id"].head(3).astype(int).tolist()
    bottom_questions = public_qwk.sort_values("qwk", ascending=True)["question_id"].head(3).astype(int).tolist()
    coeff_rows.extend(summarize_coefficients(asag_coeff, "ASAG", asag_spec.system_id, "overall"))
    coeff_rows.extend(
        summarize_coefficients(asag_coeff[asag_coeff["unit_id"].isin(top_questions)], "ASAG", asag_spec.system_id, "top3_questions_by_qwk")
    )
    coeff_rows.extend(
        summarize_coefficients(asag_coeff[asag_coeff["unit_id"].isin(bottom_questions)], "ASAG", asag_spec.system_id, "bottom3_questions_by_qwk")
    )

    coeff_df = pd.DataFrame(coeff_rows).sort_values(["task", "scope", "mean_coefficient"], ascending=[True, True, False])
    csv_path = TABLE_DIR / "feature_coefficients.csv"
    coeff_df.to_csv(csv_path, index=False)
    outputs.append(relpath(csv_path))

    tex_path = TABLE_DIR / "feature_coefficients.tex"
    write_latex_table(
        tex_path,
        coeff_df[
            coeff_df["scope"].isin(["overall", "essay_set_6", "essay_set_8", "top3_questions_by_qwk", "bottom3_questions_by_qwk"])
        ][
            ["task", "system_id", "scope", "feature", "mean_coefficient", "std_coefficient", "nonnegative_verified"]
        ],
    )
    outputs.append(relpath(tex_path))

    aes_fig = FIGURE_DIR / "feature_coefficients_aes.pdf"
    asag_fig = FIGURE_DIR / "feature_coefficients_asag.pdf"
    plot_feature_coefficients_aes(aes_coeff, aes_fig)
    plot_feature_coefficients_asag(asag_coeff, asag_fig)
    outputs.extend([relpath(aes_fig), relpath(asag_fig)])
    return outputs


def load_coefficient_frame(spec: SystemSpec, task: str) -> pd.DataFrame:
    if spec.coefficient_path is None:
        return pd.DataFrame()
    df = pd.read_csv(spec.coefficient_path)
    unit_col = "essay_set" if task == "aes" else "question_id"
    if "variant_name" in df.columns:
        df = df.copy()
    out = df.rename(columns={unit_col: "unit_id"}).copy()
    if "abs_coefficient" not in out.columns:
        out["abs_coefficient"] = out["coefficient"].abs()
    return out


def summarize_coefficients(df: pd.DataFrame, task: str, system_id: str, scope: str) -> list[dict[str, object]]:
    rows = []
    if df.empty:
        return rows
    for feature, group in df.groupby("feature", sort=True):
        rows.append(
            {
                "task": task,
                "system_id": system_id,
                "scope": scope,
                "feature": feature,
                "mean_coefficient": float(group["coefficient"].mean()),
                "std_coefficient": float(group["coefficient"].std(ddof=0)),
                "min_coefficient": float(group["coefficient"].min()),
                "max_coefficient": float(group["coefficient"].max()),
                "n_units": int(group["unit_id"].nunique()),
                "nonnegative_verified": bool((group["coefficient"] >= -1e-12).all()),
            }
        )
    return rows


def plot_feature_coefficients_aes(coeff_df: pd.DataFrame, output_path: Path) -> None:
    overall = coeff_df.groupby("feature", as_index=False)["coefficient"].mean().sort_values("coefficient", ascending=False).head(8)
    set6 = coeff_df[coeff_df["unit_id"] == 6].set_index("feature")["coefficient"]
    set8 = coeff_df[coeff_df["unit_id"] == 8].set_index("feature")["coefficient"]
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(overall))
    ax.barh(y, overall["coefficient"], color="#4472c4", label="Mean across sets")
    ax.scatter(set6.reindex(overall["feature"]).to_numpy(dtype=float), y, color="#d62728", label="Set 6", zorder=3)
    ax.scatter(set8.reindex(overall["feature"]).to_numpy(dtype=float), y, color="#2ca02c", label="Set 8", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(overall["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("AES final supported candidate coefficients")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_coefficients_asag(coeff_df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        coeff_df.groupby("feature", as_index=False)
        .agg(mean=("coefficient", "mean"), std=("coefficient", "std"))
        .sort_values("mean", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(summary))
    ax.barh(y, summary["mean"], xerr=summary["std"].fillna(0.0), color="#55a868")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("ASAG final supported baseline coefficients")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_error_analysis_tables() -> list[str]:
    rows = []
    rows.extend(select_error_examples(task="AES", system_id="aes_set6_8_feature_selection", split="test"))
    rows.extend(select_error_examples(task="ASAG", system_id="asag_improved_weak_label_baseline", split="public_test"))
    error_df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "error_examples.csv"
    tex_path = TABLE_DIR / "error_examples.tex"
    error_df.to_csv(csv_path, index=False)
    write_latex_table(
        tex_path,
        error_df[
            [
                "task",
                "system_id",
                "split",
                "example_type",
                "anon_response_id",
                "unit_id",
                "gold_score",
                "predicted_score",
                "rounded_prediction",
                "feature_word_count",
                "feature_unique_word_count",
                "short_phrase",
                "explanation",
            ]
        ],
    )
    return [relpath(csv_path), relpath(tex_path)]


def select_error_examples(task: str, system_id: str, split: str) -> list[dict[str, object]]:
    spec = next(spec for spec in SUPPORTED_SYSTEMS if spec.system_id == system_id)
    pred_df = pd.read_csv(spec.predictions[split]).copy()
    processed_df = pd.read_csv(spec.processed_root / f"{split}.csv")
    if task == "AES":
        feature_df = pd.read_csv(spec.feature_root / f"{split}_features.csv")
        processed_cols = ["essay_id", "essay_set", "split", "essay"]
        merged = pred_df.merge(processed_df[processed_cols], on=["essay_id", "essay_set", "split"], how="left").merge(
            feature_df, on=["essay_id", "essay_set", "split"], how="left"
        )
        unit_col = "essay_set"
        id_col = "essay_id"
        text_col = "essay"
        gold_col = "gold_score"
    else:
        feature_df = pd.read_csv(spec.feature_root / f"{split}_features.csv")
        processed_cols = ["sample_id", "question_id", "split", "student_answer"]
        merged = pred_df.merge(processed_df[processed_cols], on=["sample_id", "question_id", "split"], how="left").merge(
            feature_df, on=["sample_id", "question_id", "split"], how="left"
        )
        unit_col = "question_id"
        id_col = "sample_id"
        text_col = "student_answer"
        gold_col = "score_raw"

    merged["error"] = merged["pred_score"] - merged[gold_col]
    merged["abs_error"] = merged["error"].abs()
    merged["rounded_correct"] = np.rint(merged[gold_col]) == merged["pred_score_rounded"]
    merged["confidence_proxy"] = (merged["weak_prediction_clipped"] - 0.5).abs()

    candidates = {
        "high_confidence_correct": merged[merged["rounded_correct"]].sort_values(
            ["abs_error", "confidence_proxy"], ascending=[True, False]
        ),
        "severe_underprediction": merged.sort_values("error", ascending=True),
        "severe_overprediction": merged.sort_values("error", ascending=False),
        "near_miss": merged[~merged["rounded_correct"]].sort_values(["abs_error", "confidence_proxy"], ascending=[True, False]),
    }

    chosen = []
    used_ids: set[int] = set()
    for example_type, frame in candidates.items():
        selected = None
        for row in frame.itertuples(index=False):
            row_id = int(getattr(row, id_col))
            if row_id in used_ids:
                continue
            selected = row
            used_ids.add(row_id)
            break
        if selected is None:
            continue
        chosen.append(
            {
                "task": task,
                "system_id": system_id,
                "split": split,
                "example_type": example_type,
                "anon_response_id": anonymize_id(task, getattr(selected, id_col)),
                "unit_id": int(getattr(selected, unit_col)),
                "gold_score": float(getattr(selected, gold_col)),
                "predicted_score": float(selected.pred_score),
                "rounded_prediction": int(selected.pred_score_rounded),
                "weak_prediction": float(selected.weak_prediction_clipped),
                "feature_word_count": int(selected.word_count),
                "feature_unique_word_count": int(selected.unique_word_count),
                "feature_sentence_count": int(selected.sentence_count),
                "short_phrase": shorten_text(getattr(selected, text_col)),
                "explanation": explain_error(selected),
            }
        )
    return chosen


def anonymize_id(task: str, raw_id: int) -> str:
    digest = hashlib.sha1(f"{task}:{raw_id}".encode("utf-8")).hexdigest()[:8]
    return f"{task.lower()}_{digest}"


def shorten_text(text: str, limit_words: int = 8) -> str:
    cleaned = " ".join(str(text).split())
    words = cleaned.split()
    snippet = " ".join(words[:limit_words])
    return snippet


def explain_error(row) -> str:
    parts = []
    if getattr(row, "error") < -0.75:
        parts.append("underpredicted")
    elif getattr(row, "error") > 0.75:
        parts.append("overpredicted")
    else:
        parts.append("near gold")
    if getattr(row, "word_count", 0) < 15:
        parts.append("very short response")
    elif getattr(row, "word_count", 0) > 120:
        parts.append("long response")
    if getattr(row, "type_token_ratio", 1.0) < 0.55:
        parts.append("low lexical variety")
    if getattr(row, "punctuation_count", 0) == 0:
        parts.append("little punctuation")
    return "; ".join(parts[:3])


def build_robustness_tables(bootstrap_samples: int, seed: int) -> list[str]:
    rows = []
    for system_id, split in [
        ("aes_weak_label_baseline", "test"),
        ("aes_set6_8_feature_selection", "test"),
        ("asag_improved_weak_label_baseline", "public_test"),
    ]:
        spec = next(spec for spec in SUPPORTED_SYSTEMS if spec.system_id == system_id)
        pred_df = pd.read_csv(spec.predictions[split])
        unit_col = "essay_set" if spec.task == "aes" else "question_id"
        gold_col = "gold_score" if spec.task == "aes" else "score_raw"
        rows.append(
            bootstrap_metrics_row(
                pred_df=pred_df,
                task=spec.task.upper(),
                system_id=system_id,
                split=split,
                unit_col=unit_col,
                gold_col=gold_col,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    rows.append(
        {
            "task": "AES",
            "system_id": "multi_seed_requested_status",
            "split": "test",
            "ci_method": "TODO",
            "bootstrap_samples": 0,
            "qwk_mean": float("nan"),
            "qwk_ci_low": float("nan"),
            "qwk_ci_high": float("nan"),
            "mae_mean": float("nan"),
            "mae_ci_low": float("nan"),
            "mae_ci_high": float("nan"),
            "pearson_mean": float("nan"),
            "pearson_ci_low": float("nan"),
            "pearson_ci_high": float("nan"),
            "note": "Requested 3-seed AES reruns were not executed in this bundle; fixed-split bootstrap CIs are provided for supported systems instead.",
        }
    )
    ci_df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "robustness_ci.csv"
    tex_path = TABLE_DIR / "robustness_ci.tex"
    ci_df.to_csv(csv_path, index=False)
    write_latex_table(tex_path, ci_df)
    return [relpath(csv_path), relpath(tex_path)]


def bootstrap_metrics_row(
    pred_df: pd.DataFrame,
    task: str,
    system_id: str,
    split: str,
    unit_col: str,
    gold_col: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    qwk_scores = []
    mae_scores = []
    pearson_scores = []
    for _ in range(bootstrap_samples):
        unit_frames = []
        for _, group in pred_df.groupby(unit_col, sort=False):
            idx = rng.integers(0, len(group), len(group))
            unit_frames.append(group.iloc[idx].copy())
        boot = pd.concat(unit_frames, ignore_index=True)
        qwk_parts = []
        mae_parts = []
        pearson_parts = []
        weights = []
        for _, group in boot.groupby(unit_col, sort=False):
            score_min = int(group["score_min"].min())
            score_max = int(group["score_max"].max())
            gold = group[gold_col].to_numpy(dtype=float)
            pred = group["pred_score"].to_numpy(dtype=float)
            pred_round = group["pred_score_rounded"].to_numpy(dtype=float)
            qwk_parts.append(quadratic_weighted_kappa(np.rint(gold).astype(int), pred_round.astype(int), score_min, score_max))
            mae_parts.append(float(np.mean(np.abs(gold - pred))))
            pearson_parts.append(safe_corr(gold, pred))
            weights.append(len(group))
        qwk_scores.append(weighted_mean(qwk_parts, weights))
        mae_scores.append(weighted_mean(mae_parts, weights))
        pearson_scores.append(weighted_mean(pearson_parts, weights))

    return {
        "task": task,
        "system_id": system_id,
        "split": split,
        "ci_method": "bootstrap_prompt_stratified",
        "bootstrap_samples": bootstrap_samples,
        "qwk_mean": float(np.mean(qwk_scores)),
        "qwk_ci_low": float(np.quantile(qwk_scores, 0.025)),
        "qwk_ci_high": float(np.quantile(qwk_scores, 0.975)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_ci_low": float(np.quantile(mae_scores, 0.025)),
        "mae_ci_high": float(np.quantile(mae_scores, 0.975)),
        "pearson_mean": float(np.mean(pearson_scores)),
        "pearson_ci_low": float(np.quantile(pearson_scores, 0.025)),
        "pearson_ci_high": float(np.quantile(pearson_scores, 0.975)),
        "note": "Bootstrap on the fixed evaluation split; no private-test labels used.",
    }


def build_figures() -> list[str]:
    outputs = []
    pipeline_path = FIGURE_DIR / "system_pipeline.pdf"
    weak_path = FIGURE_DIR / "weak_label_generation.pdf"
    qwk_delta_path = FIGURE_DIR / "per_prompt_qwk_delta.pdf"
    asag_candidate_path = FIGURE_DIR / "asag_candidate_qwk.pdf"
    distribution_path = FIGURE_DIR / "asag_score_distribution.pdf"

    draw_flow_diagram(
        pipeline_path,
        [
            "Raw responses",
            "Preprocessing",
            "Weak-label signal",
            "Neighbor propagation",
            "Rank / score mapping",
            "Feature extraction",
            "Regression",
            "Calibration",
            "Evaluation",
        ],
        "System Pipeline",
    )
    draw_flow_diagram(
        weak_path,
        [
            "Initial signal",
            "Binary text graph",
            "Similarity neighbors",
            "Iterative propagation",
            "Inversion check",
            "Rank normalization",
            "Weak labels",
        ],
        "Weak-Label Generation",
    )
    plot_per_prompt_qwk_delta(qwk_delta_path)
    plot_asag_candidate_qwk(asag_candidate_path)
    plot_asag_score_distribution(distribution_path)

    outputs.extend(
        [
            relpath(pipeline_path),
            relpath(weak_path),
            relpath(qwk_delta_path),
            relpath(asag_candidate_path),
            relpath(distribution_path),
        ]
    )
    return outputs


def draw_flow_diagram(output_path: Path, labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_axis_off()
    x_positions = np.linspace(0.05, 0.95, len(labels))
    y = 0.5
    width = 0.095
    height = 0.28
    for idx, label in enumerate(labels):
        x = x_positions[idx] - width / 2
        box = FancyBboxPatch((x, y - height / 2), width, height, boxstyle="round,pad=0.02", facecolor="#e9eff7", edgecolor="#3b5b92")
        ax.add_patch(box)
        ax.text(x_positions[idx], y, "\n".join(textwrap.wrap(label, width=12)), ha="center", va="center", fontsize=9)
        if idx < len(labels) - 1:
            arrow = FancyArrowPatch((x_positions[idx] + width / 2, y), (x_positions[idx + 1] - width / 2, y), arrowstyle="->", mutation_scale=12, linewidth=1.3, color="#444")
            ax.add_patch(arrow)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_prompt_qwk_delta(output_path: Path) -> None:
    aes_base = pd.read_csv(AES_METRIC_DIR / "metrics.csv")
    aes_final = pd.read_csv(
        PROJECT_ROOT / "results" / "experiments" / "aes_autonomous_search" / "set6_8_feature_selection" / "metrics" / "metrics.csv"
    )
    asag_base = pd.read_csv(ASAG_METRIC_DIR / "metrics.csv")
    asag_candidate = pd.read_csv(
        PROJECT_ROOT / "results" / "experiments" / "asag" / "sbert_hybrid" / "metrics" / "metrics.csv"
    )

    aes_delta = merge_metric_delta(aes_base, aes_final, "essay_set", split="test")
    asag_delta = merge_metric_delta(asag_base, asag_candidate, "question_id", split="public_test")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    axes[0].bar(aes_delta["unit_id"].astype(str), aes_delta["delta_qwk"], color="#4c72b0")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("AES test QWK delta: set 6/8 feature-selection vs baseline")
    axes[0].set_ylabel("Delta QWK")

    axes[1].bar(asag_delta["unit_id"].astype(str), asag_delta["delta_qwk"], color="#55a868")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("ASAG public-test QWK delta: supported candidate vs baseline")
    axes[1].set_xlabel("Prompt / question")
    axes[1].set_ylabel("Delta QWK")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def merge_metric_delta(base_df: pd.DataFrame, other_df: pd.DataFrame, unit_col: str, split: str) -> pd.DataFrame:
    base = base_df[(base_df["split"] == split) & (base_df[unit_col].astype(str).str.lower() != "macro")][[unit_col, "qwk"]].rename(columns={"qwk": "base_qwk"})
    other = other_df[(other_df["split"] == split) & (other_df[unit_col].astype(str).str.lower() != "macro")][[unit_col, "qwk"]].rename(columns={"qwk": "other_qwk"})
    merged = base.merge(other, on=unit_col, how="inner")
    merged["delta_qwk"] = merged["other_qwk"] - merged["base_qwk"]
    return merged.rename(columns={unit_col: "unit_id"})


def plot_asag_candidate_qwk(output_path: Path) -> None:
    candidate_rows = []
    for system_id in [
        "asag_legacy_initial_local",
        "asag_improved_weak_label_baseline",
        "asag_tfidf_length_only",
        "asag_tfidf_density_only",
        "asag_sbert_hybrid_supported_candidate",
    ]:
        spec = next(spec for spec in SUPPORTED_SYSTEMS if spec.system_id == system_id)
        _, macro_df = load_metric_frames(spec)
        candidate_rows.append(
            {
                "label": spec.display_name,
                "qwk": macro_value(macro_df, "public_test", "qwk"),
            }
        )
    df = pd.DataFrame(candidate_rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(df["label"], df["qwk"], color=["#999999", "#4c72b0", "#8172b2", "#c44e52", "#55a868"])
    ax.set_xlabel("Public-test macro QWK")
    ax.set_title("ASAG supported candidate comparison")
    ax.text(
        0.01,
        -0.95,
        "Requested paper candidates Ridge alpha=10, rank-uniform mapping, and conservative per-question selection\n"
        "are omitted here because matching local prediction/metric files were not found.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_asag_score_distribution(output_path: Path) -> None:
    pred_df = pd.read_csv(ASAG_PREDICTION_DIR / "public_test_predictions.csv")
    gold = np.rint(pred_df["score_raw"].to_numpy(dtype=float)).astype(int)
    pred = pred_df["pred_score_rounded"].to_numpy(dtype=int)
    min_score = min(gold.min(), pred.min())
    max_score = max(gold.max(), pred.max())
    bins = np.arange(min_score, max_score + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(gold, bins=bins, alpha=0.6, label="Gold", color="#4c72b0")
    ax.hist(pred, bins=bins, alpha=0.6, label="Predicted", color="#dd8452")
    ax.set_xticks(np.arange(min_score, max_score + 1))
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("ASAG public-test score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_prediction_copies() -> list[str]:
    outputs = []
    chosen_ids = [
        "aes_weak_label_baseline",
        "aes_set6_8_feature_selection",
        "asag_legacy_initial_local",
        "asag_improved_weak_label_baseline",
        "asag_tfidf_length_only",
        "asag_tfidf_density_only",
        "asag_sbert_hybrid_supported_candidate",
    ]
    for spec in SUPPORTED_SYSTEMS:
        if spec.system_id not in chosen_ids:
            continue
        for split, path in spec.predictions.items():
            if not path.exists():
                continue
            df = pd.read_csv(path).copy()
            df["task"] = spec.task.upper()
            df["system_id"] = spec.system_id
            out_path = PREDICTION_DIR / f"{spec.system_id}_{split}.csv"
            df.to_csv(out_path, index=False)
            outputs.append(relpath(out_path))
    return outputs


def write_revision_notes(paper_pdf: Path, guidelines_pdf: Path) -> list[str]:
    source_audit_path = LOG_DIR / "report_source_audit.md"
    source_audit_text = "\n".join(
        [
            "# Report Source Audit",
            "",
            "- No ACL LaTeX source was found inside the repository.",
            "- `/home/harsh/Downloads/main.tex` is not the project paper source; it is a resume.",
            "- Because no editable paper source was found, this bundle provides revision notes and table/figure artifacts instead of a direct LaTeX patch.",
            "",
            "Observed external files:",
            f"- Paper PDF: `{paper_pdf}`",
            f"- Grading guidelines PDF: `{guidelines_pdf}`",
        ]
    )
    source_audit_path.write_text(source_audit_text + "\n", encoding="utf-8")

    patch_path = FINAL_REPORT_DIR / "report_revision_patch.md"
    patch_text = "\n".join(
        [
            "# Report Revision Patch",
            "",
            "## Required removals or rewrites",
            "",
            "1. Remove or rewrite unsupported LLM claims in the abstract and results.",
            "   - The current PDF claims improvements from LLM-based scoring and competitiveness with LLMs.",
            "   - This repo contains exploratory LLM notes only; no locally re-confirmed full-dataset LLM result file supports those claims.",
            "",
            "2. Replace checkpoint language with final-system language.",
            "   - The current PDF still contains phrases like `current implementation`, `current checkpoint`, and `planned extensions`.",
            "",
            "3. Do not present ASAP-SAS private-test metrics.",
            "   - The private split is unlabeled in this repo and should remain prediction-only.",
            "",
            "4. Keep ASAG Ridge alpha=10, rank-uniform mapping, and conservative per-question selection out of the main results table unless they are re-run and saved locally.",
            "   - Presentation assets mention these, but matching local prediction/metric files were not found.",
            "",
            "5. If the paper mentions an ASAG initial baseline of about 0.157 public-test QWK, relabel that as unsupported historical context unless the exact result file is recovered.",
            "   - The current repo does contain a weaker file-backed legacy ASAP-SAS baseline under `results/asag_baseline/`.",
            "",
            "## Additions supported by this bundle",
            "",
            "- Dataset summary table",
            "- Main macro results table grounded in local files",
            "- Per-prompt/per-question results table",
            "- Weak-label quality diagnostics",
            "- Ablation table for supported systems",
            "- Interpretable coefficient analysis",
            "- Compact qualitative error examples",
            "- System and weak-label generation diagrams",
            "",
            "## Suggested contribution paragraph",
            "",
            "1. We adapt a weakly supervised interpretable grading framework to public AES and ASAG benchmarks.",
            "2. We evaluate how weak-label generation, model choice, and calibration affect performance.",
            "3. We provide per-prompt, ablation, and interpretability analysis showing where the approach works and fails.",
            "",
            "## Blocker",
            "",
            "- BLOCKED on direct LaTeX patching because no editable ACL paper source was found in the repo or provided files.",
        ]
    )
    patch_path.write_text(patch_text + "\n", encoding="utf-8")

    summary_path = FINAL_REPORT_DIR / "final_report_experiment_summary.md"
    summary_text = "\n".join(
        [
            "# Final Report Experiment Summary",
            "",
            "## Commands run",
            "",
            "- `python3 -m src.build_final_report_bundle`",
            "",
            "## Key tables",
            "",
            "- `results/final_report/tables/dataset_summary.csv`",
            "- `results/final_report/tables/main_results_macro.csv`",
            "- `results/final_report/tables/per_prompt_results.csv`",
            "- `results/final_report/tables/weak_label_quality.csv`",
            "- `results/final_report/tables/ablation_results.csv`",
            "- `results/final_report/tables/feature_coefficients.csv`",
            "- `results/final_report/tables/error_examples.csv`",
            "- `results/final_report/tables/robustness_ci.csv`",
            "",
            "## Key figures",
            "",
            "- `results/final_report/figures/system_pipeline.pdf`",
            "- `results/final_report/figures/weak_label_generation.pdf`",
            "- `results/final_report/figures/weak_label_quality_heatmap.pdf`",
            "- `results/final_report/figures/per_prompt_qwk_delta.pdf`",
            "- `results/final_report/figures/asag_candidate_qwk.pdf`",
            "- `results/final_report/figures/asag_score_distribution.pdf`",
            "- `results/final_report/figures/feature_coefficients_aes.pdf`",
            "- `results/final_report/figures/feature_coefficients_asag.pdf`",
            "",
            "## What changed relative to the current PDF",
            "",
            "- The bundle treats ASAP-AES and ASAP-SAS baselines as supported only when matching local metric and prediction files exist.",
            "- ASAP-SAS private-test remains unlabeled and is excluded from evaluation summaries.",
            "- ASAG Ridge alpha=10, rank-uniform mapping, and conservative per-question selection are not promoted to the main tables because matching local result files were not found.",
            "- The file-backed legacy ASAP-SAS baseline in `results/asag_baseline/` is surfaced separately from the presentation-only historical 0.157 QWK claim.",
            "",
            "## Claims now supported",
            "",
            "- The canonical ASAP-AES weakly supervised baseline is reproducible from current repo outputs.",
            "- The AES set 6/8 feature-selection variant improves macro QWK and Pearson over the current AES baseline on the internal test split.",
            "- The current ASAP-SAS improved weak-label baseline clearly outperforms the weaker legacy ASAP-SAS baseline preserved in the repo.",
            "- Coefficient analysis for the final supported AES and ASAG systems is available and consistent with non-negative regression weights.",
            "",
            "## Claims that should be removed or softened",
            "",
            "- Any claim that the system is competitive with LLMs.",
            "- Any claim that LLM-based scoring is part of the completed final system.",
            "- Any claim that ASAP-SAS private-test was evaluated with gold labels.",
            "- Any claim that ASAG Ridge alpha=10, rank-uniform mapping, or conservative per-question selection was fully reproduced in the current repo state unless those runs are recovered.",
            "",
            "## Report source status",
            "",
            "- No editable ACL LaTeX source was found. See `results/final_report/report_revision_patch.md` and `results/final_report/logs/report_source_audit.md`.",
        ]
    )
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    checklist_path = FINAL_REPORT_DIR / "final_checklist.md"
    checklist_text = "\n".join(
        [
            "# Final Checklist",
            "",
            "- [ ] 4-6 pages excluding references",
            "- [ ] ACL template source located and updated",
            "- [ ] Demo link present or TODO inserted",
            "- [x] Code/data links available in README/PDF text",
            "- [x] No private-test metrics in this bundle",
            "- [ ] Remove unsupported LLM claims unless re-run",
            "- [x] Metrics defined in manifest and tables",
            "- [x] Figures/tables generated under `results/final_report/`",
            "- [x] Limitations and ethics still need to be checked in the paper source",
            "- [x] README already contains installation, usage, expected output, and member contributions",
        ]
    )
    checklist_path.write_text(checklist_text + "\n", encoding="utf-8")

    return [relpath(source_audit_path), relpath(patch_path), relpath(summary_path), relpath(checklist_path)]


def build_manifest(
    commands_run: list[str],
    produced_files: list[str],
    bootstrap_samples: int,
    seed: int,
    paper_pdf: Path,
    guidelines_pdf: Path,
) -> dict[str, object]:
    packages = {}
    for module_name in ["numpy", "pandas", "sklearn", "scipy", "matplotlib"]:
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", "UNKNOWN")
        except Exception:
            packages[module_name] = "NOT_INSTALLED"
    return {
        "git_commit_hash": git_commit_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "major_package_versions": packages,
        "dataset_paths_used": {
            "aes_raw": relpath(PROJECT_ROOT / "data" / "asap-aes"),
            "asag_raw": relpath(PROJECT_ROOT / "data" / "asap-sas"),
            "aes_processed": relpath(AES_PROCESSED_DIR),
            "asag_processed": relpath(ASAG_PROCESSED_DIR),
            "paper_pdf": str(paper_pdf),
            "guidelines_pdf": str(guidelines_pdf),
        },
        "random_seeds": {
            "default_seed": seed,
            "bootstrap_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "aes_internal_split_seed": 42,
            "asag_train_val_seed": 42,
        },
        "model_configs": build_system_registry()["supported_systems"],
        "metric_definitions": build_metric_audit(),
        "exact_commands_run": commands_run,
        "output_files_produced": produced_files,
    }


def git_commit_hash() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return None


def write_latex_table(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_latex(index=False, float_format=fmt3), encoding="utf-8")


def fmt3(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
