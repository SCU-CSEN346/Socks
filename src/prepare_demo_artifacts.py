"""Prepare local demo artifacts for the ASAP-SAS Streamlit app.

This script builds lightweight per-question JSON artifacts for live inference
from the supported ASAP-SAS weakly supervised baseline. It prefers existing
file-backed model summaries from the repository and falls back to refitting the
same positive linear models from processed features plus weak labels if needed.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .aes_features import (
    DEFAULT_FEATURE_DIR as DEFAULT_AES_FEATURE_DIR,
    DEFAULT_PROCESSED_DIR as DEFAULT_AES_PROCESSED_DIR,
    FEATURE_COLUMNS as AES_FEATURE_COLUMNS,
    extract_feature_splits as extract_aes_feature_splits,
    load_processed_splits as load_aes_processed_splits,
)
from .asag_features import DEFAULT_FEATURE_DIR, DEFAULT_PROCESSED_DIR, FEATURE_COLUMNS, extract_feature_splits, load_processed_splits
from .data_loading import PROJECT_ROOT


DEMO_ARTIFACT_DIR = PROJECT_ROOT / "demo_artifacts"
DEFAULT_COEFFICIENT_PATH = PROJECT_ROOT / "results" / "models" / "asap-sas" / "positive_linear_coefficients.csv"
DEFAULT_WEAK_LABEL_PATH = PROJECT_ROOT / "results" / "weak_labels" / "asap-sas" / "train_signal_clustering.csv"
DEFAULT_SCORE_RANGE_PATH = PROJECT_ROOT / "results" / "processed" / "asap-sas" / "score_ranges.csv"
DEFAULT_MAIN_RESULTS_PATH = PROJECT_ROOT / "results" / "final_report" / "tables" / "main_results_macro.csv"
DEFAULT_WEAK_QUALITY_PATH = PROJECT_ROOT / "results" / "final_report" / "tables" / "weak_label_quality.csv"
DEFAULT_PER_PROMPT_RESULTS_PATH = PROJECT_ROOT / "results" / "final_report" / "tables" / "per_prompt_results.csv"
DEFAULT_FINAL_REPORT_PREDICTION_DIR = PROJECT_ROOT / "results" / "final_report" / "predictions"
DEFAULT_RESULTS_PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions" / "asap-sas"
DEFAULT_METRIC_PATH = PROJECT_ROOT / "results" / "metrics" / "asap-sas" / "metrics.csv"
DEFAULT_DESCRIPTION_ZIP_PATH = PROJECT_ROOT / "data" / "asap-sas" / "Data_Set_Descriptions.zip"
DEFAULT_AES_COEFFICIENT_PATH = PROJECT_ROOT / "results" / "models" / "asap-aes" / "positive_linear_coefficients.csv"
DEFAULT_AES_PROCESSED_SCORE_PATH = PROJECT_ROOT / "results" / "processed" / "asap-aes" / "train.csv"
DEFAULT_AES_METRIC_PATH = PROJECT_ROOT / "results" / "metrics" / "asap-aes" / "metrics.csv"
DEFAULT_AES_DESCRIPTION_ZIP_PATH = PROJECT_ROOT / "data" / "asap-aes" / "Essay_Set_Descriptions.zip"


SUPPORTED_LIVE_SYSTEM = "asag_improved_weak_label_baseline"
MODEL_TYPE = "non_negative_linear_regression"
SUPPORTED_AES_LIVE_SYSTEM = "aes_weak_label_baseline"
DOCX_NAME_RE = re.compile(r"Data Set #(\d+)--ReadMeFirst\.docx$")
AES_DOCX_NAME_RE = re.compile(r"Essay_Set_Descriptions/Essay Set #(\d+)--ReadMeFirst\.docx$")
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Streamlit demo artifacts for ASAP-SAS.")
    parser.add_argument("--artifact-dir", type=Path, default=DEMO_ARTIFACT_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--weak-label-path", type=Path, default=DEFAULT_WEAK_LABEL_PATH)
    parser.add_argument("--coefficient-path", type=Path, default=DEFAULT_COEFFICIENT_PATH)
    parser.add_argument("--score-range-path", type=Path, default=DEFAULT_SCORE_RANGE_PATH)
    parser.add_argument("--main-results-path", type=Path, default=DEFAULT_MAIN_RESULTS_PATH)
    parser.add_argument("--weak-quality-path", type=Path, default=DEFAULT_WEAK_QUALITY_PATH)
    parser.add_argument("--per-prompt-results-path", type=Path, default=DEFAULT_PER_PROMPT_RESULTS_PATH)
    parser.add_argument("--metric-path", type=Path, default=DEFAULT_METRIC_PATH)
    parser.add_argument("--description-zip-path", type=Path, default=DEFAULT_DESCRIPTION_ZIP_PATH)
    parser.add_argument("--aes-processed-dir", type=Path, default=DEFAULT_AES_PROCESSED_DIR)
    parser.add_argument("--aes-feature-dir", type=Path, default=DEFAULT_AES_FEATURE_DIR)
    parser.add_argument("--aes-coefficient-path", type=Path, default=DEFAULT_AES_COEFFICIENT_PATH)
    parser.add_argument("--aes-metric-path", type=Path, default=DEFAULT_AES_METRIC_PATH)
    parser.add_argument("--aes-description-zip-path", type=Path, default=DEFAULT_AES_DESCRIPTION_ZIP_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = prepare_demo_artifacts(
        artifact_dir=args.artifact_dir,
        processed_dir=args.processed_dir,
        feature_dir=args.feature_dir,
        weak_label_path=args.weak_label_path,
        coefficient_path=args.coefficient_path,
        score_range_path=args.score_range_path,
        main_results_path=args.main_results_path,
        weak_quality_path=args.weak_quality_path,
        per_prompt_results_path=args.per_prompt_results_path,
        metric_path=args.metric_path,
        description_zip_path=args.description_zip_path,
        aes_processed_dir=args.aes_processed_dir,
        aes_feature_dir=args.aes_feature_dir,
        aes_coefficient_path=args.aes_coefficient_path,
        aes_metric_path=args.aes_metric_path,
        aes_description_zip_path=args.aes_description_zip_path,
    )

    print("Demo artifacts ready.")
    print(f"- manifest: {outputs['manifest_path']}")
    print(f"- ASAG replay examples: {outputs['asag_replay_path']}")
    print(f"- AES replay examples: {outputs['aes_replay_path']}")
    print("- live ASAG questions:", ", ".join(str(qid) for qid in outputs["questions"]))
    print("- live AES essay sets:", ", ".join(str(set_id) for set_id in outputs["essay_sets"]))


def prepare_demo_artifacts(
    artifact_dir: Path = DEMO_ARTIFACT_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    weak_label_path: Path = DEFAULT_WEAK_LABEL_PATH,
    coefficient_path: Path = DEFAULT_COEFFICIENT_PATH,
    score_range_path: Path = DEFAULT_SCORE_RANGE_PATH,
    main_results_path: Path = DEFAULT_MAIN_RESULTS_PATH,
    weak_quality_path: Path = DEFAULT_WEAK_QUALITY_PATH,
    per_prompt_results_path: Path = DEFAULT_PER_PROMPT_RESULTS_PATH,
    metric_path: Path = DEFAULT_METRIC_PATH,
    description_zip_path: Path = DEFAULT_DESCRIPTION_ZIP_PATH,
    aes_processed_dir: Path = DEFAULT_AES_PROCESSED_DIR,
    aes_feature_dir: Path = DEFAULT_AES_FEATURE_DIR,
    aes_coefficient_path: Path = DEFAULT_AES_COEFFICIENT_PATH,
    aes_metric_path: Path = DEFAULT_AES_METRIC_PATH,
    aes_description_zip_path: Path = DEFAULT_AES_DESCRIPTION_ZIP_PATH,
) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    processed_splits = ensure_processed_splits(processed_dir)
    feature_splits = ensure_feature_splits(processed_splits, feature_dir)
    weak_labels = load_weak_labels(weak_label_path)
    weak_label_variant = infer_weak_label_variant(weak_labels)
    score_ranges = load_score_ranges(score_range_path, processed_splits["train"])
    question_descriptions = load_question_descriptions(description_zip_path)

    coefficient_df, coefficient_source = load_or_fit_coefficients(
        coefficient_path=coefficient_path,
        weak_labels=weak_labels,
        processed_splits=processed_splits,
        feature_splits=feature_splits,
    )

    per_question_metrics = load_optional_csv(metric_path)
    final_report_results = load_optional_csv(main_results_path)
    weak_quality = load_optional_csv(weak_quality_path)
    per_prompt_results = load_optional_csv(per_prompt_results_path)

    created_at = datetime.now(timezone.utc).isoformat()
    question_ids = sorted(int(question_id) for question_id in score_ranges["question_id"].tolist())
    artifact_paths = {}
    question_summaries = []

    for question_id in question_ids:
        artifact = build_question_artifact(
            question_id=question_id,
            coefficient_df=coefficient_df,
            score_ranges=score_ranges,
            processed_splits=processed_splits,
            weak_label_variant=weak_label_variant,
            created_at=created_at,
            coefficient_source=coefficient_source,
            question_description=question_descriptions.get(question_id, {}),
        )
        artifact_path = artifact_dir / f"asag_question_{question_id}.json"
        write_json(artifact_path, artifact)
        artifact_paths[str(question_id)] = str(artifact_path.relative_to(PROJECT_ROOT))
        question_summaries.append(
            {
                "question_id": question_id,
                "score_min": artifact["score_min"],
                "score_max": artifact["score_max"],
                "train_rows": artifact["training_rows"],
                "val_rows": artifact["validation_rows"],
                "public_test_rows": artifact["public_test_rows"],
                "private_test_rows": artifact["private_test_rows"],
                "weak_label_variant": artifact["weak_label_variant"],
            }
        )

    replay_examples = build_replay_examples(
        processed_splits=processed_splits,
        feature_splits=feature_splits,
        final_report_prediction_dir=DEFAULT_FINAL_REPORT_PREDICTION_DIR,
        fallback_prediction_dir=DEFAULT_RESULTS_PREDICTION_DIR,
    )
    replay_path = artifact_dir / "sample_replay_examples.csv"
    replay_examples.to_csv(replay_path, index=False)

    manifest = build_manifest(
        created_at=created_at,
        artifact_dir=artifact_dir,
        artifact_paths=artifact_paths,
        question_summaries=question_summaries,
        coefficient_source=coefficient_source,
        weak_label_variant=weak_label_variant,
        final_report_results=final_report_results,
        weak_quality=weak_quality,
        per_question_metrics=per_question_metrics,
        per_prompt_results=per_prompt_results,
        replay_path=replay_path,
        score_ranges=score_ranges,
    )

    aes_bundle = prepare_aes_demo_artifacts(
        artifact_dir=artifact_dir,
        processed_dir=aes_processed_dir,
        feature_dir=aes_feature_dir,
        coefficient_path=aes_coefficient_path,
        final_report_results=final_report_results,
        per_prompt_results=per_prompt_results,
        metric_path=aes_metric_path,
        description_zip_path=aes_description_zip_path,
        created_at=created_at,
    )
    manifest["additional_tasks"] = {"AES": aes_bundle["manifest"]}
    manifest_path = artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)

    return {
        "manifest_path": manifest_path,
        "asag_replay_path": replay_path,
        "aes_replay_path": aes_bundle["replay_path"],
        "questions": question_ids,
        "essay_sets": aes_bundle["essay_sets"],
    }


def ensure_processed_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return load_processed_splits(processed_dir)


def ensure_feature_splits(
    processed_splits: dict[str, pd.DataFrame],
    feature_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_names = ["train", "val", "public_test", "private_test"]
    feature_splits: dict[str, pd.DataFrame] = {}
    missing = []

    for split_name in split_names:
        path = feature_dir / f"{split_name}_features.csv"
        if path.exists():
            feature_splits[split_name] = pd.read_csv(path)
        elif split_name in processed_splits:
            missing.append(split_name)

    if missing:
        extracted = extract_feature_splits(
            {split_name: processed_splits[split_name] for split_name in missing}
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in extracted.items():
            path = feature_dir / f"{split_name}_features.csv"
            df.to_csv(path, index=False)
            feature_splits[split_name] = df

    return feature_splits


def load_weak_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing weak-label file: {path}. Expected a train-only ASAP-SAS weak-label file."
        )
    df = pd.read_csv(path)
    required = {"sample_id", "question_id", "weak_label_normalized"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Weak-label file is missing required columns: {missing}")
    return df


def infer_weak_label_variant(weak_labels: pd.DataFrame) -> str:
    if "method" in weak_labels.columns and weak_labels["method"].notna().any():
        return str(weak_labels["method"].dropna().iloc[0])
    return "UNKNOWN"


def load_score_ranges(score_range_path: Path, train_df: pd.DataFrame) -> pd.DataFrame:
    if score_range_path.exists():
        score_ranges = pd.read_csv(score_range_path)
        required = {"question_id", "score_min", "score_max"}
        missing = sorted(required.difference(score_ranges.columns))
        if missing:
            raise ValueError(f"Score-range file is missing required columns: {missing}")
        return score_ranges.sort_values("question_id").reset_index(drop=True)

    derived = (
        train_df.groupby("question_id", as_index=False)
        .agg(score_min=("score_raw", "min"), score_max=("score_raw", "max"))
        .sort_values("question_id")
        .reset_index(drop=True)
    )
    return derived


def load_or_fit_coefficients(
    coefficient_path: Path,
    weak_labels: pd.DataFrame,
    processed_splits: dict[str, pd.DataFrame],
    feature_splits: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    if coefficient_path.exists():
        coefficient_df = pd.read_csv(coefficient_path)
        validate_coefficient_df(coefficient_df)
        return coefficient_df, str(coefficient_path.relative_to(PROJECT_ROOT))

    train_features = feature_splits["train"].merge(
        weak_labels[["sample_id", "question_id", "weak_label_normalized"]],
        on=["sample_id", "question_id"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for question_id, group in train_features.groupby("question_id", sort=True):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(group[FEATURE_COLUMNS])
        model = LinearRegression(positive=True)
        model.fit(x_scaled, group["weak_label_normalized"])
        for feature_name, coefficient, mean, scale in zip(
            FEATURE_COLUMNS,
            model.coef_,
            scaler.mean_,
            scaler.scale_,
        ):
            rows.append(
                {
                    "question_id": int(question_id),
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "scaler_mean": float(mean),
                    "scaler_scale": float(scale),
                    "intercept": float(model.intercept_),
                }
            )
    coefficient_df = pd.DataFrame(rows)
    validate_coefficient_df(coefficient_df)
    return coefficient_df, "refit_from_processed_features_and_weak_labels"


def validate_coefficient_df(coefficient_df: pd.DataFrame) -> None:
    required = {
        "question_id",
        "feature",
        "coefficient",
        "scaler_mean",
        "scaler_scale",
        "intercept",
    }
    missing = sorted(required.difference(coefficient_df.columns))
    if missing:
        raise ValueError(f"Coefficient summary is missing required columns: {missing}")

    for question_id, group in coefficient_df.groupby("question_id", sort=True):
        features = group["feature"].tolist()
        if features != FEATURE_COLUMNS:
            raise ValueError(
                f"Question {question_id} coefficients do not match expected feature order."
            )


def build_question_artifact(
    question_id: int,
    coefficient_df: pd.DataFrame,
    score_ranges: pd.DataFrame,
    processed_splits: dict[str, pd.DataFrame],
    weak_label_variant: str,
    created_at: str,
    coefficient_source: str,
    question_description: dict[str, str],
) -> dict[str, object]:
    group = coefficient_df[coefficient_df["question_id"] == question_id].copy()
    group = group.set_index("feature").loc[FEATURE_COLUMNS].reset_index()
    score_row = score_ranges[score_ranges["question_id"] == question_id]
    if score_row.empty:
        raise ValueError(f"Missing score range for question {question_id}")

    score_min = int(score_row["score_min"].iloc[0])
    score_max = int(score_row["score_max"].iloc[0])
    intercept = float(group["intercept"].iloc[0])

    artifact = OrderedDict(
        task="ASAG",
        question_id=int(question_id),
        question_label=question_description.get("question_label", f"ASAP-SAS Question {question_id}"),
        prompt_title=question_description.get("prompt_title", f"Question {question_id}"),
        prompt_text=question_description.get("prompt_text", ""),
        prompt_context=question_description.get("prompt_context", ""),
        question_text=question_description.get("question_text", question_description.get("prompt_text", "")),
        response_type=question_description.get("response_type", "UNKNOWN"),
        grade_level=question_description.get("grade_level", "UNKNOWN"),
        subject=question_description.get("subject", "UNKNOWN"),
        model_type=MODEL_TYPE,
        feature_names=list(FEATURE_COLUMNS),
        coef=[float(value) for value in group["coefficient"].tolist()],
        intercept=intercept,
        scaler_mean=[float(value) for value in group["scaler_mean"].tolist()],
        scaler_scale=[float(value) if float(value) != 0 else 1.0 for value in group["scaler_scale"].tolist()],
        score_min=score_min,
        score_max=score_max,
        weak_label_variant=weak_label_variant,
        training_split_used="results/processed/asap-sas/train.csv",
        training_rows=count_question_rows(processed_splits, "train", question_id),
        validation_rows=count_question_rows(processed_splits, "val", question_id),
        public_test_rows=count_question_rows(processed_splits, "public_test", question_id),
        private_test_rows=count_question_rows(processed_splits, "private_test", question_id),
        artifact_created_at=created_at,
        provenance={
            "coefficient_source": coefficient_source,
            "processed_train": "results/processed/asap-sas/train.csv",
            "weak_labels": "results/weak_labels/asap-sas/train_signal_clustering.csv",
            "feature_source": "results/features/asap-sas/train_features.csv",
            "prediction_reference": "results/predictions/asap-sas/public_test_predictions.csv",
            "metric_reference": "results/metrics/asap-sas/metrics.csv",
        },
        created_by="src.prepare_demo_artifacts",
        gold_used_for_training=False,
    )
    return artifact


def count_question_rows(
    processed_splits: dict[str, pd.DataFrame],
    split_name: str,
    question_id: int,
) -> int:
    if split_name not in processed_splits:
        return 0
    df = processed_splits[split_name]
    return int((df["question_id"] == question_id).sum())


def load_question_descriptions(description_zip_path: Path) -> dict[int, dict[str, str]]:
    if not description_zip_path.exists():
        return {}

    descriptions: dict[int, dict[str, str]] = {}
    with ZipFile(description_zip_path) as outer_zip:
        for name in outer_zip.namelist():
            match = DOCX_NAME_RE.search(name)
            if not match:
                continue
            question_id = int(match.group(1))
            lines = extract_docx_lines(outer_zip.read(name))
            descriptions[question_id] = parse_question_description(question_id, lines)
    return descriptions


def extract_docx_lines(docx_bytes: bytes) -> list[str]:
    with ZipFile(BytesIO(docx_bytes)) as docx_zip:
        xml_bytes = docx_zip.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    lines = []
    for paragraph in root.findall(".//w:p", DOCX_NS):
        line = "".join((node.text or "") for node in paragraph.findall(".//w:t", DOCX_NS)).strip()
        if line:
            lines.append(line)
    return lines


def parse_question_description(question_id: int, lines: list[str]) -> dict[str, str]:
    prompt_idx = next((idx for idx, line in enumerate(lines) if line.startswith("Prompt")), None)
    prompt_title = f"Question {question_id}"
    prompt_text = ""
    prompt_context = ""
    question_text = ""
    response_type = extract_following_value(lines, "Type of response:")
    if prompt_idx is not None:
        prompt_line = lines[prompt_idx]
        if "—" in prompt_line:
            prompt_title = prompt_line.split("—", 1)[1].strip()
        else:
            prompt_title = prompt_line.replace("Prompt", "").strip(" :-") or prompt_title
        prompt_context, question_text, prompt_text = extract_asag_prompt_parts(
            lines,
            prompt_idx=prompt_idx,
            response_type=response_type,
        )

    return {
        "question_label": f"ASAP-SAS Question {question_id}",
        "prompt_title": prompt_title,
        "prompt_text": prompt_text,
        "prompt_context": prompt_context,
        "question_text": question_text,
        "response_type": response_type,
        "grade_level": extract_following_value(lines, "Grade level:"),
        "subject": extract_following_value(lines, "Subject:"),
    }


def extract_following_value(lines: list[str], label: str) -> str:
    try:
        idx = lines.index(label)
    except ValueError:
        return "UNKNOWN"
    return lines[idx + 1] if idx + 1 < len(lines) else "UNKNOWN"


def extract_asag_prompt_parts(
    lines: list[str],
    prompt_idx: int,
    response_type: str,
) -> tuple[str, str, str]:
    end_markers = (
        "Rubric for ",
        "Scoring Rubric",
        "Rubric:",
        "Possible Correct Responses:",
        "Key Elements:",
    )
    end_idx = len(lines)
    for idx in range(prompt_idx + 1, len(lines)):
        if any(lines[idx].startswith(marker) for marker in end_markers):
            end_idx = idx
            break

    section = lines[prompt_idx + 1 : end_idx]
    question_lines = [line for line in section if is_priority_prompt_line(line)]
    context_line = next((line for line in section if is_context_prompt_line(line)), "")

    if question_lines:
        candidates = question_lines
        if context_line and context_line not in candidates:
            candidates = [context_line] + candidates
    else:
        candidates = [line for line in section if is_prompt_candidate(line)]
        if not candidates:
            candidates = [line for line in section[:4] if line and not line.endswith(":")]

    prompt_text = " ".join(candidates[:5]).strip()
    question_text = " ".join(question_lines[:5]).strip() or prompt_text

    prompt_context = ""
    if "source dependent" in response_type.lower():
        prompt_context = extract_asag_source_context(lines, prompt_idx)

    return prompt_context, question_text, prompt_text


def extract_asag_source_context(lines: list[str], prompt_idx: int) -> str:
    start_idx = find_asag_content_start(lines)
    if start_idx >= prompt_idx:
        return ""
    context_lines = [line for line in lines[start_idx:prompt_idx] if keep_context_line(line)]
    return " ".join(context_lines).strip()


def find_asag_content_start(lines: list[str]) -> int:
    labels = ["Rubric range:", "Final score:", "Average length of responses:"]
    for label in labels:
        try:
            idx = lines.index(label)
        except ValueError:
            continue
        step = 2 if label != "Final score:" else 1
        return min(idx + step + 1, len(lines))
    return 0


def keep_context_line(line: str) -> bool:
    if not line or line.endswith(":"):
        return False
    if line.startswith("Prompt"):
        return False
    if line.startswith("Rubric for "):
        return False
    return True


def is_prompt_candidate(line: str) -> bool:
    lower = line.lower()
    if not line or line.endswith(":"):
        return False
    if re.fullmatch(r"[\d.\-–°()A-Za-z]{1,8}", line):
        return False
    if line in {"Sample", "Data", "Procedure", "Materials", "Hypothesis"}:
        return False
    if len(line) < 20:
        return False
    if is_excluded_prompt_line(line):
        return False
    return is_priority_prompt_line(line) or line.endswith(".")


def is_priority_prompt_line(line: str) -> bool:
    lower = line.lower()
    keywords = (
        "question:",
        "describe",
        "explain",
        "list and describe",
        "after reading",
        "starting with",
        "use the results",
        "choose a paint color",
        "what is the effect",
        "would need",
        "in your description",
    )
    return any(keyword in lower for keyword in keywords)


def is_context_prompt_line(line: str) -> bool:
    lower = line.lower()
    starters = (
        "a group of students",
        "brandi and jerry",
        "starting with",
    )
    return any(lower.startswith(starter) for starter in starters)


def is_excluded_prompt_line(line: str) -> bool:
    lower = line.lower()
    excluded_starts = (
        "procedure",
        "materials",
        "hypothesis",
        "data",
        "sample",
        "put ",
        "place ",
        "repeat ",
        "make sure ",
        "after 10 minutes",
        "turn off ",
        "calculate and record ",
    )
    return any(lower.startswith(prefix) for prefix in excluded_starts)


def build_replay_examples(
    processed_splits: dict[str, pd.DataFrame],
    feature_splits: dict[str, pd.DataFrame],
    final_report_prediction_dir: Path,
    fallback_prediction_dir: Path,
) -> pd.DataFrame:
    frames = []
    prediction_specs = [
        ("val", "asag_improved_weak_label_baseline_val.csv", "val_predictions.csv"),
        (
            "public_test",
            "asag_improved_weak_label_baseline_public_test.csv",
            "public_test_predictions.csv",
        ),
    ]

    for split_name, final_report_name, fallback_name in prediction_specs:
        prediction_path = final_report_prediction_dir / final_report_name
        source_label = str(prediction_path.relative_to(PROJECT_ROOT))
        if not prediction_path.exists():
            prediction_path = fallback_prediction_dir / fallback_name
            source_label = str(prediction_path.relative_to(PROJECT_ROOT))
        if not prediction_path.exists():
            continue

        predictions = pd.read_csv(prediction_path)
        if split_name not in feature_splits or split_name not in processed_splits:
            continue

        feature_df = feature_splits[split_name].copy()
        processed_df = processed_splits[split_name][["sample_id", "question_id", "score_raw"]].copy()
        merged = predictions.merge(
            feature_df,
            on=["sample_id", "question_id", "split"],
            how="left",
            validate="one_to_one",
        ).merge(
            processed_df,
            on=["sample_id", "question_id"],
            how="left",
            suffixes=("", "_processed"),
            validate="one_to_one",
        )

        if "score_raw_processed" in merged.columns:
            merged["score_raw"] = merged["score_raw"].fillna(merged["score_raw_processed"])
            merged = merged.drop(columns=["score_raw_processed"])

        merged["absolute_error"] = (merged["pred_score"] - merged["score_raw"]).abs()
        merged["signed_error"] = merged["pred_score"] - merged["score_raw"]
        merged["error_type"] = merged.apply(classify_error_type, axis=1)
        merged["source_system_id"] = SUPPORTED_LIVE_SYSTEM
        merged["prediction_source"] = source_label

        selected = []
        for question_id, group in merged.groupby("question_id", sort=True):
            selected.extend(select_replay_rows(group))

        frames.append(pd.DataFrame(selected))

    if not frames:
        return pd.DataFrame(
            columns=[
                "split",
                "sample_id",
                "question_id",
                "score_raw",
                "score_min",
                "score_max",
                "pred_score",
                "pred_score_rounded",
                "weak_prediction_clipped",
                "absolute_error",
                "signed_error",
                "error_type",
                "selection_reason",
                "source_system_id",
                "prediction_source",
            ]
            + FEATURE_COLUMNS
        )

    replay_examples = pd.concat(frames, ignore_index=True)
    replay_examples = replay_examples.sort_values(["split", "question_id", "selection_reason", "sample_id"])
    return replay_examples.reset_index(drop=True)


def classify_error_type(row: pd.Series) -> str:
    if pd.isna(row.get("score_raw")):
        return "gold_unavailable"
    gold = float(row["score_raw"])
    rounded = float(row["pred_score_rounded"])
    if rounded == gold:
        return "correct_after_rounding"
    if float(row["pred_score"]) < gold:
        return "underprediction"
    return "overprediction"


def select_replay_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    picks: list[pd.Series] = []

    def add_row(row: pd.Series | None, reason: str) -> None:
        if row is None:
            return
        sample_id = int(row["sample_id"])
        if any(int(existing["sample_id"]) == sample_id for existing in picks):
            return
        row = row.copy()
        row["selection_reason"] = reason
        picks.append(row)

    exact = group[group["pred_score_rounded"] == group["score_raw"]]
    if not exact.empty:
        add_row(exact.sort_values("absolute_error").iloc[0], "closest_correct")

    near = group[np.abs(group["pred_score_rounded"] - group["score_raw"]) <= 1]
    if not near.empty:
        add_row(near.sort_values("absolute_error").iloc[0], "near_miss")

    under = group[group["signed_error"] < 0]
    if not under.empty:
        add_row(under.sort_values("signed_error").iloc[0], "severe_underprediction")

    over = group[group["signed_error"] > 0]
    if not over.empty:
        add_row(over.sort_values("signed_error", ascending=False).iloc[0], "severe_overprediction")

    if not picks:
        add_row(group.sort_values("absolute_error").iloc[0], "closest_available")

    output_rows = []
    for row in picks:
        output_rows.append(
            {
                "split": row["split"],
                "sample_id": int(row["sample_id"]),
                "question_id": int(row["question_id"]),
                "score_raw": float(row["score_raw"]),
                "score_min": int(row["score_min"]),
                "score_max": int(row["score_max"]),
                "pred_score": float(row["pred_score"]),
                "pred_score_rounded": float(row["pred_score_rounded"]),
                "weak_prediction_clipped": float(row["weak_prediction_clipped"]),
                "absolute_error": float(row["absolute_error"]),
                "signed_error": float(row["signed_error"]),
                "error_type": str(row["error_type"]),
                "selection_reason": str(row["selection_reason"]),
                "source_system_id": str(row["source_system_id"]),
                "prediction_source": str(row["prediction_source"]),
                **{feature: float(row[feature]) for feature in FEATURE_COLUMNS},
            }
        )
    return output_rows


def build_manifest(
    created_at: str,
    artifact_dir: Path,
    artifact_paths: dict[str, str],
    question_summaries: list[dict[str, object]],
    coefficient_source: str,
    weak_label_variant: str,
    final_report_results: pd.DataFrame | None,
    weak_quality: pd.DataFrame | None,
    per_question_metrics: pd.DataFrame | None,
    per_prompt_results: pd.DataFrame | None,
    replay_path: Path,
    score_ranges: pd.DataFrame,
) -> dict[str, object]:
    supported_results = []
    if final_report_results is not None:
        filtered = final_report_results[
            (final_report_results["task"] == "ASAG")
            & (final_report_results["supported_local"] == True)
            & (
                final_report_results["system_id"].isin(
                    [SUPPORTED_LIVE_SYSTEM, "asag_sbert_hybrid_supported_candidate"]
                )
            )
        ].copy()
        supported_results = filtered.to_dict(orient="records")

    weak_quality_excerpt = []
    if weak_quality is not None:
        filtered = weak_quality[weak_quality["task"] == "ASAG"].copy()
        weak_quality_excerpt = filtered.head(12).to_dict(orient="records")

    per_question_metric_excerpt = []
    if per_question_metrics is not None:
        per_question_metric_excerpt = per_question_metrics.to_dict(orient="records")

    per_prompt_excerpt = []
    if per_prompt_results is not None:
        filtered = per_prompt_results[
            (per_prompt_results["task"] == "ASAG")
            & (per_prompt_results["system_id"] == SUPPORTED_LIVE_SYSTEM)
        ].copy()
        per_prompt_excerpt = filtered.to_dict(orient="records")

    return {
        "task": "ASAG",
        "supported_live_system": SUPPORTED_LIVE_SYSTEM,
        "artifact_builder": "src.prepare_demo_artifacts",
        "created_at_utc": created_at,
        "artifact_dir": str(artifact_dir.relative_to(PROJECT_ROOT)),
        "available_questions": sorted(int(question_id) for question_id in artifact_paths.keys()),
        "live_inference_available": True,
        "sample_replay_available": replay_path.exists(),
        "gold_used_for_training": False,
        "model_type": MODEL_TYPE,
        "weak_label_variant": weak_label_variant,
        "feature_names": list(FEATURE_COLUMNS),
        "question_score_ranges": score_ranges.to_dict(orient="records"),
        "question_summaries": question_summaries,
        "artifact_files": artifact_paths,
        "replay_example_file": str(replay_path.relative_to(PROJECT_ROOT)),
        "source_files": {
            "coefficient_source": coefficient_source,
            "processed_train": "results/processed/asap-sas/train.csv",
            "processed_val": "results/processed/asap-sas/val.csv",
            "processed_public_test": "results/processed/asap-sas/public_test.csv",
            "processed_private_test": "results/processed/asap-sas/private_test.csv",
            "weak_labels": "results/weak_labels/asap-sas/train_signal_clustering.csv",
            "features_train": "results/features/asap-sas/train_features.csv",
            "features_val": "results/features/asap-sas/val_features.csv",
            "features_public_test": "results/features/asap-sas/public_test_features.csv",
            "prediction_reference": "results/predictions/asap-sas/public_test_predictions.csv",
            "metrics_reference": "results/metrics/asap-sas/metrics.csv",
            "main_results_reference": "results/final_report/tables/main_results_macro.csv",
            "weak_quality_reference": "results/final_report/tables/weak_label_quality.csv",
        },
        "supported_results_excerpt": supported_results,
        "weak_label_quality_excerpt": weak_quality_excerpt,
        "per_question_metrics_excerpt": per_question_metric_excerpt,
        "per_prompt_results_excerpt": per_prompt_excerpt,
        "notes": [
            "Gold labels are not used as training targets for live demo artifacts.",
            "ASAP-SAS private-test labels are unavailable and are not reported.",
            "Sample replay mode replays file-backed validation/public-test predictions only.",
        ],
    }


def prepare_aes_demo_artifacts(
    artifact_dir: Path,
    processed_dir: Path,
    feature_dir: Path,
    coefficient_path: Path,
    final_report_results: pd.DataFrame | None,
    per_prompt_results: pd.DataFrame | None,
    metric_path: Path,
    description_zip_path: Path,
    created_at: str,
) -> dict[str, object]:
    processed_splits = load_aes_processed_splits(processed_dir)
    feature_splits = ensure_aes_feature_splits(processed_splits, feature_dir)
    coefficient_df = pd.read_csv(coefficient_path)
    validate_aes_coefficient_df(coefficient_df)
    essay_descriptions = load_aes_descriptions(description_zip_path)
    score_ranges = derive_aes_score_ranges(processed_splits["train"])
    per_set_metrics = load_optional_csv(metric_path)

    artifact_paths = {}
    set_summaries = []
    essay_sets = sorted(int(essay_set) for essay_set in score_ranges["essay_set"].tolist())
    for essay_set in essay_sets:
        artifact = build_aes_artifact(
            essay_set=essay_set,
            coefficient_df=coefficient_df,
            processed_splits=processed_splits,
            score_ranges=score_ranges,
            created_at=created_at,
            essay_description=essay_descriptions.get(essay_set, {}),
        )
        artifact_path = artifact_dir / f"aes_set_{essay_set}.json"
        write_json(artifact_path, artifact)
        artifact_paths[str(essay_set)] = str(artifact_path.relative_to(PROJECT_ROOT))
        set_summaries.append(
            {
                "essay_set": essay_set,
                "score_min": artifact["score_min"],
                "score_max": artifact["score_max"],
                "train_rows": artifact["training_rows"],
                "val_rows": artifact["validation_rows"],
                "test_rows": artifact["test_rows"],
            }
        )

    replay_examples = build_aes_replay_examples(
        processed_splits=processed_splits,
        feature_splits=feature_splits,
        final_report_prediction_dir=DEFAULT_FINAL_REPORT_PREDICTION_DIR,
        fallback_prediction_dir=PROJECT_ROOT / "results" / "predictions" / "asap-aes",
    )
    replay_path = artifact_dir / "aes_sample_replay_examples.csv"
    replay_examples.to_csv(replay_path, index=False)

    supported_results = []
    if final_report_results is not None:
        filtered = final_report_results[
            (final_report_results["task"] == "AES")
            & (final_report_results["supported_local"] == True)
            & (
                final_report_results["system_id"].isin(
                    [SUPPORTED_AES_LIVE_SYSTEM, "aes_set6_8_feature_selection"]
                )
            )
        ].copy()
        supported_results = filtered.to_dict(orient="records")

    per_prompt_excerpt = []
    if per_prompt_results is not None:
        filtered = per_prompt_results[
            (per_prompt_results["task"] == "AES")
            & (per_prompt_results["system_id"] == SUPPORTED_AES_LIVE_SYSTEM)
        ].copy()
        per_prompt_excerpt = filtered.to_dict(orient="records")

    manifest = {
        "task": "AES",
        "supported_live_system": SUPPORTED_AES_LIVE_SYSTEM,
        "artifact_builder": "src.prepare_demo_artifacts",
        "created_at_utc": created_at,
        "artifact_dir": str(artifact_dir.relative_to(PROJECT_ROOT)),
        "available_sets": essay_sets,
        "live_inference_available": True,
        "sample_replay_available": replay_path.exists(),
        "gold_used_for_training": False,
        "model_type": MODEL_TYPE,
        "weak_label_variant": "signal_clustering_length_similarity",
        "feature_names": list(AES_FEATURE_COLUMNS),
        "unit_score_ranges": score_ranges.to_dict(orient="records"),
        "unit_summaries": set_summaries,
        "artifact_files": artifact_paths,
        "replay_example_file": str(replay_path.relative_to(PROJECT_ROOT)),
        "source_files": {
            "coefficient_source": str(coefficient_path.relative_to(PROJECT_ROOT)),
            "processed_train": "results/processed/asap-aes/train.csv",
            "processed_val": "results/processed/asap-aes/val.csv",
            "processed_test": "results/processed/asap-aes/test.csv",
            "features_train": "results/features/asap-aes/train_features.csv",
            "features_val": "results/features/asap-aes/val_features.csv",
            "features_test": "results/features/asap-aes/test_features.csv",
            "prediction_reference": "results/predictions/asap-aes/test_predictions.csv",
            "metrics_reference": "results/metrics/asap-aes/metrics.csv",
            "main_results_reference": "results/final_report/tables/main_results_macro.csv",
        },
        "supported_results_excerpt": supported_results,
        "per_unit_metrics_excerpt": per_set_metrics.to_dict(orient="records") if per_set_metrics is not None else [],
        "per_prompt_results_excerpt": per_prompt_excerpt,
        "notes": [
            "Gold labels are not used as training targets for live demo artifacts.",
            "Sample replay mode replays file-backed validation/test predictions only.",
            "AES demo support uses the same local artifact-packaging flow as the ASAG demo.",
        ],
    }
    return {"manifest": manifest, "replay_path": replay_path, "essay_sets": essay_sets}


def ensure_aes_feature_splits(
    processed_splits: dict[str, pd.DataFrame],
    feature_dir: Path,
) -> dict[str, pd.DataFrame]:
    split_names = ["train", "val", "test"]
    feature_splits: dict[str, pd.DataFrame] = {}
    missing = []
    for split_name in split_names:
        path = feature_dir / f"{split_name}_features.csv"
        if path.exists():
            feature_splits[split_name] = pd.read_csv(path)
        else:
            missing.append(split_name)
    if missing:
        extracted = extract_aes_feature_splits({name: processed_splits[name] for name in missing})
        feature_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in extracted.items():
            path = feature_dir / f"{split_name}_features.csv"
            df.to_csv(path, index=False)
            feature_splits[split_name] = df
    return feature_splits


def validate_aes_coefficient_df(coefficient_df: pd.DataFrame) -> None:
    required = {"essay_set", "feature", "coefficient", "scaler_mean", "scaler_scale", "intercept"}
    missing = sorted(required.difference(coefficient_df.columns))
    if missing:
        raise ValueError(f"AES coefficient summary is missing required columns: {missing}")


def derive_aes_score_ranges(train_df: pd.DataFrame) -> pd.DataFrame:
    return (
        train_df.groupby("essay_set", as_index=False)
        .agg(score_min=("score_min", "first"), score_max=("score_max", "first"))
        .sort_values("essay_set")
        .reset_index(drop=True)
    )


def build_aes_artifact(
    essay_set: int,
    coefficient_df: pd.DataFrame,
    processed_splits: dict[str, pd.DataFrame],
    score_ranges: pd.DataFrame,
    created_at: str,
    essay_description: dict[str, str],
) -> dict[str, object]:
    group = coefficient_df[coefficient_df["essay_set"] == essay_set].copy()
    group = group.set_index("feature").loc[AES_FEATURE_COLUMNS].reset_index()
    score_row = score_ranges[score_ranges["essay_set"] == essay_set]
    score_min = float(score_row["score_min"].iloc[0])
    score_max = float(score_row["score_max"].iloc[0])

    return OrderedDict(
        task="AES",
        essay_set=int(essay_set),
        question_label=f"ASAP-AES Essay Set {essay_set}",
        prompt_title=essay_description.get("prompt_title", f"Essay Set {essay_set}"),
        prompt_text=essay_description.get("prompt_text", ""),
        prompt_context=essay_description.get("prompt_context", ""),
        question_text=essay_description.get("question_text", essay_description.get("prompt_text", "")),
        response_type=essay_description.get("type_of_essay", "UNKNOWN"),
        grade_level=essay_description.get("grade_level", "UNKNOWN"),
        model_type=MODEL_TYPE,
        feature_names=list(AES_FEATURE_COLUMNS),
        coef=[float(value) for value in group["coefficient"].tolist()],
        intercept=float(group["intercept"].iloc[0]),
        scaler_mean=[float(value) for value in group["scaler_mean"].tolist()],
        scaler_scale=[float(value) if float(value) != 0 else 1.0 for value in group["scaler_scale"].tolist()],
        score_min=score_min,
        score_max=score_max,
        weak_label_variant="signal_clustering_length_similarity",
        training_split_used="results/processed/asap-aes/train.csv",
        training_rows=count_aes_rows(processed_splits, "train", essay_set),
        validation_rows=count_aes_rows(processed_splits, "val", essay_set),
        test_rows=count_aes_rows(processed_splits, "test", essay_set),
        artifact_created_at=created_at,
        provenance={
            "coefficient_source": "results/models/asap-aes/positive_linear_coefficients.csv",
            "processed_train": "results/processed/asap-aes/train.csv",
            "features_train": "results/features/asap-aes/train_features.csv",
            "prediction_reference": "results/predictions/asap-aes/test_predictions.csv",
            "metric_reference": "results/metrics/asap-aes/metrics.csv",
        },
        created_by="src.prepare_demo_artifacts",
        gold_used_for_training=False,
    )


def count_aes_rows(processed_splits: dict[str, pd.DataFrame], split_name: str, essay_set: int) -> int:
    if split_name not in processed_splits:
        return 0
    df = processed_splits[split_name]
    return int((df["essay_set"] == essay_set).sum())


def load_aes_descriptions(description_zip_path: Path) -> dict[int, dict[str, str]]:
    if not description_zip_path.exists():
        return {}
    descriptions: dict[int, dict[str, str]] = {}
    with ZipFile(description_zip_path) as outer_zip:
        for name in outer_zip.namelist():
            match = AES_DOCX_NAME_RE.search(name)
            if not match:
                continue
            essay_set = int(match.group(1))
            lines = extract_docx_lines(outer_zip.read(name))
            descriptions[essay_set] = parse_aes_description(essay_set, lines)
    return descriptions


def parse_aes_description(essay_set: int, lines: list[str]) -> dict[str, str]:
    prompt_idx = next((idx for idx, line in enumerate(lines) if line == "Prompt"), None)
    source_idx = next((idx for idx, line in enumerate(lines) if line == "Source Essay"), None)
    rubric_idx = next((idx for idx, line in enumerate(lines) if line.startswith("Rubric Guidelines")), len(lines))

    prompt_lines = lines[prompt_idx + 1 : rubric_idx] if prompt_idx is not None else []
    prompt_text = " ".join(prompt_lines).strip()

    context_lines: list[str] = []
    prompt_title = f"Essay Set {essay_set}"
    if source_idx is not None and prompt_idx is not None and source_idx < prompt_idx:
        context_lines = [line for line in lines[source_idx + 1 : prompt_idx] if keep_context_line(line)]
        if context_lines:
            prompt_title = context_lines[0][:120]

    return {
        "prompt_title": prompt_title,
        "prompt_text": prompt_text,
        "prompt_context": " ".join(context_lines).strip(),
        "question_text": prompt_text,
        "type_of_essay": extract_following_value(lines, "Type of essay:"),
        "grade_level": extract_following_value(lines, "Grade level:"),
    }


def build_aes_replay_examples(
    processed_splits: dict[str, pd.DataFrame],
    feature_splits: dict[str, pd.DataFrame],
    final_report_prediction_dir: Path,
    fallback_prediction_dir: Path,
) -> pd.DataFrame:
    frames = []
    prediction_specs = [
        ("val", "aes_weak_label_baseline_val.csv", "val_predictions.csv"),
        ("test", "aes_weak_label_baseline_test.csv", "test_predictions.csv"),
    ]
    for split_name, final_report_name, fallback_name in prediction_specs:
        prediction_path = final_report_prediction_dir / final_report_name
        source_label = str(prediction_path.relative_to(PROJECT_ROOT))
        if not prediction_path.exists():
            prediction_path = fallback_prediction_dir / fallback_name
            source_label = str(prediction_path.relative_to(PROJECT_ROOT))
        if not prediction_path.exists():
            continue

        predictions = pd.read_csv(prediction_path)
        feature_df = feature_splits[split_name].copy()
        processed_df = processed_splits[split_name][["essay_id", "essay_set", "gold_score"]].copy()
        merged = predictions.merge(
            feature_df,
            on=["essay_id", "essay_set", "split"],
            how="left",
            validate="one_to_one",
        ).merge(
            processed_df,
            on=["essay_id", "essay_set"],
            how="left",
            suffixes=("", "_processed"),
            validate="one_to_one",
        )
        if "gold_score_processed" in merged.columns:
            merged["gold_score"] = merged["gold_score"].fillna(merged["gold_score_processed"])
            merged = merged.drop(columns=["gold_score_processed"])

        merged["absolute_error"] = (merged["pred_score"] - merged["gold_score"]).abs()
        merged["signed_error"] = merged["pred_score"] - merged["gold_score"]
        merged["error_type"] = merged.apply(classify_aes_error_type, axis=1)
        merged["source_system_id"] = SUPPORTED_AES_LIVE_SYSTEM
        merged["prediction_source"] = source_label

        selected = []
        for essay_set, group in merged.groupby("essay_set", sort=True):
            selected.extend(select_aes_replay_rows(group))
        frames.append(pd.DataFrame(selected))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["split", "essay_set", "selection_reason", "essay_id"]).reset_index(drop=True)


def classify_aes_error_type(row: pd.Series) -> str:
    gold = float(row["gold_score"])
    rounded = float(row["pred_score_rounded"])
    if rounded == gold:
        return "correct_after_rounding"
    if float(row["pred_score"]) < gold:
        return "underprediction"
    return "overprediction"


def select_aes_replay_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    picks: list[pd.Series] = []

    def add_row(row: pd.Series | None, reason: str) -> None:
        if row is None:
            return
        essay_id = int(row["essay_id"])
        if any(int(existing["essay_id"]) == essay_id for existing in picks):
            return
        row = row.copy()
        row["selection_reason"] = reason
        picks.append(row)

    exact = group[group["pred_score_rounded"] == group["gold_score"]]
    if not exact.empty:
        add_row(exact.sort_values("absolute_error").iloc[0], "closest_correct")

    near = group[np.abs(group["pred_score_rounded"] - group["gold_score"]) <= 1]
    if not near.empty:
        add_row(near.sort_values("absolute_error").iloc[0], "near_miss")

    under = group[group["signed_error"] < 0]
    if not under.empty:
        add_row(under.sort_values("signed_error").iloc[0], "severe_underprediction")

    over = group[group["signed_error"] > 0]
    if not over.empty:
        add_row(over.sort_values("signed_error", ascending=False).iloc[0], "severe_overprediction")

    if not picks:
        add_row(group.sort_values("absolute_error").iloc[0], "closest_available")

    rows = []
    for row in picks:
        rows.append(
            {
                "split": row["split"],
                "essay_id": int(row["essay_id"]),
                "essay_set": int(row["essay_set"]),
                "gold_score": float(row["gold_score"]),
                "score_min": float(row["score_min"]),
                "score_max": float(row["score_max"]),
                "pred_score": float(row["pred_score"]),
                "pred_score_rounded": float(row["pred_score_rounded"]),
                "weak_prediction_clipped": float(row["weak_prediction_clipped"]),
                "absolute_error": float(row["absolute_error"]),
                "signed_error": float(row["signed_error"]),
                "error_type": str(row["error_type"]),
                "selection_reason": str(row["selection_reason"]),
                "source_system_id": str(row["source_system_id"]),
                "prediction_source": str(row["prediction_source"]),
                **{feature: float(row[feature]) for feature in AES_FEATURE_COLUMNS},
            }
        )
    return rows


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
