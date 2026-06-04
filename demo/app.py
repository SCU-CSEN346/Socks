from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.demo_logic import (  # noqa: E402
    build_plain_english_explanation,
    build_rounding_review_note,
    classify_score_band,
)
from src.aes_features import FEATURE_COLUMNS as AES_FEATURE_COLUMNS, extract_feature_frame as extract_aes_feature_frame  # noqa: E402
from src.asag_features import FEATURE_COLUMNS as ASAG_FEATURE_COLUMNS, extract_feature_frame as extract_asag_feature_frame  # noqa: E402


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "demo_config.json"
SAMPLE_INPUT_PATH = APP_DIR / "sample_inputs.json"
QUESTION_CONTEXT_PATH = APP_DIR / "question_context.json"
DEMO_ARTIFACT_DIR = REPO_ROOT / "demo_artifacts"
FINAL_REPORT_RESULTS_PATH = REPO_ROOT / "results" / "final_report" / "tables" / "main_results_macro.csv"
WEAK_QUALITY_PATH = REPO_ROOT / "results" / "final_report" / "tables" / "weak_label_quality.csv"
PREPARE_COMMAND = "python3 -m src.prepare_demo_artifacts"
REPORT_BUNDLE_COMMAND = "python3 -m src.build_final_report_bundle"
APP_COMMAND = "streamlit run demo/app.py"


FEATURE_LABELS = {
    "character_count": "Character count",
    "word_count": "Word count",
    "sentence_count": "Sentence count",
    "average_word_length": "Average word length",
    "average_sentence_length": "Average sentence length",
    "unique_word_count": "Unique word count",
    "type_token_ratio": "Type-token ratio",
    "long_word_count": "Long-word count",
    "digit_count": "Digit count",
    "punctuation_count": "Punctuation count",
    "uppercase_count": "Uppercase count",
    "stopword_ratio": "Stopword ratio",
    "short_answer_bin": "Short-answer bin",
    "medium_answer_bin": "Medium-answer bin",
    "long_answer_bin": "Long-answer bin",
    "paragraph_count": "Paragraph count",
}

CONTEXT_STATUS_LABELS = {
    "available": "Available",
    "missing_context": "Missing local context",
    "self_contained": "Self-contained prompt",
    "concept_helper": "Concept helper",
    "source_supported": "Source-supported",
}

RECOMMENDATION_PRIORITY = {
    "recommended": 0,
    "okay": 1,
    "avoid_for_live_demo": 2,
}

ERROR_TYPE_LABELS = {
    "all": "All error types",
    "correct_after_rounding": "Correct-ish after rounding",
    "near_miss": "Near miss",
    "underprediction": "Underprediction",
    "overprediction": "Overprediction",
    "closest_available": "Closest available",
}


st.set_page_config(
    page_title="Weakly Supervised Interpretable Grading Demo",
    page_icon="📝",
    layout="wide",
)


@st.cache_resource
def load_config() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_sample_inputs() -> dict[str, object]:
    with SAMPLE_INPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_question_contexts() -> dict[str, dict[str, dict[str, object]]]:
    if not QUESTION_CONTEXT_PATH.exists():
        return {}
    with QUESTION_CONTEXT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_demo_bundle() -> dict[str, object]:
    manifest_path = DEMO_ARTIFACT_DIR / "manifest.json"
    if not manifest_path.exists():
        return {
            "manifest_path": manifest_path,
            "manifest": None,
            "task_bundles": {},
        }

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    task_manifests = {"ASAG": extract_primary_task_manifest(manifest)}
    task_manifests.update(manifest.get("additional_tasks", {}))
    task_bundles = {
        task_name: build_task_bundle(task_manifest)
        for task_name, task_manifest in task_manifests.items()
        if task_manifest
    }

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "task_bundles": task_bundles,
    }


@st.cache_data
def load_main_results() -> pd.DataFrame:
    if FINAL_REPORT_RESULTS_PATH.exists():
        return pd.read_csv(FINAL_REPORT_RESULTS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_weak_quality() -> pd.DataFrame:
    if WEAK_QUALITY_PATH.exists():
        return pd.read_csv(WEAK_QUALITY_PATH)
    return pd.DataFrame()


def extract_primary_task_manifest(manifest: dict[str, object]) -> dict[str, object]:
    primary = dict(manifest)
    primary.pop("additional_tasks", None)
    return primary


def build_task_bundle(task_manifest: dict[str, object]) -> dict[str, object]:
    artifacts = {}
    for unit_id, relative_path in task_manifest.get("artifact_files", {}).items():
        path = REPO_ROOT / relative_path
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                artifacts[int(unit_id)] = json.load(handle)

    replay_path = REPO_ROOT / task_manifest.get("replay_example_file", "")
    replay_examples = pd.read_csv(replay_path) if replay_path.exists() else pd.DataFrame()
    return {
        "manifest": task_manifest,
        "artifacts": artifacts,
        "replay_examples": replay_examples,
        "live_available": bool(artifacts),
    }


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def task_unit_label(task_name: str) -> str:
    return "question" if task_name == "ASAG" else "essay set"


def build_context_payload(
    task_name: str,
    unit_id: int,
    artifact: dict[str, object],
    question_contexts: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    entry = question_contexts.get(task_name, {}).get(str(unit_id), {})
    context_status = str(entry.get("context_status", "available"))
    background_context = str(entry.get("background_context", "")).strip()
    artifact_context = str(artifact.get("prompt_context", "")).strip()
    if not background_context and context_status in {"available", "source_supported"}:
        background_context = artifact_context

    question_text = str(
        entry.get("question_text")
        or artifact.get("question_text")
        or artifact.get("prompt_text")
        or ""
    ).strip()
    response_should_do = str(entry.get("response_should_do", "")).strip()
    score_range_note = str(entry.get("score_range_note", "")).strip()
    if not score_range_note:
        if "score_min" in artifact and "score_max" in artifact:
            score_range_note = (
                f"This {task_unit_label(task_name)} uses a "
                f"{int(float(artifact['score_min']))} to {int(float(artifact['score_max']))} score range."
            )
        else:
            score_range_note = "Score range metadata is unavailable in the current local artifact."

    return {
        "display_name": str(entry.get("display_name", artifact.get("question_label", f"{task_name} {unit_id}"))),
        "context_status": context_status,
        "background_context": background_context,
        "question_text": question_text,
        "response_should_do": response_should_do,
        "score_range_note": score_range_note,
        "demo_recommendation": str(entry.get("demo_recommendation", "okay")),
        "context_warning": str(entry.get("context_warning", "")).strip(),
    }


def ordered_units(
    task_name: str,
    available_units: list[int],
    artifacts: dict[int, dict[str, object]],
    question_contexts: dict[str, dict[str, dict[str, object]]],
) -> list[int]:
    def sort_key(unit_id: int) -> tuple[int, int]:
        payload = build_context_payload(task_name, unit_id, artifacts.get(unit_id, {}), question_contexts)
        return (RECOMMENDATION_PRIORITY.get(payload["demo_recommendation"], 1), unit_id)

    return sorted(available_units, key=sort_key)


def format_unit_label(
    task_name: str,
    unit_id: int,
    artifacts: dict[int, dict[str, object]],
    question_contexts: dict[str, dict[str, dict[str, object]]],
) -> str:
    artifact = artifacts.get(unit_id, {})
    payload = build_context_payload(task_name, unit_id, artifact, question_contexts)
    prefix = "★ " if payload["demo_recommendation"] == "recommended" else ""
    base = f"ASAP-SAS Question {unit_id}" if task_name == "ASAG" else f"ASAP-AES Essay Set {unit_id}"
    if payload["demo_recommendation"] == "avoid_for_live_demo":
        return f"{prefix}{base} (context incomplete)"
    return f"{prefix}{base}"


def pretty_error_type(error_type: str) -> str:
    return ERROR_TYPE_LABELS.get(error_type, error_type.replace("_", " ").title())


def add_replay_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["display_error_type"] = output["error_type"].fillna("").astype(str)
    near_mask = output["selection_reason"].fillna("").astype(str).eq("near_miss")
    output.loc[near_mask, "display_error_type"] = "near_miss"
    fallback_mask = output["display_error_type"].eq("") & output["selection_reason"].notna()
    output.loc[fallback_mask, "display_error_type"] = output.loc[fallback_mask, "selection_reason"].astype(str)
    return output


def score_live_answer(task_name: str, unit_id: int, answer_text: str, artifact: dict[str, object]) -> dict[str, object]:
    cleaned = clean_text(answer_text)
    if task_name == "ASAG":
        feature_input = pd.DataFrame(
            [
                {
                    "sample_id": 0,
                    "question_id": int(unit_id),
                    "split": "live_demo",
                    "student_answer": cleaned,
                }
            ]
        )
        feature_row = extract_asag_feature_frame(feature_input).iloc[0].to_dict()
    else:
        feature_input = pd.DataFrame(
            [
                {
                    "essay_id": 0,
                    "essay_set": int(unit_id),
                    "split": "live_demo",
                    "essay": cleaned,
                }
            ]
        )
        feature_row = extract_aes_feature_frame(feature_input).iloc[0].to_dict()
    return score_from_features(task_name, unit_id, feature_row, artifact, mode_label="Live inference")


def score_from_features(
    task_name: str,
    unit_id: int,
    feature_row: dict[str, float],
    artifact: dict[str, object],
    mode_label: str,
) -> dict[str, object]:
    feature_names = artifact["feature_names"]
    values = np.array([float(feature_row[name]) for name in feature_names], dtype=float)
    means = np.array(artifact["scaler_mean"], dtype=float)
    scales = np.array(artifact["scaler_scale"], dtype=float)
    coefs = np.array(artifact["coef"], dtype=float)
    safe_scales = np.where(scales == 0, 1.0, scales)
    standardized = (values - means) / safe_scales
    contributions = standardized * coefs
    normalized_raw = float(artifact["intercept"]) + float(contributions.sum())
    normalized_clipped = float(np.clip(normalized_raw, 0.0, 1.0))
    score_min = float(artifact["score_min"])
    score_max = float(artifact["score_max"])
    pred_score = score_min + normalized_clipped * (score_max - score_min)
    rounded = float(np.clip(np.rint(pred_score), score_min, score_max))
    score_band = classify_score_band(normalized_clipped)
    review_note, boundary_distance = build_rounding_review_note(pred_score, score_min, score_max)

    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "label": [FEATURE_LABELS.get(name, name) for name in feature_names],
            "value": values,
            "standardized_value": standardized,
            "coefficient": coefs,
            "contribution": contributions,
        }
    ).sort_values("contribution", ascending=False)

    top_influential = (
        feature_table.assign(abs_contribution=lambda df: df["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(5)
        .copy()
    )
    explanation_bullets = build_plain_english_explanation(
        task_name=task_name,
        feature_table=feature_table,
        normalized_score=normalized_clipped,
        score_band=score_band,
    )
    return {
        "task_name": task_name,
        "unit_id": int(unit_id),
        "mode_label": mode_label,
        "model_type": artifact["model_type"],
        "weak_label_variant": artifact["weak_label_variant"],
        "question_label": artifact.get("question_label", f"{task_name} Unit {unit_id}"),
        "prompt_title": artifact.get("prompt_title", f"Unit {unit_id}"),
        "prompt_text": artifact.get("prompt_text", ""),
        "prompt_context": artifact.get("prompt_context", ""),
        "question_text": artifact.get("question_text", artifact.get("prompt_text", "")),
        "subject": artifact.get("subject", "UNKNOWN"),
        "grade_level": artifact.get("grade_level", "UNKNOWN"),
        "response_type": artifact.get("response_type", "UNKNOWN"),
        "score_min": int(score_min),
        "score_max": int(score_max),
        "normalized_raw": normalized_raw,
        "normalized_clipped": normalized_clipped,
        "pred_score": float(pred_score),
        "pred_score_rounded": rounded,
        "score_band": score_band,
        "review_note": review_note,
        "boundary_distance": boundary_distance,
        "feature_table": feature_table,
        "top_influential_contributions": top_influential,
        "explanation_bullets": explanation_bullets,
    }


def render_sidebar(
    config: dict[str, object],
    bundle: dict[str, object],
    question_contexts: dict[str, dict[str, dict[str, object]]],
) -> tuple[str, str, int, bool]:
    st.sidebar.title("Weakly Supervised Interpretable Grading Demo")
    st.sidebar.caption(config.get("subtitle", "Decision-support estimate, not a final grade"))
    presentation_mode = st.sidebar.toggle("Presentation mode", value=True)

    available_tasks = [task for task in ["ASAG", "AES"] if task in bundle["task_bundles"]]
    if not available_tasks:
        available_tasks = ["ASAG"]
    task_name = st.sidebar.selectbox(
        "Task",
        available_tasks,
        format_func=lambda task: "ASAP-SAS Short Answer Grading" if task == "ASAG" else "ASAP-AES Essay Scoring",
    )

    mode = st.sidebar.selectbox(
        "Mode",
        [
            "Live grading",
            "Sample replay demo — not live grading",
            "Reproducibility dashboard",
        ],
    )

    task_bundle = bundle["task_bundles"].get(task_name, {"artifacts": {}})
    default_units = config.get("default_questions", list(range(1, 11))) if task_name == "ASAG" else list(range(1, 9))
    available_units = sorted(task_bundle["artifacts"].keys()) or default_units
    available_units = ordered_units(task_name, available_units, task_bundle["artifacts"], question_contexts)
    label = "Question" if task_name == "ASAG" else "Essay set"
    unit_id = st.sidebar.selectbox(
        label,
        available_units,
        format_func=lambda uid: format_unit_label(task_name, int(uid), task_bundle["artifacts"], question_contexts),
    )

    st.sidebar.markdown("---")
    st.sidebar.warning("Gold labels are not used as training targets.")
    render_links(config.get("links", {}), presentation_mode)

    return task_name, mode, int(unit_id), presentation_mode


def render_links(links: dict[str, str], presentation_mode: bool) -> None:
    st.sidebar.subheader("Links")
    for label, value in links.items():
        pretty = label.replace("_", " ").title()
        if value.startswith("http://") or value.startswith("https://"):
            st.sidebar.markdown(f"- [{pretty}]({value})")
            continue
        if presentation_mode:
            if value.endswith("_TODO"):
                st.sidebar.markdown(f"- `{pretty}: {value}`")
            continue
        st.sidebar.markdown(f"- `{pretty}: {value}`")


def get_task_samples(task_name: str, unit_id: int, sample_inputs: dict[str, object]) -> list[dict[str, str]]:
    if task_name in sample_inputs:
        return sample_inputs.get(task_name, {}).get(str(unit_id), [])
    return []


def render_top_banner(presentation_mode: bool) -> None:
    if presentation_mode:
        st.info("Live weakly supervised grading demo: local model, no LLM calls.")


def render_live_mode(
    task_name: str,
    unit_id: int,
    task_bundle: dict[str, object],
    sample_inputs: dict[str, object],
    question_contexts: dict[str, dict[str, dict[str, object]]],
    presentation_mode: bool,
) -> None:
    render_top_banner(presentation_mode)
    st.title("Weakly Supervised Interpretable Grading Demo")
    st.caption("This model estimates a score from weakly supervised training signals and human-readable features.")

    artifacts = task_bundle["artifacts"]
    if unit_id not in artifacts:
        st.error(
            "Live artifacts are missing for this unit. Run `python3 -m src.prepare_demo_artifacts` first."
        )
        return

    render_question_panel(task_name, unit_id, artifacts[unit_id], question_contexts)
    st.info("Use as decision support, not final grading.")

    samples = get_task_samples(task_name, unit_id, sample_inputs)
    text_key = f"live_answer_{task_name}_{unit_id}"
    sample_key = f"live_sample_{task_name}_{unit_id}"
    if text_key not in st.session_state:
        st.session_state[text_key] = ""
    if sample_key not in st.session_state:
        st.session_state[sample_key] = "Custom answer"

    sample_labels = ["Custom answer"] + [sample["label"] for sample in samples]
    selected_sample = st.selectbox(
        "Try a demo-safe synthetic response.",
        sample_labels,
        key=f"sample_select_{task_name}_{unit_id}",
    )

    selected_note = ""
    if selected_sample != st.session_state[sample_key]:
        st.session_state[sample_key] = selected_sample
        if selected_sample == "Custom answer":
            st.session_state[text_key] = ""
        else:
            chosen = next(sample for sample in samples if sample["label"] == selected_sample)
            st.session_state[text_key] = chosen["text"]
            selected_note = chosen.get("demo_note", "")
    elif selected_sample != "Custom answer":
        chosen = next(sample for sample in samples if sample["label"] == selected_sample)
        selected_note = chosen.get("demo_note", "")

    st.caption("Synthetic examples are for demonstration only and are not part of the training data.")
    if selected_note:
        st.caption(selected_note)

    st.text_area(
        "Response text",
        key=text_key,
        height=180,
        help="Synthetic examples are written for demo use and are not copied from raw student responses.",
    )

    if st.button("Grade response", type="primary"):
        answer_text = st.session_state[text_key]
        if not clean_text(answer_text):
            st.error("Please enter a response before grading.")
        else:
            st.session_state["live_result"] = score_live_answer(task_name, unit_id, answer_text, artifacts[unit_id])

    result = st.session_state.get("live_result")
    if not result or result["task_name"] != task_name or result["unit_id"] != unit_id:
        render_method_panel(presentation_mode)
        return

    render_result_cards(result, mode="Live inference")
    render_explanation_panel(result)
    render_feature_outputs(result, presentation_mode)
    render_method_panel(presentation_mode)


def render_result_cards(result: dict[str, object], mode: str) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted score", f"{result['pred_score']:.2f} / {result['score_max']}")
    col2.metric("Rounded score", f"{int(result['pred_score_rounded'])} / {result['score_max']}")
    col3.metric("Score band", result["score_band"])

    normalized_text = f"{result['normalized_clipped']:.3f}"
    st.markdown(
        f"""
        - **Mode:** {mode}
        - **Question:** {result['question_label']}
        - **Score range:** {result['score_min']} to {result['score_max']}
        - **Normalized score:** {normalized_text}
        - **Review note:** {result['review_note']}
        - **Model type:** `{result['model_type']}`
        - **Weak-label variant:** `{result['weak_label_variant']}`
        """
    )


def render_explanation_panel(result: dict[str, object]) -> None:
    st.subheader("Why this score?")
    for bullet in result["explanation_bullets"]:
        st.markdown(f"- {bullet}")

    with st.expander("What this explanation means", expanded=False):
        st.markdown(
            """
            The contribution table shows which human-readable features pushed the score estimate up.
            It does not prove that the answer matches the rubric. Because the model was trained on
            weak labels, it can over-reward verbose or keyword-rich answers and underpredict concise
            correct answers.

            - Use as decision support, not final grading.
            - Human review is required for high-stakes use.
            """
        )


def render_feature_outputs(result: dict[str, object], presentation_mode: bool) -> None:
    influential = result["top_influential_contributions"].copy()
    if influential.empty:
        st.info("No feature contributions were available for this response.")
        return

    def draw_body() -> None:
        summary_df = influential[["label", "value", "coefficient", "contribution"]].copy().round(4)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        chart = build_contribution_chart(influential)
        st.pyplot(chart, clear_figure=True)
        st.caption("Positive bars push the estimate upward. Negative bars pull it downward.")

        st.subheader("Full feature view")
        feature_table = result["feature_table"].copy()
        display_table = feature_table[["label", "value", "standardized_value", "coefficient", "contribution"]].round(4)
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        st.caption("This is a linear-feature contribution explanation, not a full semantic rubric explanation.")

    if presentation_mode:
        with st.expander("Detailed feature contributions", expanded=False):
            draw_body()
    else:
        st.subheader("Feature contributions")
        draw_body()


def build_contribution_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_df = df.sort_values("contribution", ascending=True)
    colors = ["#d95f5f" if value < 0 else "#4f8cc9" for value in plot_df["contribution"]]
    ax.barh(plot_df["label"], plot_df["contribution"], color=colors)
    ax.set_xlabel("Standardized value × coefficient")
    ax.set_ylabel("")
    ax.set_title("Most influential features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def render_question_panel(
    task_name: str,
    unit_id: int,
    artifact: dict[str, object],
    question_contexts: dict[str, dict[str, dict[str, object]]],
) -> None:
    context = build_context_payload(task_name, unit_id, artifact, question_contexts)
    status_label = CONTEXT_STATUS_LABELS.get(context["context_status"], context["context_status"])
    recommendation = context["demo_recommendation"]

    label = artifact.get("question_label", context["display_name"])
    title = artifact.get("prompt_title", context["display_name"])
    st.subheader(f"{label} — {title}")
    meta = (
        f"Subject: {artifact.get('subject', 'UNKNOWN')} | "
        f"Grade: {artifact.get('grade_level', 'UNKNOWN')} | "
        f"Response type: {artifact.get('response_type', 'UNKNOWN')}"
    )
    st.caption(meta)

    recommendation_text = "★ Recommended live demo" if recommendation == "recommended" else "Okay for live demo"
    if recommendation == "avoid_for_live_demo":
        recommendation_text = "Better to avoid for live demo"

    st.markdown("### Prompt / Question Context")
    st.markdown(
        f"- **Context status:** {status_label}\n"
        f"- **Demo recommendation:** {recommendation_text}\n"
        f"- **Score note:** {context['score_range_note']}"
    )

    if context["context_warning"]:
        st.warning(context["context_warning"])

    if context["background_context"]:
        with st.expander("1. Background / passage context", expanded=(task_name == "ASAG")):
            st.text_area(
                "Use this context if you want to draft or compare answers.",
                value=context["background_context"],
                height=220,
                key=f"context_{task_name}_{unit_id}",
            )
    else:
        st.info("1. Background / passage context: no additional local context is surfaced for this unit.")

    st.markdown("**2. Actual question or prompt**")
    if context["question_text"]:
        st.markdown(f"> {context['question_text']}")
    else:
        st.info("Prompt text is unavailable for this unit in the current local artifacts.")

    st.markdown("**3. What a response should do**")
    if context["response_should_do"]:
        st.markdown(f"- {context['response_should_do']}")
    else:
        st.info("A concise response guide is not available in the local demo metadata.")


def render_method_panel(presentation_mode: bool) -> None:
    with st.expander("How this demo score is produced", expanded=not presentation_mode):
        st.markdown(
            """
            This model estimates a score from weakly supervised training signals and
            human-readable features.

            The demo pipeline used here:

            1. Build train-only weak labels from unsupervised response signals.
            2. Extract lightweight interpretable text features.
            3. Fit one non-negative linear model per prompt/question.
            4. Clip predictions to the valid score range and round only for the
               displayed discrete-score path.

            Because the model uses length, lexical variety, and similarity-derived
            weak labels, concise correct answers may be underpredicted and verbose
            keyword-rich incorrect answers may be overpredicted.
            """
        )
    st.warning("Responsible use: this is a decision-support estimate, not a final teacher grade.")


def render_sample_replay_mode(
    task_name: str,
    unit_id: int,
    task_bundle: dict[str, object],
    question_contexts: dict[str, dict[str, dict[str, object]]],
    presentation_mode: bool,
) -> None:
    render_top_banner(presentation_mode)
    st.title("Sample replay demo — not live grading")
    st.warning("This mode replays file-backed predictions from held-out results. It is useful for showing success and failure modes, but it is not scoring newly typed text.")

    replay_examples = task_bundle["replay_examples"]
    if replay_examples.empty:
        st.error(
            "No replay examples are available yet. Run `python3 -m src.prepare_demo_artifacts` first."
        )
        return

    replay_examples = add_replay_bucket(replay_examples)
    unit_column = "question_id" if task_name == "ASAG" else "essay_set"
    id_column = "sample_id" if task_name == "ASAG" else "essay_id"
    gold_column = "score_raw" if task_name == "ASAG" else "gold_score"
    filtered = replay_examples[replay_examples[unit_column] == unit_id].copy()
    if filtered.empty:
        st.info("No replay examples are available for this unit.")
        return

    split = st.selectbox("Replay split", sorted(filtered["split"].unique().tolist()))
    filtered = filtered[filtered["split"] == split].copy()
    error_choices = ["all"] + sorted(filtered["display_error_type"].dropna().astype(str).unique().tolist())
    selected_error = st.selectbox(
        "Error type filter",
        error_choices,
        format_func=pretty_error_type,
    )
    if selected_error != "all":
        filtered = filtered[filtered["display_error_type"] == selected_error].copy()

    if filtered.empty:
        st.info("No replay examples match this filter for the selected unit.")
        return

    filtered["example_label"] = filtered.apply(
        lambda row: f"Response {int(row[id_column])} — {row['selection_reason']}",
        axis=1,
    )
    selected_label = st.selectbox("Replay example", filtered["example_label"].tolist())
    replay_row = filtered[filtered["example_label"] == selected_label].iloc[0]

    artifact = task_bundle["artifacts"].get(unit_id)
    if artifact is None:
        st.error("Missing live artifact for this unit, so feature contributions cannot be shown.")
        return

    render_question_panel(task_name, unit_id, artifact, question_contexts)
    st.caption(f"Response ID: {int(replay_row[id_column])}. Raw student response text is intentionally hidden in this demo.")

    feature_values = {feature: float(replay_row[feature]) for feature in artifact["feature_names"]}
    result = score_from_features(task_name, unit_id, feature_values, artifact, mode_label="Sample replay")
    render_result_cards(result, mode="Sample replay")

    gold_score = float(replay_row[gold_column]) if not pd.isna(replay_row[gold_column]) else None
    extra_cols = st.columns(3)
    extra_cols[0].metric("Held-out gold score", f"{gold_score:.1f}" if gold_score is not None else "N/A")
    extra_cols[1].metric("Absolute error", f"{float(replay_row['absolute_error']):.2f}")
    extra_cols[2].metric("Error type", pretty_error_type(str(replay_row["display_error_type"])))

    prediction_source = "File-backed held-out result" if presentation_mode else f"`{replay_row['prediction_source']}`"
    st.markdown(
        f"""
        - **Selection reason:** {replay_row['selection_reason']}
        - **Prediction source:** {prediction_source}
        """
    )

    render_explanation_panel(result)
    render_feature_outputs(result, presentation_mode)
    render_method_panel(presentation_mode)


def render_dashboard(task_name: str, bundle: dict[str, object], presentation_mode: bool) -> None:
    render_top_banner(presentation_mode)
    st.title("Reproducibility dashboard")
    st.caption("This panel points back to the file-backed report bundle and demo artifact builder.")

    st.subheader("Commands")
    st.code(
        "\n".join(
            [
                PREPARE_COMMAND,
                APP_COMMAND,
                REPORT_BUNDLE_COMMAND,
            ]
        ),
        language="bash",
    )

    asag_bundle = bundle["task_bundles"].get("ASAG", {"artifacts": {}, "manifest": None})
    aes_bundle = bundle["task_bundles"].get("AES", {"artifacts": {}, "manifest": None})
    manifest = bundle.get("manifest")
    created_at = None
    if task_name == "ASAG":
        created_at = asag_bundle.get("manifest", {}).get("created_at_utc")
    elif task_name == "AES":
        created_at = aes_bundle.get("manifest", {}).get("created_at_utc")
    if not created_at and manifest is not None:
        created_at = manifest.get("created_at_utc")

    st.subheader("Artifact status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ASAG artifacts", len(asag_bundle.get("artifacts", {})))
    col2.metric("AES artifacts", len(aes_bundle.get("artifacts", {})))
    col3.metric("Manifest timestamp", created_at or "UNKNOWN")
    live_available = "Yes" if any(bundle_item.get("live_available") for bundle_item in bundle["task_bundles"].values()) else "No"
    col4.metric("Live grading available", live_available)

    st.warning("No ASAP-SAS private-test metrics are reported.")
    st.warning("Gold labels are not used as live model training targets.")
    st.warning("Some ASAP-SAS question contexts are incomplete in the local demo.")

    main_results = load_main_results()
    if not main_results.empty:
        st.subheader("Supported result summary")
        supported_systems = {
            "ASAG": ["asag_improved_weak_label_baseline", "asag_sbert_hybrid_supported_candidate"],
            "AES": ["aes_weak_label_baseline", "aes_set6_8_feature_selection"],
        }
        task_results = main_results[
            (main_results["task"] == task_name)
            & (main_results["supported_local"] == True)
            & (main_results["system_id"].isin(supported_systems.get(task_name, [])))
        ].copy()
        if not task_results.empty and "display_name" in task_results.columns:
            exploratory_mask = task_results["system_id"].eq("asag_sbert_hybrid_supported_candidate")
            task_results.loc[exploratory_mask, "display_name"] = task_results.loc[exploratory_mask, "display_name"] + " (exploratory)"
            keep_cols = [
                "display_name",
                "split",
                "qwk_macro",
                "mae_macro",
                "pearson_macro",
                "note",
            ]
            existing_cols = [column for column in keep_cols if column in task_results.columns]
            st.dataframe(task_results[existing_cols], use_container_width=True, hide_index=True)

    weak_quality = load_weak_quality()
    if not weak_quality.empty:
        st.subheader("Weak-label diagnostics")
        task_quality = weak_quality[weak_quality["task"] == task_name].copy()
        keep_cols = [
            "variant_name",
            "unit_id",
            "pearson_with_gold",
            "spearman_with_gold",
            "mae_scaled",
            "qwk_scaled",
            "source_note",
        ]
        existing_cols = [column for column in keep_cols if column in task_quality.columns]
        st.dataframe(task_quality[existing_cols], use_container_width=True, hide_index=True)

    task_bundle = bundle["task_bundles"].get(task_name, {})
    task_manifest = task_bundle.get("manifest")
    if task_manifest is not None and not presentation_mode:
        with st.expander("Loaded demo artifact manifest", expanded=False):
            st.json(task_manifest, expanded=False)


def main() -> None:
    config = load_config()
    sample_inputs = load_sample_inputs()
    question_contexts = load_question_contexts()
    bundle = load_demo_bundle()
    task_name, mode, unit_id, presentation_mode = render_sidebar(config, bundle, question_contexts)
    task_bundle = bundle["task_bundles"].get(task_name, {"artifacts": {}, "replay_examples": pd.DataFrame(), "manifest": None})

    if mode == "Live grading":
        render_live_mode(task_name, unit_id, task_bundle, sample_inputs, question_contexts, presentation_mode)
    elif mode == "Sample replay demo — not live grading":
        render_sample_replay_mode(task_name, unit_id, task_bundle, question_contexts, presentation_mode)
    else:
        render_dashboard(task_name, bundle, presentation_mode)


if __name__ == "__main__":
    main()
