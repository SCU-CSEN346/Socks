from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "character_count": "length",
    "word_count": "length",
    "sentence_count": "length",
    "average_sentence_length": "length",
    "short_answer_bin": "length",
    "medium_answer_bin": "length",
    "long_answer_bin": "length",
    "paragraph_count": "length",
    "unique_word_count": "lexical",
    "type_token_ratio": "lexical",
    "long_word_count": "lexical",
    "average_word_length": "lexical",
    "punctuation_count": "surface",
    "digit_count": "surface",
    "uppercase_count": "surface",
    "stopword_ratio": "surface",
}


def classify_score_band(normalized_score: float) -> str:
    if normalized_score < 0.34:
        return "lower score-range estimate"
    if normalized_score < 0.67:
        return "middle score-range estimate"
    return "higher score-range estimate"


def rounding_boundary_distance(pred_score: float, score_min: float, score_max: float) -> float | None:
    lower = int(np.floor(score_min))
    upper = int(np.ceil(score_max))
    boundaries = [value + 0.5 for value in range(lower, upper) if score_min < value + 0.5 < score_max]
    if not boundaries:
        return None
    return float(min(abs(pred_score - boundary) for boundary in boundaries))


def build_rounding_review_note(
    pred_score: float,
    score_min: float,
    score_max: float,
    threshold: float = 0.15,
) -> tuple[str, float | None]:
    distance = rounding_boundary_distance(pred_score, score_min, score_max)
    if distance is not None and distance <= threshold:
        return "Near a rounding boundary; human review especially important.", distance
    return "Stable relative to nearest rounded score, but still requires human review.", distance


def build_plain_english_explanation(
    task_name: str,
    feature_table: pd.DataFrame,
    normalized_score: float,
    score_band: str,
) -> list[str]:
    if feature_table.empty:
        return [
            f"This lands in the {score_band} for this {unit_label(task_name)}.",
            "This does not prove the answer is semantically correct; it explains what the linear model used.",
        ]

    working = feature_table.copy()
    working["group"] = working["feature"].map(FEATURE_GROUPS).fillna("other")
    working["abs_contribution"] = working["contribution"].abs()

    group_summary = (
        working.groupby("group", as_index=False)["contribution"]
        .sum()
        .assign(abs_contribution=lambda df: df["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
    )

    bullets: list[str] = []
    for _, row in group_summary.head(2).iterrows():
        group = str(row["group"])
        contribution = float(row["contribution"])
        top_features = (
            working[working["group"] == group]
            .sort_values("abs_contribution", ascending=False)["label"]
            .head(2)
            .tolist()
        )
        bullets.append(group_explanation(group, contribution, top_features))

    top_feature = working.sort_values("abs_contribution", ascending=False).iloc[0]
    feature_label = str(top_feature["label"]).lower()
    direction = "increases" if float(top_feature["contribution"]) >= 0 else "pulls down"
    bullets.append(
        f"The strongest single feature here is {feature_label}, which {direction} the estimate."
    )
    bullets.append(f"Overall this lands in the {score_band} for this {unit_label(task_name)}.")
    bullets.append(
        "This does not prove the answer is semantically correct; it explains what the linear model used."
    )

    return dedupe_preserve_order(bullets)[:5]


def group_explanation(group: str, contribution: float, top_features: Iterable[str]) -> str:
    feature_text = ", ".join(top_features)
    if group == "length":
        if contribution >= 0:
            return (
                "The model is rewarding response length/completeness proxies here, "
                f"especially {feature_text}."
            )
        return (
            "The response is lighter on the model's length/completeness proxies, "
            f"so features like {feature_text} pull the estimate down."
        )

    if group == "lexical":
        if contribution >= 0:
            return f"The model is rewarding lexical variety here, especially {feature_text}."
        return f"Lower lexical variety signals, including {feature_text}, pull the estimate down."

    if group == "surface":
        if contribution >= 0:
            return (
                "The model is using surface structure as a writing-quality proxy here, "
                f"with features like {feature_text} pushing upward."
            )
        return (
            "Surface-structure signals, including "
            f"{feature_text}, are pulling the estimate down for this response."
        )

    if contribution >= 0:
        return f"Other measurable cues, including {feature_text}, also push the estimate upward."
    return f"Other measurable cues, including {feature_text}, pull the estimate downward."


def unit_label(task_name: str) -> str:
    return "question" if task_name == "ASAG" else "essay set"


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        output.append(item)
        seen.add(item)
    return output
