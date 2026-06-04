from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from demo.demo_logic import build_plain_english_explanation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ARTIFACT_DIR = REPO_ROOT / "demo_artifacts"


class DemoArtifactSmokeTest(unittest.TestCase):
    def test_manifest_exists_after_preparation(self) -> None:
        manifest_path = DEMO_ARTIFACT_DIR / "manifest.json"
        self.assertTrue(manifest_path.exists(), "demo_artifacts/manifest.json is missing")

    def test_asag_artifact_loads(self) -> None:
        artifact_path = DEMO_ARTIFACT_DIR / "asag_question_1.json"
        self.assertTrue(artifact_path.exists(), "Missing ASAG artifact")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("task"), "ASAG")

    def test_aes_artifact_loads(self) -> None:
        artifact_path = DEMO_ARTIFACT_DIR / "aes_set_1.json"
        self.assertTrue(artifact_path.exists(), "Missing AES artifact")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("task"), "AES")

    def test_sample_inputs_json_is_valid(self) -> None:
        sample_input_path = REPO_ROOT / "demo" / "sample_inputs.json"
        payload = json.loads(sample_input_path.read_text(encoding="utf-8"))
        self.assertIn("ASAG", payload)
        self.assertIn("AES", payload)

    def test_question_context_json_is_valid(self) -> None:
        context_path = REPO_ROOT / "demo" / "question_context.json"
        payload = json.loads(context_path.read_text(encoding="utf-8"))
        self.assertIn("ASAG", payload)
        self.assertIn("AES", payload)

    def test_plain_english_explanation_is_non_empty(self) -> None:
        feature_table = pd.DataFrame(
            [
                {
                    "feature": "word_count",
                    "label": "Word count",
                    "value": 80.0,
                    "standardized_value": 1.25,
                    "coefficient": 0.30,
                    "contribution": 0.375,
                },
                {
                    "feature": "unique_word_count",
                    "label": "Unique word count",
                    "value": 55.0,
                    "standardized_value": 1.0,
                    "coefficient": 0.20,
                    "contribution": 0.200,
                },
            ]
        )
        bullets = build_plain_english_explanation(
            task_name="ASAG",
            feature_table=feature_table,
            normalized_score=0.72,
            score_band="higher score-range estimate",
        )
        self.assertTrue(bullets)
        self.assertTrue(all(isinstance(item, str) and item.strip() for item in bullets))


if __name__ == "__main__":
    unittest.main()
