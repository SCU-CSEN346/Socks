# Demo README

This folder contains the local Streamlit demo for the final project:

- ASAP-SAS automatic short answer grading
- ASAP-AES automatic essay scoring

The demo is designed for a live poster session or a short recording. It uses
local weakly supervised model artifacts already produced in this repository. It
does not call an LLM, and it does not use gold labels as training targets for
the live demo models.

## What the app demonstrates

The app has three modes:

1. `Live grading`
   - Scores newly typed ASAP-SAS or ASAP-AES responses with local artifacts
   - Shows prompt/question context, score interpretation, and feature-based
     explanation
   - Uses question-specific or essay-set-specific non-negative linear models

2. `Sample replay demo — not live grading`
   - Replays file-backed held-out predictions
   - Shows success and failure cases such as near misses, underprediction, and
     overprediction
   - Hides long raw student responses

3. `Reproducibility dashboard`
   - Shows the commands used to regenerate demo artifacts and the final report
     bundle
   - Summarizes supported result tables and weak-label diagnostics when present

## Build artifacts

Prepare demo artifacts from existing processed data, weak labels, features,
coefficients, and report outputs:

```bash
python3 -m src.prepare_demo_artifacts
```

This creates or refreshes:

```text
demo_artifacts/manifest.json
demo_artifacts/asag_question_1.json
...
demo_artifacts/asag_question_10.json
demo_artifacts/aes_set_1.json
...
demo_artifacts/aes_set_8.json
demo_artifacts/sample_replay_examples.csv
demo_artifacts/aes_sample_replay_examples.csv
```

## Run locally

```bash
streamlit run demo/app.py
```

Optional supporting command:

```bash
python3 -m src.build_final_report_bundle
```

## Recommended live demo units

ASAG:
- Question 3
- Question 7
- Question 10

AES:
- Essay Set 1
- Essay Set 3
- Essay Set 6
- Essay Set 8

These units either have usable source context in local artifacts or are
self-contained enough for a short live demo.

Questions to avoid for live demo right now:
- ASAP-SAS Question 1
- ASAP-SAS Question 2
- ASAP-SAS Question 8

Those questions are scoreable by the local model, but the full source context
is not yet surfaced cleanly in the demo.

## Live grading vs sample replay

`Live grading`:
- takes newly typed text
- extracts lightweight interpretable features
- standardizes the feature vector with stored scaler statistics
- applies a prompt/question-specific non-negative linear model
- clips to the valid score range
- shows a continuous score estimate, rounded score, score band, and plain-English explanation

`Sample replay demo — not live grading`:
- replays file-backed held-out predictions only
- is useful for showing model behavior on success/failure cases
- does not grade newly typed text

## Presentation notes

- The app includes a `Presentation mode` toggle.
- Recommended units are marked with a star in the selector.
- Synthetic sample responses are included for demo use only.
- The `Why this score?` panel translates feature contributions into plain English.
- The app is intended for decision support and analysis, not final grading.

## Recommended recording flow

1. Open the app in `Presentation mode`.
2. Start with ASAP-SAS Question 3, 7, or 10.
3. Load a synthetic response from the dropdown.
4. Click `Grade response`.
5. Walk through:
   - predicted score
   - rounded score
   - score band
   - plain-English explanation
   - feature contribution chart
6. Switch to `Sample replay demo — not live grading` and show one
   underprediction or overprediction example.
7. Open `Reproducibility dashboard` and show the exact regeneration commands.
8. Repeat once with an AES essay set for balanced coverage.

Recording script:

> First, I select a recommended ASAP-SAS question and load a synthetic response.
> When I click Grade response, the app extracts human-readable features,
> standardizes them, and applies a question-specific non-negative linear model
> trained on weak labels. The app returns a continuous score estimate, a rounded
> score, and the valid score range. The “Why this score?” panel translates the
> top feature contributions into plain English, showing whether the model relied
> on length, lexical variety, or surface structure. This is useful because the
> system is reviewable rather than black-box. The replay tab shows file-backed
> held-out examples, including underprediction and overprediction cases. The
> reproducibility tab shows the commands used to regenerate artifacts and the
> final report bundle. The output is decision support only, not a final teacher
> grade.

## Limitations

- The explanation is feature-based, not a semantic rubric proof.
- Weak labels rely on response length, lexical breadth, and similarity signals.
- Verbose or keyword-rich answers can be over-rewarded.
- Concise correct answers can be underpredicted.
- ASAP-SAS private-test labels are unavailable, so private-test metrics are not
  reported.
- Some ASAP-SAS question contexts are incomplete in the local demo.

## Guardrails

- Do not call the displayed score a final grade.
- Do not describe the score band or review note as confidence.
- Do not imply the model has verified semantic correctness.
- Do not report ASAP-SAS private-test metrics from this demo.
