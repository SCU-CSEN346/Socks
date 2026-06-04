# Improving Weakly Supervised and Interpretable Models for Automatic Short Answer Grading and Essay Scoring

## Project Option

This project follows **Option 2**: selecting a recent ACL Anthology paper with
available code, reproducing the baseline, and improving the approach.

This repository contains our master's-level NLP final project on weakly
supervised, interpretable grading for two educational NLP tasks:

- Automatic Essay Scoring (AES) with ASAP-AES
- Automatic Short Answer Grading (ASAG) with ASAP-SAS

## Overview

Automatically scoring open-ended student responses is useful in educational
settings, but many high-performing systems rely on large labeled datasets and
opaque models. We focus on a lightweight alternative that keeps the scoring
pipeline readable and reproducible.

The project implements:

- train-only weak-label generation from unsupervised response signals
- prompt/question-specific preprocessing and modeling
- lightweight, human-readable text features
- prompt/question-specific non-negative linear regression models
- evaluation with Quadratic Weighted Kappa (QWK), Mean Absolute Error (MAE),
  and Pearson correlation

AES and ASAG differ in response length and prompt dependence, so we handle them
as separate tasks with task-specific preprocessing and evaluation. This project
is inspired by Urrutia et al. (2025), but it is an adapted public-benchmark
implementation rather than a dataset-matched reproduction for ASAG: ASAP-SAS is
used as a public replacement for the original Spanish ASAG dataset from the
paper.

## Repository Structure

```text
.
├── data/                                # Local dataset files and dataset notes
├── demo/                                # Streamlit demo source, demo config, synthetic examples
├── demo_artifacts/                      # Generated local demo artifacts (created locally, git-ignored)
├── notebooks/                           # Exploratory notebooks and preserved helpers
├── results/
│   ├── final_report/                    # Submission-ready report bundle
│   │   ├── configs/                     # Manifest-adjacent config summaries
│   │   ├── figures/                     # Paper figures
│   │   ├── logs/                        # Audit notes and smoke-test logs
│   │   ├── predictions/                 # Prediction copies used in the report bundle
│   │   └── tables/                      # CSV/LaTeX result tables
│   ├── metrics/                         # Canonical baseline metrics
│   ├── predictions/                     # Canonical baseline predictions
│   ├── processed/                       # Canonical processed splits
│   └── weak_labels/                     # Canonical weak-label outputs
├── paper_pipeline/                      # Preserved earlier AES/SAS pipeline artifacts
├── src/
│   ├── build_final_report_bundle.py     # Final report bundle generator
│   ├── prepare_demo_artifacts.py        # Demo-artifact builder for ASAP-SAS and ASAP-AES
│   ├── run_aes_baseline.py              # ASAP-AES baseline runner
│   ├── run_asag_baseline.py             # ASAP-SAS baseline runner
│   ├── asag_baseline.py                 # ASAP-SAS training/evaluation pipeline
│   └── ...
├── tests/                               # Lightweight smoke tests
└── requirements.txt
```

Important final-report artifacts:

- `src/build_final_report_bundle.py`
- `results/final_report/experiment_manifest.json`
- `results/final_report/final_report_experiment_summary.md`
- `results/final_report/report_revision_patch.md`
- `results/final_report/tables/`
- `results/final_report/figures/`
- `results/final_report/configs/`
- `results/final_report/logs/`

Legacy `paper_pipeline/` note:

- `paper_pipeline/AES/` and `paper_pipeline/SAS/` preserve earlier task-specific
  notebooks and scripts. If you use those older paths, place raw ASAP-AES files
  under `paper_pipeline/AES/data/` and raw ASAP-SAS files under
  `paper_pipeline/SAS/data/`.

Paper source note:

- No editable ACL/Overleaf paper source is currently tracked in this repository.
  See `results/final_report/logs/report_source_audit.md`.

## Data

We use two public benchmark datasets:

- ASAP-AES for essay scoring
- ASAP-SAS for short answer grading

Raw datasets are expected locally under:

```text
data/asap-aes/
data/asap-sas/
```

Dataset sources:

- ASAP-AES: <https://www.kaggle.com/competitions/asap-aes/overview>
- ASAP-SAS: <https://www.kaggle.com/competitions/asap-sas/overview>

Notes:

- ASAP-AES is used for the essay-scoring experiments.
- ASAP-SAS is used as the public ASAG replacement dataset.
- ASAP-SAS public-test labels are available and used for evaluation.
- The ASAP-SAS private test split is prediction-only because labels are not
  available, so no private-split evaluation scores are reported.
- This repository is organized around code and derived artifacts; it does not
  make new claims about redistributing raw dataset content.

## Installation

Python 3.8+ is expected. The final report bundle was generated with Python
3.8.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current Python dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `streamlit`

## Local Demo

The repository also includes a local Streamlit demo for both tasks:

- ASAP-SAS short answer grading
- ASAP-AES essay scoring

Demo-specific source files are kept under `demo/`, and generated local demo
artifacts are written under `demo_artifacts/` so the presentation files stay
separate from the training and report pipelines.

To build the local demo artifacts from existing processed/model outputs:

```bash
python3 -m src.prepare_demo_artifacts
```

To launch the demo locally:

```bash
streamlit run demo/app.py
```

Main demo files:

- `demo/app.py`
- `demo/question_context.json`
- `demo/sample_inputs.json`
- `demo/README_DEMO.md`
- `src/prepare_demo_artifacts.py`

Main generated demo outputs:

- `demo_artifacts/manifest.json`
- `demo_artifacts/asag_question_<id>.json`
- `demo_artifacts/aes_set_<id>.json`
- `demo_artifacts/sample_replay_examples.csv`
- `demo_artifacts/aes_sample_replay_examples.csv`

Recommended live demo units:

- ASAP-SAS: questions `3`, `7`, `10`
- ASAP-AES: essay sets `1`, `3`, `6`, `8`

Known ASAP-SAS context gaps in the current local demo:

- questions `1`, `2`, and `8` are scoreable, but the full source context is
  not yet surfaced cleanly in the demo UI

The demo output is intended for decision support and analysis, not final
grading.

## Reproducing the Final Report Bundle

From the repository root, run:

```bash
python3 -m src.build_final_report_bundle
```

This command creates or updates `results/final_report/` with:

- tables in CSV and LaTeX format
- figures for the paper
- logs and audit notes
- config summaries
- prediction copies used in the bundle
- the experiment manifest

## Expected Output

Key artifacts produced by the final bundle include:

```text
results/final_report/tables/main_results_macro.csv
results/final_report/tables/per_prompt_results.csv
results/final_report/tables/weak_label_quality.csv
results/final_report/tables/ablation_results.csv
results/final_report/figures/system_pipeline.pdf
results/final_report/figures/per_prompt_qwk_delta.pdf
results/final_report/experiment_manifest.json
results/final_report/final_report_experiment_summary.md
results/final_report/report_revision_patch.md
```

## Main Results Summary

The summary below reflects file-backed local results from
`results/final_report/tables/main_results_macro.csv`.

- AES weak-label baseline: internal test QWK `0.5689`, MAE `1.6248`,
  Pearson `0.7079`
- AES set 6/8 feature-selection candidate: internal test QWK `0.5726`,
  MAE `1.6249`, Pearson `0.7202`
- ASAG improved weak-label baseline: public-test QWK `0.3234`,
  MAE `0.7488`, Pearson `0.4572`
- ASAG SBERT-hybrid weak-label exploratory variant: public-test QWK `0.3410`,
  MAE `0.7547`, Pearson `0.4498`

The SBERT-hybrid ASAG result is included as an exploratory file-backed variant,
not as the default final system. This README summarizes only local results that
are backed by files in the final report bundle.

### Optional Targeted AES GBDT Candidate

The default AES baseline remains unchanged. An optional targeted AES GBDT
runner can be used for candidate analysis:

```bash
python3 -m src.run_aes_gbdt_targeted
```

The targeted AES GBDT runner evaluates a prompt-specific nonlinear candidate
while keeping the default AES baseline unchanged. The best candidate uses GBDT
for validation-supported essay sets 1, 6, and 7, applies shrinked train-weak
quantile calibration only to essay set 6, and excludes essay set 8 due to an
MAE tradeoff. Compact outputs are saved under
`results/clean_results/aes_gbdt_targeted/`.

## Metrics

Primary metric:

- Quadratic Weighted Kappa (QWK)

Secondary metrics:

- Mean Absolute Error (MAE)
- Pearson correlation

Evaluation convention used in the final bundle:

- predictions are clipped to the valid prompt/question score range
- rounded predictions are used for QWK
- continuous clipped predictions are used for MAE and Pearson

These conventions are recorded in
`results/final_report/experiment_manifest.json` and
`results/final_report/configs/metric_audit.json`.

## ASAP-AES Canonical Baseline

The canonical AES runner is:

```bash
python3 -m src.run_aes_baseline
```

Pipeline:

```text
raw ASAP-AES training_set_rel3.tsv
-> reproducible internal train/val/test split per essay_set
-> train-only weak labels
-> lightweight interpretable features
-> one non-negative linear model per essay_set
-> internal validation/test evaluation
```

## ASAP-AES Low-Level Preprocessing

To preprocess one essay set from the raw training split:

```bash
python3 -m src.preprocess_asap --split train --essay-set 1
```

To preprocess the full raw training split:

```bash
python3 -m src.preprocess_asap --split train
```

## Contributions

Helen Wang - Wrote the related works section of the report, added modified scripts for signal clustering (signal_pred and matrix) as well as dividing dataset data into subsections to follow project structure, added the contributions section to README, added read data script to divide SAS dataset into subsections, wrote dataset section of report, updated weak signal code to improve SAS scores, updated README to include dataset section and links, modified code for weak signal to improve AES scores, updated weak signal section of report to reflect changes in code.

Harshvardhan Garude - Set up baseline project structure, added scripts for loading ASAP-AES dataset splits (data_loading) and preprocessing data (preprocess_asap) with essay-set filtering and text cleaning, and updated README and data documentation for reproducibility. Experimented with different methods to improve the AES pipeline, including Gradient Boosted Decision Trees (GBDT), to improve scoring performance.

Shravan Dhanasekaran - Full reproduction of the paper pipeline across both AES (8 essay sets) and SAS (10 question sets), implementing the complete three-phase approach from the original paper: weak label generation, NLLF feature extraction, and positive linear regression with genetic feature selection. Replicated signal clustering, ULRA, LLM weak label generation (vanilla, chain-of-thought, weak signal), NLLF feature extraction via BSQs, similarity-based scoring, BERT classification, and positive linear regression training and evaluation. Switched the final regression model from positive LinearRegression to Ridge(alpha=10) for improved regularization.

## ASAP-SAS ASAG Baseline

The canonical ASAG runner is:

```bash
python3 -m src.run_asag_baseline
```

Pipeline:

```text
raw ASAP-SAS files
-> canonical preprocessing
-> train-only weak labels
-> lightweight interpretable features
-> one non-negative linear model per question_id
-> validation/public-test evaluation
```

Additional ASAP-SAS entry points:

```bash
python3 -m src.preprocess_asap_sas
python3 -m src.asag_weak_labels
```

## Reproducibility Notes

- `results/final_report/experiment_manifest.json` records the git commit hash,
  package versions, seeds, dataset paths, commands, and output files used in
  the final bundle.
- Gold labels are not used to generate weak labels or train weak-label models.
- Gold labels are used only for validation, held-out evaluation, and
  diagnostics.
- The ASAP-SAS private split remains prediction-only because public labels are
  unavailable.
- The final report bundle summarizes only locally supported results.

## Limitations

- Weak labels rely on response length, lexical breadth, and similarity signals.
- The system can over-reward verbose or keyword-rich incorrect answers.
- Concise correct answers can be underpredicted.
- ASAP-SAS is a public replacement for the original ASAG dataset, not a
  dataset-matched reproduction.

## Report Links

- Final report bundle summary:
  `results/final_report/final_report_experiment_summary.md`
- Report revision notes:
  `results/final_report/report_revision_patch.md`
- Manifest:
  `results/final_report/experiment_manifest.json`

## Helper / Legacy Scripts

The main runnable entry points for this repository are:

- `python3 -m src.run_aes_baseline`
- `python3 -m src.run_asag_baseline`
- `python3 -m src.build_final_report_bundle`

Preserved helper or exploratory files include:

- `src/matrix.py`
- `src/signal_pred.py`
- `notebooks/read_data_sas.py`
- `notebooks/matrix_sas.py`
- `notebooks/pred_sas.py`
