## Project Option

This project follows **Option 2**: selecting a recent ACL Anthology paper with available code, reproducing the baseline, and improving the approach.

---

## paper_pipeline

Shravan Dhanasekaran's full reproduction of the paper pipeline across both AES (8 essay sets) and SAS (10 question sets), implementing the complete three-phase approach from the original paper: weak label generation, NLLF feature extraction, and positive linear regression with genetic feature selection.

```text
paper_pipeline/
├── AES/        # Automated Essay Scoring pipeline (ASAP-AES, essay sets 1–8)
└── SAS/        # Short Answer Scoring pipeline (ASAP-SAS, question sets 1–10)
```

Place the extracted ASAP-AES files under `paper_pipeline/AES/data/` and the extracted ASAP-SAS files under `paper_pipeline/SAS/data/` before running any scripts. See `paper_pipeline/AES/read_data.ipynb` and `paper_pipeline/SAS/read_data.ipynb` to preprocess the raw dataset files.

**Contributions — Shravan Dhanasekaran:** Replicated the full AES and SAS pipelines from the original paper across all essay and question sets, including signal clustering, ULRA, LLM weak label generation (vanilla, chain-of-thought, weak signal), NLLF feature extraction via BSQs, similarity-based scoring, BERT classification, and positive linear regression training and evaluation. Switched the final regression model from positive LinearRegression to Ridge(alpha=10) for improved regularization.

---

# Socks



Baseline setup for a course project on weakly supervised and interpretable
automated essay scoring.

This repository is being organized around a small, reviewable baseline. The
raw ASAP-AES dataset is expected to live locally under `data/asap-aes/`, while
raw ASAP-SAS dataset is expected to live locally under `data/asap-sas/`.
Source code, notebooks, and generated results are kept separate.

## Project Layout

```text
.
├── data/          # Local dataset files and dataset notes
├── notebooks/     # Exploratory notebooks
├── results/       # Generated outputs, metrics, and reports
├── src/           # Project source package and preprocessing code
└── requirements.txt
```

## Datasets

The ASAP-AES dataset we used was the same dataset used in the original paper for the AES tasks while the ASAP-SAS dataset we used was to substitute the original dataset used for the SAS tasks. 

The ASAP-AES dataset we used can be downloaded from: https://www.kaggle.com/competitions/asap-aes/overview

The ASAP-SAS dataset we used can be downloaded from: https://www.kaggle.com/competitions/asap-sas/overview 

## Setup

Install the baseline dependency:

```bash
python3 -m pip install -r requirements.txt
```

The raw datasets should remain extracted at:

```text
data/asap-aes/
data/asap-sas/
```

Those directories are ignored by git.

## Loading Data

Use `src.data_loading` when scripts or notebooks need the raw TSV files:

```python
from src.data_loading import load_asap_split

train_set_1 = load_asap_split("train", essay_set=1)
valid_set_1 = load_asap_split("valid", essay_set=1)
test_set_1 = load_asap_split("test", essay_set=1)
```

Valid split names are `train`, `valid`, and `test`.

## ASAP-AES Canonical Baseline

ASAP-AES is the canonical AES dataset for the paper-style weakly supervised,
white-box baseline in this repository.

The canonical AES baseline command is:

```bash
python3 -m src.run_aes_baseline
```

This executes:

```text
raw ASAP-AES training_set_rel3.tsv
-> preprocessing and reproducible internal train/val/test split per essay_set
-> train-only signal-clustering weak labels
-> lightweight interpretable expert features
-> one positive linear regression model per essay_set
-> evaluation on internal val and test with QWK / MAE / Pearson
```

Expected AES outputs:

```text
results/processed/asap-aes/train.csv
results/processed/asap-aes/val.csv
results/processed/asap-aes/test.csv
results/processed/asap-aes/split_summary.csv
results/weak_labels/asap-aes/train_weak_labels.csv
results/weak_labels/asap-aes/train_weak_label_diagnostics.csv
results/features/asap-aes/train_features.csv
results/features/asap-aes/val_features.csv
results/features/asap-aes/test_features.csv
results/predictions/asap-aes/val_predictions.csv
results/predictions/asap-aes/test_predictions.csv
results/predictions/asap-aes/val_test_predictions.csv
results/metrics/asap-aes/metrics.csv
results/metrics/asap-aes/metrics.json
results/metrics/asap-aes/summary_table.txt
results/models/asap-aes/positive_linear_coefficients.csv
```

Current AES feature set:

- `word_count`
- `character_count`
- `sentence_count`
- `average_word_length`
- `unique_word_count`
- `type_token_ratio`
- `long_word_count`
- `punctuation_count`
- `digit_count`
- `paragraph_count`

This baseline intentionally focuses on the simplest paper-aligned path:
signal-clustering weak labels plus interpretable features plus positive linear
regression. The canonical runner reuses the preserved teammate AES helper logic
from `src/matrix.py` and `src/signal_pred.py` for similarity construction and
weak-signal propagation. It does not implement NLLF, BSQ, or LLM weak labels.

### Optional Targeted AES GBDT Candidate

The default AES baseline remains unchanged. An optional targeted AES GBDT
runner can be used for candidate analysis:

```bash
python3 -m src.run_aes_gbdt_targeted
```

The targeted AES GBDT runner evaluates a prompt-specific nonlinear candidate
while keeping the default AES baseline unchanged. The best candidate uses GBDT
for validation-supported essay sets 1, 6, and 7, applies shrinked train-weak
quantile calibration only to essay set 6, and excludes essay set 8 due to an MAE
tradeoff. Compact outputs are saved under
`results/clean_results/aes_gbdt_targeted/`.

On the current split, the targeted candidate improves AES test QWK from 0.5689
to 0.5813, test MAE from 1.6248 to 1.6199, and test Pearson from 0.7079 to
0.7142. This is reported as an optional candidate using interpretable features
with a tree-based model, not as a replacement for the canonical baseline.

## ASAP-AES Low-Level Preprocessing

Run the preprocessing CLI from the repository root. To prepare one essay set
from the training split:

```bash
python3 -m src.preprocess_asap --split train --essay-set 1
```

This writes:

```text
results/processed/asap_train_set_1.csv
```

To preprocess the full training split:

```bash
python3 -m src.preprocess_asap --split train
```

Validation and test splits can be prepared the same way:

```bash
python3 -m src.preprocess_asap --split valid --essay-set 1
python3 -m src.preprocess_asap --split test --essay-set 1
```

The preprocessing step normalizes essay whitespace, removes empty essay rows,
drops rater-level columns, and creates a standardized `score` column only when
`domain1_score` is present. Splits without labels remain unlabeled.

Processed outputs are written under `results/`, which is ignored by git except
for placeholder files currently.

## Contributions

Helen Wang - Wrote the related works section of the report, added modified scripts for signal clustering (signal_pred and matrix) as well as dividing dataset data into subsections to follow project structure, added the contributions section to README, added read data script to divide SAS dataset into subsections, wrote dataset section of report, updated weak signal code to improve scores, updated README to include dataset section and links.

Harshvardhan Garude - Set up baseline project structure, added scripts for loading ASAP-AES dataset splits (data_loading) and preprocessing data (preprocess_asap) with essay-set filtering and text cleaning, and updated README and data documentation for reproducibility.

## ASAP-SAS ASAG Baseline

ASAP-SAS is the project's replacement dataset for the ASAG portion of the
original paper.

The canonical ASAG baseline command is:

```bash
python3 -m src.run_asag_baseline
```

This executes:

```text
raw ASAP-SAS files
-> canonical preprocessing under results/processed/asap-sas/
-> train-only signal-clustering weak labels
-> lightweight interpretable text features
-> one positive linear regression model per question_id
-> evaluation on val and public_test
```

The canonical ASAP-SAS preprocessing command is:

```bash
python3 -m src.preprocess_asap_sas
```

The canonical train-only weak-label command is:

```bash
python3 -m src.asag_weak_labels
```

Expected ASAG outputs:

```text
results/processed/asap-sas/train.csv
results/processed/asap-sas/val.csv
results/processed/asap-sas/public_test.csv
results/processed/asap-sas/private_test.csv
results/processed/asap-sas/score_ranges.csv
results/weak_labels/asap-sas/train_signal_clustering.csv
results/weak_labels/asap-sas/train_signal_clustering_diagnostics.csv
results/features/asap-sas/train_features.csv
results/features/asap-sas/val_features.csv
results/features/asap-sas/public_test_features.csv
results/features/asap-sas/private_test_features.csv
results/predictions/asap-sas/val_predictions.csv
results/predictions/asap-sas/public_test_predictions.csv
results/predictions/asap-sas/private_test_predictions.csv
results/metrics/asap-sas/metrics.csv
results/metrics/asap-sas/metrics.json
results/metrics/asap-sas/summary_table.txt
results/models/asap-sas/positive_linear_coefficients.csv
```

Current feature set:

- `character_count`
- `word_count`
- `sentence_count`
- `average_word_length`
- `average_sentence_length`
- `unique_word_count`
- `type_token_ratio`
- `long_word_count`
- `digit_count`
- `punctuation_count`
- `uppercase_count`
- `stopword_ratio`
- `short_answer_bin`
- `medium_answer_bin`
- `long_answer_bin`

## Helper / Legacy Scripts

The canonical runnable baselines live under `src.run_aes_baseline` and
`src.run_asag_baseline`.

The following files are preserved as teammate helper or exploratory scripts and
are not the main end-to-end entry points:

- `src/matrix.py`
- `src/signal_pred.py`
- `notebooks/read_data_sas.py`
- `notebooks/matrix_sas.py`
- `notebooks/pred_sas.py`
