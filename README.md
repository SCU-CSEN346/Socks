## Project Option

This project follows **Option 2**: selecting a recent ACL Anthology paper with available code, reproducing the baseline, and improving the approach.

# Socks



Baseline setup for a course project on weakly supervised and interpretable
automated essay scoring.

This repository is being organized around a small, reviewable baseline. The
raw ASAP-AES dataset is expected to live locally under `data/asap-aes/`, while
source code, notebooks, and generated results are kept separate.

## Status

The repository now has:

- ASAP-AES data loading and preprocessing utilities
- Teammate helper scripts for AES / ASAP-SAS signal clustering
- One canonical ASAP-SAS ASAG baseline path under `src/`

The immediate milestone focus is a clean baseline-first setup, not a full
reproduction of the original paper.

## Project Layout

```text
.
├── data/          # Local dataset files and dataset notes
├── notebooks/     # Exploratory notebooks
├── results/       # Generated outputs, metrics, and reports
├── src/           # Project source package and preprocessing code
└── requirements.txt
```

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

## Preprocessing

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

Helen Wang - Wrote the related works section of the report, added modified scripts for signal clustering (signal_pred and matrix) as well as dividing dataset data into subsections to follow project structure, added the contributions section to README, added read data script to divide SAS dataset into subsections, wrote dataset section of report.

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

## Helper Scripts

The following files are preserved as teammate helper / exploratory scripts and
are not the canonical baseline entry point:

- `notebooks/read_data_sas.py`
- `notebooks/matrix_sas.py`
- `notebooks/pred_sas.py`
- `src/matrix.py`
- `src/signal_pred.py`

The canonical ASAG baseline path is the `src.asag_*` modules plus
`src.run_asag_baseline`.
