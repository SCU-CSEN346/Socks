## Project Option

This project follows **Option 2**: selecting a recent ACL Anthology paper with available code, reproducing the baseline, and improving the approach.

# Socks



Baseline setup for a course project on weakly supervised and interpretable
automated essay scoring.

This repository is being organized around a small, reviewable baseline. The
raw ASAP-AES dataset is expected to live locally under `data/asap-aes/`, while
source code, notebooks, and generated results are kept separate.

## Status

Baseline setup is in progress. A minimal ASAP-AES loading and preprocessing
pipeline is available, but the training, evaluation, and reporting pipeline has
not been implemented yet.

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

The raw dataset should remain extracted at:

```text
data/asap-aes/
```

That directory is ignored by git.

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

Helen Wang - Wrote the related works section of the report, added modified scripts for signal clustering (signal_pred and matrix) as well as dividing dataset data into subsections to follow project structure.

Harshvardhan Garude - Set up baseline project structure, added scripts for loading ASAP-AES dataset splits (data_loading) and preprocessing data (preprocess_asap) with essay-set filtering and text cleaning, and updated README and data documentation for reproducibility.
