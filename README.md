## Project Option

This project follows **Option 2**: selecting a recent ACL Anthology paper with available code, reproducing the baseline, and improving the approach.

# Socks

Baseline setup for a course project on weakly supervised and interpretable
automated essay scoring (AES) and short answer scoring (SAS).

This repository contains both the AES pipeline across 8 essay sets and the
SAS pipeline across 10 question sets. The raw ASAP-AES dataset files are
expected to live locally under `AES/data/` and the raw ASAP-SAS dataset files
under `SAS/data/`.

## Project Layout

```text
.
├── AES/                        # Automated essay scoring pipeline
│   ├── 1/                      # Essay set 1
│   │   ├── bert/               # BERT-based classifier
│   │   ├── dummy/              # Dummy baselines (length, random)
│   │   ├── human/              # Human grading baseline
│   │   ├── llm/                # LLM-based predictions
│   │   │   ├── cot/            # Chain-of-thought prompting
│   │   │   ├── vanilla/        # Standard prompting
│   │   │   ├── vanilla_norm/   # Normalized vanilla prompting
│   │   │   ├── weak_signal/    # LLM-generated weak labels
│   │   │   └── weak_signal_norm/ # Normalized weak signal
│   │   ├── nllf_method/        # NLLF feature-based method
│   │   │   ├── bsq_lab/        # Binary subtask question labeling
│   │   │   ├── lr_Z_C/         # Positive linear regression model
│   │   │   └── nllf/           # Natural language learned features
│   │   ├── signal_clustering/  # Signal clustering weak labels
│   │   │   ├── none/           # No weak signal propagation
│   │   │   └── weak_signal/    # With weak signal propagation
│   │   ├── similarity/         # Similarity-based methods
│   │   │   ├── dense_pred.py   # Dense (embedding) similarity
│   │   │   └── sparse_pred.py  # Sparse (TF-IDF) similarity
│   │   └── ulra_paper/         # ULRA weak label method
│   │       ├── lf/             # Learned features
│   │       ├── none/           # No weak signal propagation
│   │       └── weak_signal/    # With weak signal propagation
│   ├── 2_1/ 2_2/ 3/ ... 8/    # Same structure for each essay set
│   ├── data/                   # ASAP-AES dataset files
│   ├── viz/                    # Visualizations
│   ├── read_data.ipynb         # Data reading and preprocessing notebook
│   └── results.ipynb           # Results evaluation notebook
├── SAS/                        # Short answer scoring pipeline
│   ├── 1/                      # Question set 1
│   │   ├── bert/               # BERT-based classifier
│   │   ├── dummy/              # Dummy baselines (length, random)
│   │   ├── llm/                # LLM-based predictions
│   │   │   ├── cot/            # Chain-of-thought prompting
│   │   │   ├── vanilla/        # Standard prompting
│   │   │   ├── vanilla_norm/   # Normalized vanilla prompting
│   │   │   └── weak_signal/    # LLM-generated weak labels
│   │   ├── nllf_method/        # NLLF feature-based method
│   │   │   ├── bsq_lab/        # Binary subtask question labeling
│   │   │   ├── lr_Z_C/         # Positive linear regression model
│   │   │   └── nllf/           # Natural language learned features
│   │   ├── signal_clustering/  # Signal clustering weak labels
│   │   │   ├── none/           # No weak signal propagation
│   │   │   └── weak_signal/    # With weak signal propagation
│   │   ├── similarity/         # Similarity-based methods
│   │   │   ├── dense_pred.py   # Dense (embedding) similarity
│   │   │   └── sparse_pred.py  # Sparse (TF-IDF) similarity
│   │   └── ulra_paper/         # ULRA weak label method
│   │       ├── lf/             # Learned features
│   │       ├── none/           # No weak signal propagation
│   │       └── weak_signal/    # With weak signal propagation
│   ├── 2/ ... 10/              # Same structure for each question set
│   ├── data/                   # ASAP-SAS dataset files
│   ├── read_data.ipynb         # Data reading and preprocessing notebook
│   └── results.ipynb           # Results evaluation notebook
├── README.md
└── requirements.txt
```

## Pipeline Overview

Both the AES and SAS pipelines follow the same three phases:

```text
Phase 1 — Weak Label Generation
  raw student responses
  -> signal_clustering/  (Chen et al., 2010)
  -> ulra_paper/         (Wang et al., 2023)
  -> llm/                (Jiang et al., 2023)

Phase 2 — Feature Extraction
  -> nllf_method/bsq_lab/   Natural Language Learned Features via BSQs (Urrutia et al., 2023)
  -> nllf_method/nllf/      NLLF representations
  -> expert features        (character count, word count, sentence length, etc.)

Phase 3 — Model Training and Analysis
  -> nllf_method/lr_Z_C/    positive linear regression with feature selection
  -> evaluation on test set with QWK / MAE / Pearson
  -> interpretability analysis of sparse weight structure
```

## Datasets

The ASAP-AES dataset was used for the AES tasks while the ASAP-SAS dataset
was used as the replacement dataset for the SAS tasks.

The ASAP-AES dataset can be downloaded from: https://www.kaggle.com/competitions/asap-aes/overview

The ASAP-SAS dataset can be downloaded from: https://www.kaggle.com/competitions/asap-sas/overview

Place the extracted ASAP-AES files under `AES/data/` and the extracted
ASAP-SAS files under `SAS/data/` before running any scripts.

## Setup

Install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Loading Data

**AES:** Open `AES/read_data.ipynb` to read and preprocess the ASAP-AES dataset.
This notebook splits the raw `train.tsv` into per-essay-set CSV files under
`AES/data/`:

```text
AES/data/essay_set_1.csv
AES/data/essay_set_2.csv
...
AES/data/essay_set_8.csv
```

**SAS:** Open `SAS/read_data.ipynb` to read and preprocess the ASAP-SAS dataset.
This notebook splits the raw `train_rel_2.tsv` into per-question CSV files under
`SAS/data/`:

```text
SAS/data/essay_set_1.csv
SAS/data/essay_set_2.csv
...
SAS/data/essay_set_10.csv
```

## Running the AES Pipeline

Each essay set folder (e.g. `AES/1/`, `AES/2_1/`, ...) follows the same
structure. Scripts are run from inside the relevant method subfolder.

**Phase 1 — Generate weak labels via signal clustering:**

```bash
cd AES/1/signal_clustering/weak_signal/
python run.py
```

**Phase 1 — Generate weak labels via ULRA:**

```bash
cd AES/1/ulra_paper/weak_signal/
python run.py
```

**Phase 1 — Generate weak labels via LLM:**

```bash
cd AES/1/llm/weak_signal/
python run.py
```

**Phase 3 — Train and evaluate the linear model:**

```bash
cd AES/1/nllf_method/lr_Z_C/
python pred.py
```

**Evaluate all methods:**

Open `AES/results.ipynb` to aggregate predictions and compute QWK / MAE /
Pearson metrics across all essay sets and methods.

## Running the SAS Pipeline

Each question set folder (e.g. `SAS/1/`, `SAS/2/`, ...) follows the same
structure. Scripts are run from inside the relevant method subfolder.

**Phase 1 — Generate weak labels via signal clustering:**

```bash
cd SAS/1/signal_clustering/weak_signal/
python run.py
```

**Phase 1 — Generate weak labels via ULRA:**

```bash
cd SAS/1/ulra_paper/weak_signal/
python run.py
```

**Phase 1 — Generate weak labels via LLM:**

```bash
cd SAS/1/llm/weak_signal/
python run.py
```

**Phase 3 — Train and evaluate the linear model:**

```bash
cd SAS/1/nllf_method/lr_Z_C/
python pred.py
```

**Evaluate all methods:**

Open `SAS/results.ipynb` to aggregate predictions and compute QWK / MAE /
Pearson metrics across all question sets and methods.

## Subdirectory Explanation

Each task folder under `AES/` or `SAS/` follows the same layout:

- `params.json` — model and pipeline hyperparameters
- `bert/pred.py` — BERT classifier prediction script
- `dummy/pred.py` — dummy baseline (length-based and random)
- `human/pred.py` — human grading baseline (AES only)
- `llm/{variant}/run.py` — LLM grading script for each prompting variant
- `llm/{variant}/prompt.json` — prompt template for that variant
- `similarity/` — similarity-based scoring (sparse and dense)
- `signal_clustering/{mode}/run.py` — signal clustering weak label generation
- `ulra_paper/{mode}/` — ULRA weak label generation and learned features
- `nllf_method/lr_Z_C/pred.py` — final positive linear regression predictor
- `*/pred/` — generated prediction CSV files (e.g. `test.csv`)

## Contributions

Helen Wang - Wrote the related works section of the report, added modified scripts for signal clustering (signal_pred and matrix) as well as dividing dataset data into subsections to follow project structure, added the contributions section to README, added read data script to divide SAS dataset into subsections, wrote dataset section of report, updated weak signal code to improve scores, updated README to include dataset section and links.

Harshvardhan Garude - Set up baseline project structure, added scripts for loading ASAP-AES dataset splits and preprocessing data with essay-set filtering and text cleaning, and updated README and data documentation for reproducibility.

Shravan Dhanasekaran - Replicated the full SAS pipeline from the original AES pipeline across all 10 ASAP-SAS question sets, including signal clustering, ULRA, LLM weak label generation, NLLF feature extraction, similarity-based scoring, BERT classification, and positive linear regression training and evaluation.
