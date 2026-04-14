# Data

This directory is for local dataset files used by the AES project.

## Current Contents

The local dataset currently includes an `asap-aes/` folder containing the
extracted ASAP AES dataset files:

- `training_set_rel3.tsv`, plus Excel copies, used as the training split
- `valid_set.tsv`, plus Excel copies, used as the validation split
- `test_set.tsv`, used as the test split
- `valid_sample_submission_*.csv`, small sample submission templates
- `Essay_Set_Descriptions.zip`, containing essay set description documents
- `Training_Materials.zip` and `Training_Materials/`, containing scoring
  guides, practice sets, and related documentation

The `asap-aes/` directory is intentionally ignored by git because it contains
raw dataset files and large extracted documents. Keep it local unless the team
explicitly decides to track a small, safe derivative file.

## Expected Usage

Do not move, rename, or delete the raw dataset files during baseline setup.
Scripts or notebooks should read from the local `data/asap-aes/` directory
through the reusable loader in `src.data_loading`.

Generated preprocessing outputs should be written to `results/processed/`, not
back into `data/`. The `results/` directory is ignored by git so local
experiment outputs do not become part of the repository by accident.
