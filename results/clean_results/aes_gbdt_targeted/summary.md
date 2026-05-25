# AES Targeted GBDT Experiment

This AES-only experiment selects GBDT essay sets using validation metrics, then reports test metrics after selection.

## Baseline

- test: QWK 0.568899, MAE 1.624826, Pearson 0.707871.
- val: QWK 0.605201, MAE 1.687453, Pearson 0.727802.

## Selected Sets

- `qwk_only_selection`: 1, 7.
- `balanced_selection`: 1, 7.
- `conservative_selection`: 1, 7.
- `manual_safe_group`: 1, 6, 7.
- `val_selected_per_essay_set_model`: 1, 2, 3, 4, 5, 6, 7, 8.
- `manual_safe_group_recommended_calibration`: 6.

## Best Results

- Best clean validation candidate: `gbdt_manual_safe_group_plus_recommended_calibration` with val QWK 0.619918, MAE 1.669650, Pearson 0.727614.
- Best clean test candidate: `gbdt_manual_safe_group_plus_recommended_calibration` with test QWK 0.581297, MAE 1.619947, Pearson 0.714240.
- Set-6-only candidate: test QWK 0.569722, MAE 1.620763, Pearson 0.712364.
- Validation-selected candidate: test QWK 0.572180, MAE 1.627202, Pearson 0.713838.

## Per-Set Diagnostics

- Set 1: best GBDT `sklearn_gbdt_all_sets`, val delta QWK +0.012093, val delta MAE -0.023750, recommendation `use_gbdt`.
- Set 2: best GBDT `sklearn_gbdt_all_sets`, val delta QWK -0.013458, val delta MAE +0.024522, recommendation `use_baseline`.
- Set 3: best GBDT `hist_gbdt_all_sets`, val delta QWK -0.091285, val delta MAE +0.002069, recommendation `use_baseline`.
- Set 4: best GBDT `hist_gbdt_all_sets`, val delta QWK -0.076000, val delta MAE +0.011918, recommendation `use_baseline`.
- Set 5: best GBDT `hist_gbdt_all_sets`, val delta QWK -0.024745, val delta MAE +0.012847, recommendation `use_baseline`.
- Set 6: best GBDT `hist_gbdt_all_sets`, val delta QWK +0.000270, val delta MAE -0.014003, recommendation `use_gbdt_plus_calibration`.
- Set 7: best GBDT `hist_gbdt_all_sets`, val delta QWK +0.026886, val delta MAE -0.059965, recommendation `use_gbdt`.
- Set 8: best GBDT `sklearn_gbdt_all_sets`, val delta QWK -0.046123, val delta MAE +0.696609, recommendation `avoid`.

## Feature Importance

- `unique_word_count` mean importance 0.432601.
- `punctuation_count` mean importance 0.034709.
- `digit_count` mean importance 0.017281.
- `average_word_length` mean importance 0.010558.
- `character_count` mean importance 0.010542.
- `long_word_count` mean importance 0.009248.

## Mistral Subset Ablation

- skipped: No cached AES Mistral label file was found in this worktree; no API calls were made.

## Recommendation

- A targeted GBDT candidate is clean enough to consider extracting.
- Do not apply GBDT globally.
