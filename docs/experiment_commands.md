# Experiment Commands Guide

This document provides a concise guide to the main scripts used throughout the project, grouped by experimental phase. It is not intended to list every command executed during development, but to show representative examples of how the main stages of the pipeline were launched and reproduced.

## 1. Environment setup

Activate the virtual environment before running any script:

```bash
cd ~/projects/tfg-bracs
source .venv/bin/activate
```

To inspect MLflow runs locally on the server:

```bash
mlflow ui --backend-store-uri ./outputs/mlruns --host 0.0.0.0 --port 5000
```

Then create an SSH tunnel from the local machine:

```bash
ssh -L 5000:localhost:5000 bracs-ugr
```

---

## 2. Phase 1: Baseline models and embeddings

### Goal
Establish the initial patch-level baselines and extract foundation-model embeddings for subsequent experiments.

### Representative scripts
- `src/bracs/experiments/baseline/train_cnn_roi.py`
- `src/bracs/experiments/baseline/train_vit_roi.py`
- `src/bracs/experiments/baseline/train_foundation_roi.py`
- `src/bracs/experiments/phase1_baseline/extract_foundation_embeddings.py`
- `src/bracs/experiments/phase1_baseline/train_linear_on_clean_embeddings.py`

### Example command: extract embeddings
```bash
python src/bracs/experiments/phase1_baseline/extract_foundation_embeddings.py \
  --model virchow2 \
  --split train
```

### Example command: train linear head on embeddings
```bash
python src/bracs/experiments/phase1_baseline/train_linear_on_clean_embeddings.py \
  --model virchow2 \
  --n_clases 7 \
  --train_h5 outputs/embeddings/virchow2_train.h5 \
  --val_h5 outputs/embeddings/virchow2_val.h5 \
  --seed 42 \
  --epochs 10 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --scheduler cosine
```

---

## 3. Phase 2: Patch cleaning

### Goal
Study whether removing less informative or potentially noisy patches improves downstream performance.

### Representative scripts
- `src/bracs/experiments/phase2_patch_cleaning/clean_embeddings_MI.py`
- `src/bracs/experiments/phase2_patch_cleaning/apply_random_under_embeddings.py`
- `src/bracs/experiments/phase2_patch_cleaning/apply_ncr_embeddings.py`

### Example command: random undersampling
```bash
python src/bracs/experiments/phase2_patch_cleaning/apply_random_under_embeddings.py \
  --input_h5 outputs/embeddings/virchow2_train.h5 \
  --output_h5 outputs/cleaned_embeddings/random_under/virchow2_train_random_under.h5
```

### Example command: NCR
```bash
python src/bracs/experiments/phase2_patch_cleaning/apply_ncr_embeddings.py \
  --input_h5 outputs/embeddings/virchow2_train.h5 \
  --output_h5 outputs/cleaned_embeddings/ncr/virchow2_train_ncr.h5
```

---

## 4. Phase 3: ROI-level evaluation

### Goal
Aggregate patch-level predictions into ROI-level predictions and compare voting rules.

### Representative scripts
- `src/bracs/experiments/phase3_roi_evaluation/aggregate_patch_predictions_to_roi.py`
- `src/bracs/experiments/phase3_roi_evaluation/evaluate_roi_predictions.py`
- `src/bracs/experiments/phase3_roi_evaluation/build_roi_predictions_3cls_from_7cls.py`
- `src/bracs/experiments/phase3_roi_evaluation/evaluate_test_roi_3cls.py`

### Example command: aggregate patch predictions to ROI
```bash
python src/bracs/experiments/phase3_roi_evaluation/aggregate_patch_predictions_to_roi.py \
  --input_csv outputs/predictions/test_patches/virchow2/baseline_patch_predictions.csv \
  --output_csv outputs/predictions/test_roi/virchow2/baseline_roi_predictions.csv \
  --n_clases 7
```

### Example command: evaluate ROI predictions
```bash
python src/bracs/experiments/phase3_roi_evaluation/evaluate_roi_predictions.py \
  --input_csv outputs/predictions/test_roi/virchow2/baseline_roi_predictions.csv \
  --voting_method mean_proba \
  --output_dir outputs/metrics/test_roi/virchow2 \
  --n_clases 7
```

---

## 5. Phase 4: Abstention and doubtful-case analysis

### Goal
Incorporate a review mechanism so that the model can abstain on highly uncertain ROIs instead of forcing a prediction.

### Representative scripts
- `src/bracs/experiments/phase4_abstention/evaluate_roi_predictions_with_abstention.py`
- `src/bracs/experiments/phase4_abstention/summarize_roi_abstention_reviews.py`
- `src/bracs/experiments/phase4_abstention/select_roi_case_studies.py`
- `src/bracs/experiments/phase4_abstention/plot_roi_case_studies.py`

### Example command: evaluate abstention
```bash
python src/bracs/experiments/phase4_abstention/evaluate_roi_predictions_with_abstention.py \
  --input_csv outputs/predictions/test_roi/virchow2/random_under_roi_predictions.csv \
  --tau 0.10 \
  --output_dir outputs/metrics/test_roi_abstention/virchow2
```

---

## 6. Phase 5: Extra analyses

### Goal
Further analyse the structure of the problem beyond standard supervised evaluation.

### Representative scripts
- `src/bracs/experiments/phase5_extra_analysis/run_unsupervised_kmeans_analysis.py`
- `src/bracs/experiments/phase5_extra_analysis/build_test_roi_embeddings_from_patch_embeddings.py`
- `src/bracs/experiments/phase5_extra_analysis/select_clear_roi_prototypes.py`
- `src/bracs/experiments/phase5_extra_analysis/find_nearest_clear_prototypes.py`
- `src/bracs/experiments/phase5_extra_analysis/summarize_prototype_neighbors.py`
- `src/bracs/experiments/phase5_extra_analysis/select_prototype_case_candidates.py`

### Example command: unsupervised analysis
```bash
python src/bracs/experiments/phase5_extra_analysis/run_unsupervised_kmeans_analysis.py \
  --model virchow2 \
  --train_h5 outputs/embeddings/virchow2_train.h5
```

---

## 7. Phase 6: Feature selection

### Goal
Assess whether a reduced subset of embedding dimensions can maintain ROI-level performance.

### Representative scripts
- `src/bracs/experiments/phase6_feature_selection/run_hyindex_feature_selection.py`
- `src/bracs/experiments/phase6_feature_selection/run_mrmr_feature_selection.py`
- `src/bracs/experiments/phase6_feature_selection/build_reduced_embedding_h5.py`

### Example command: mRMR feature selection
```bash
python src/bracs/experiments/phase6_feature_selection/run_mrmr_feature_selection.py \
  --train_h5 outputs/embeddings/virchow2_train.h5 \
  --output_dir outputs/mrmr/virchow2_k652 \
  --model virchow2 \
  --k_features 652 \
  --random_state 42
```

### Example command: rebuild reduced H5
```bash
python src/bracs/experiments/phase6_feature_selection/build_reduced_embedding_h5.py \
  --input_h5 outputs/embeddings/virchow2_train.h5 \
  --selected_idx_npy outputs/mrmr/virchow2_k652/virchow2_mrmr_selected_idx_k652.npy \
  --output_h5 outputs/embeddings_reduced/virchow2/virchow2_train_mrmr_k652.h5
```

---

## 8. Notes

- Large datasets, checkpoints and generated outputs are intentionally excluded from version control.
- The commands above are representative examples and may require adapting paths or parameters depending on the experiment being reproduced.
- For a higher-level description of the experimental pipeline, see the project report and the repository README.
