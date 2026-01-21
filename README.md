# Explainable and Fair ML for ADRD

Dementia, a degenerative disease affecting millions globally, is projected to triple by 2050. Early and precise diagnosis is essential for effective treatment and improved quality of life. However, current diagnostic approaches frequently demonstrate inconsistent precision and impartiality, particularly among diverse cultural groups. This study investigates performance discrepancies in dementia classification among 6,584 White American, 1,263 African American, and 713 Hispanic populations. We reveal significant cross-group bias, particularly when models trained on one group are tested on another. To address this, we introduce a novel combination of few-shot learning and domain alignment to improve model adaptability across underrepresented populations. Our results show that these techniques substantially reduce inter-group performance gaps, especially between White American and Hispanic cohorts. This finding highlights the crucial need for fairness-aware strategies and the inclusion of diverse populations in training data to ensure accurate and equitable dementia diagnoses.

## What is included
- Core training logic (baseline, CR, KMM, SSDA, RegAlign)
- Models: XGBoost, SVM, MLP
- CLI: `dai train`, `dai predict`, `dai evaluate`, `dai prefit_kmm`, `dai compute_shap`
- Configs in `configs/` for defaults and grids
- Sample splits in `data_splits/` (train/test/external)

## Data splits
Sample data are drawn from:
`data_splits/merged_muse_ADNI_NACC_SCAN_STAC_HABSHD_Nov13_2025_baseline_QC_harmonized_phases.csv`

The split rule is:
- 100 rows for `train.csv` and 100 rows for `test.csv` from NACC + SCAN
- 100 rows for `external.csv` from ADNI + STAC

The resulting CSVs are in `data_splits/` and match the structure of
`data_splits/harmonized_test/balanced_split.csv` except:
- `ICV` is included for scale_mode=5 but excluded from training features
- `COHORT`, `PHASE`, `PTID`, `RID`, `MRI_ID` are integer-encoded

## Installation
From the repo root:

```bash
conda create -n dementia_ai python=3.11
conda activate dementia_ai
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -e .
```

Optional (for KMM via ADAPT):

```bash
pip install -e .[adapt]
```

## Neuroimaging preprocessing for T1w scans
- LPS reorientation
- N4 bias correction
- skull-stripping HD-BET: https://github.com/MIC-DKFZ/HD-BET
- MUSE feature extraction: via MUSE: https://github.com/CBICA/MUSE

## Quick start

Train a model (single run):

```bash
dai train \
  --train_csv data_splits/train.csv \
  --test_csv data_splits/test.csv \
  --external_csv data_splits/external.csv \
  --save_root runs \
  --scenario all \
  --method baseline \
  --model_name xgboost_vanilla \
  --n_repeats 1 \
  --run_range 0-0
```

Use the provided runner:

```bash
./run_train.sh
```

## CLI usage

### Train
`dai train` forwards arguments to `dementia_ai.training.trainer_small`.

Key options:
- `--train_csv`, `--test_csv`, `--external_csv`
- `--scenario` (all|white|black|hispanic|white+black|white+hispanic|black+hispanic)
- `--method` (baseline|cr|kmm|ssda|regalign)
- `--model_name` (xgboost_vanilla|svm|mlp)

If you provide `--data_csv`, the trainer will ignore `train/test` and do scenario-based splitting.

### Predict
Adds `y_score` and `y_prob` to a CSV using a trained model.

```bash
dai predict \
  --model_path runs/train_small/baseline_xgboost_vanilla/all/run_000/model.pkl \
  --input_csv data_splits/test.csv
```

### Evaluate
Computes group metrics and pairwise FNR/FPR gaps.

```bash
dai evaluate \
  --data_csv data_splits/test_pred.csv \
  --group_col ETHNICITY \
  --label_col DX \
  --score_col y_prob
```

### Prefit KMM
Train mode (save a KMM model + weights for a pair):

```bash
dai prefit_kmm train \
  --data_csv data_splits/train.csv \
  --group_col ETHNICITY \
  --pairs White->Black \
  --output_root checkpoints/kmm_models
```

Apply mode (use prefitted models to generate weights for all pairs):

```bash
dai prefit_kmm apply \
  --data_csv data_splits/train.csv \
  --group_col ETHNICITY \
  --pairs White->Black,White->Hispanic \
  --models_root checkpoints/kmm_models \
  --param_id kmm_default \
  --output_csv checkpoints/kmm_weights.csv
```

### Compute SHAP
TreeSHAP values for each feature. Uses results under `--results_root`.

```bash
dai compute_shap \
  --scenario all \
  --method_tag baseline_xgboost_vanilla \
  --results_root runs \
  --output_root runs/shap_small
```

## Configs
Defaults and grids live in:
- `configs/default.yaml`
- `configs/grids.yaml`

These are loaded by `dementia_ai.core.train_utils` for baseline/model tuning and method grids.

## Outputs
Training produces:
- `model.pkl`: trained model + scaler
- `predictions_rep.csv`: per-run predictions
- `summary_metrics_*.json`: metrics
- `sensitivity*.json`: method sensitivity records
- `meta.json`: run metadata
