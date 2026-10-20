# Explainable and Fair ML for ADRD

Code accompanying the paper [**Advancing fair and explainable machine learning for neuroimaging dementia pattern classification in multi-racial and multi-ethnic populations**](https://www.nature.com/articles/s41467-026-74515-w), published in *Nature Communications* **17**, 8026 (2026). [DOI: 10.1038/s41467-026-74515-w](https://doi.org/10.1038/s41467-026-74515-w).

Dementia, a degenerative disease affecting millions globally, is projected to triple by 2050. Early and precise diagnosis is essential for effective treatment and improved quality of life. However, current diagnostic approaches frequently demonstrate inconsistent precision and impartiality, particularly among diverse cultural groups. This study investigates performance discrepancies in dementia classification among 6,584 White American, 1,263 African American, and 713 Hispanic populations. We reveal significant cross-group bias, particularly when models trained on one group are tested on another. To address this, we introduce a novel combination of few-shot learning and domain alignment to improve model adaptability across underrepresented populations. Our results show that these techniques substantially reduce inter-group performance gaps, especially between White American and Hispanic cohorts. This finding highlights the crucial need for fairness-aware strategies and the inclusion of diverse populations in training data to ensure accurate and equitable dementia diagnoses.

## What is included
- Core training logic (baseline, CR, KMM, SSDA, RegAlign)
- Models: XGBoost, SVM, MLP
- CLI: `dai train`, `dai predict`, `dai evaluate`, `dai prefit_kmm`, `dai compute_shap`
- Configs in `configs/` for defaults and grids
- Sample splits in `data_splits/` (train/test/external)


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
  --save_root results \
  --scenario all \
  --method baseline \
  --model_name xgboost_vanilla \
  --cv_splits 3 \
  --n_repeats 2 \
  --run_range 0-1
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
  --model_path results/train_small/baseline_xgboost_vanilla/all/run_000/model.pkl \
  --input_csv data_splits/test.csv \
  --output_csv results/train_small/baseline_xgboost_vanilla/all/run_000/predictions/test_pred.csv
```

### Evaluate
Computes group metrics and pairwise FNR/FPR gaps.

```bash
dai evaluate \
  --data_csv results/train_small/baseline_xgboost_vanilla/all/run_000/predictions/test_pred.csv \
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
  --pairs 'White->Black' \
  --output_root checkpoints/kmm_models
```

Apply mode (use prefitted models to generate weights for a pair):

```bash
dai prefit_kmm apply \
  --data_csv data_splits/train.csv \
  --group_col ETHNICITY \
  --pairs 'White->Black' \
  --models_root checkpoints/kmm_models \
  --param_id kmm_default \
  --output_csv checkpoints/kmm_weights.csv
```

### Compute SHAP
TreeSHAP values for each feature. Uses results under `--results_root`.
https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

```bash
dai compute_shap \
  --scenario all \
  --method_tag baseline_xgboost_vanilla \
  --results_root results \
  --output_root results/shap_small
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

## Citation

If you use this code or the methods in your research, please cite:

Ho, N.-H., Charisis, S., Honnorat, N. et al. Advancing fair and explainable machine learning for neuroimaging dementia pattern classification in multi-racial and multi-ethnic populations. *Nature Communications* **17**, 8026 (2026). https://doi.org/10.1038/s41467-026-74515-w

```bibtex
@article{ho2026advancing,
  title = {Advancing fair and explainable machine learning for neuroimaging dementia pattern classification in multi-racial and multi-ethnic populations},
  author = {Ho, Ngoc-Huynh and Charisis, Sokratis and Honnorat, Nicolas and Brandigampala, Sachintha Ransara and Wang, Di and Heckbert, Susan R. and Fox, Peter T. and Martinez, David and Wang, David H. and Hughes, Timothy M. and Archer, Derek B. and Hohman, Timothy J. and Seshadri, Sudha and Davatzikos, Christos and Habes, Mohamad},
  journal = {Nature Communications},
  volume = {17},
  number = {1},
  pages = {8026},
  year = {2026},
  doi = {10.1038/s41467-026-74515-w},
  url = {https://doi.org/10.1038/s41467-026-74515-w}
}
```
