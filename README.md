# Advancing Fair and Explainable Machine Learning for Neuroimaging Dementia Pattern Classification in Multi-Ethnic Populations

## 🧠 Abstract

Current diagnostic approaches may show inconsistent precision and fairness, particularly among diverse cultural groups.

This research investigates the fairness of machine learning (ML) models in dementia classification across multi-ethnic populations, specifically focusing on White American, African American, and Hispanic groups. A gradient boosting classifier is employed to assess classification performance using discrepancy mitigation techniques such as:

- **Data harmonization**
- [**Correlation remover (CR)**](https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.CorrelationRemover.html)
- [**Domain adaptation (Kernel Mean Matching, KMM)**](http://adapt-python.github.io/adapt/generated/adapt.instance_based.KMM.html)
- **Semi-supervised domain adaptation**
- **Few-shot domain alignment**

We find significant performance discrepancies when models trained on one group are tested on others, particularly between African American and Hispanic populations. Discrepancy mitigation techniques, especially CR, KMM, and few-shot learning, reduce performance differences—most notably between White American and Hispanic groups. These findings underscore the importance of representative training data and mitigation strategies in developing equitable AI models for dementia diagnosis.

---

## 🔧 Installation
```
conda env create -f environment.yml
conda activate dementia_ai
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 🧪 Scripts Overview

### `train_ml_gridsearch_cv.py`

Train ML classifiers (XGBoost or SVM) with cross-validation and domain adaptation. Supports harmonization, correlation remover, and KMM.

#### Key Arguments:
```python
--save_model_path         Path to save trained models  
--save_result_path        Path to save evaluation results  
--df_roi                  Path to harmonized MUSE ROI CSV  
--train_type              Training group: All, NHW, NHA, or Hispanic  
--train_drop_rate         Percentage of training data to drop  
--model_name              Classifier: 'xgboost' or 'svm'  
--sample_weight_method    Domain adaptation method: e.g., 'kmm'  
--n_splits                Number of CV folds  
--data_scaling            Whether to normalize data (0/1/2/3/4)  
--corr_remove             Whether to remove correlated features  
--corr_alpha              Threshold for correlation removal  
--harmonize               Whether to apply harmonization (0/1)  
--sim_threshold           Similarity threshold for KMM  
```

### `train_custom_obj.py`

Train XGBoost models with custom objectives for:
- Few-shot learning (e.g., subtype '1B')
- Semi-supervised learning (e.g., subtype '2B')

#### Key Arguments:
```python
--scenario               Training scenario
--save_model_path        Path to save models  
--save_result_path       Path to save results 
--objective_subtype      Custom loss variant: '1C' or '2B'  
--num_few_shot
--total_steps
--inner_splits
--outer_splits
--n_job
```

### Training
Revise the hyper-parameters in `run_python.sh` first.
Example:
```
bash run_python.sh mode1
bash run_python.sh mode2
```
- mode1: train with baseline, CR, KMM, data harmonization
- mode2: train with SSDA (1C) and RegAlign (2B)

### `visualize.ipynb`
Notebook for visualizing key results:
- Performance metrics (Balanced Accuracy, FPR, FNR)
- Pareto front plots for trade-offs between accuracy and fairness
- Confidence interval plots
- SHAP heatmaps of regional brain contributions
- Partial dependence plots to interpret model behavior between SHAP values and brain volumes

## 📂 Folder Structure (Recommended)
```bash
dementia_classification/
├── models/                  
├── results/                 
├── data/
├──── split_fold_w_augmented
├──── filtered_data_09182024.csv
├──── filtered_data_09182024_Harmonized_DX.csv
├── train_ml_gridsearch_cv.py
│   train_custom_obj.py
│   ops.py
│   models.py
│   custom_objectives.py
├── visualize.ipynb
│   viz_utils.sh
└── README.md
```

## 👨‍⚕️ Authors
- Dr. Ngoc-Huynh Ho (Henry) – Postdoctoral Researcher, The Biggs Institute for Alzheimer's & Neurodegenerative Diseases
- Advisor: Prof. Mohamad Habes
