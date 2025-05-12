# Representative and Explainable AI for Multi-Ethnic Dementia Pattern Recognition

## 🧠 Abstract

Dementia, a degenerative disease affecting millions globally, is projected to triple in prevalence by 2050. Early and precise diagnosis is essential for effective treatment and improved quality of life. However, current diagnostic approaches may show inconsistent precision and fairness, particularly among diverse cultural groups.

This research investigates the fairness of machine learning (ML) models in dementia classification across multi-ethnic populations, specifically focusing on White American, African American, and Hispanic groups. A gradient boosting classifier is employed to assess classification performance using discrepancy mitigation techniques such as:

- **Data harmonization**
- **Correlation removal (CR)**
- **Domain adaptation (Kernel Mean Matching, KMM)**
- **Semi-supervised domain adaptation**
- **Few-shot domain alignment**

We find significant performance discrepancies when models trained on one group are tested on others, particularly between African American and Hispanic populations. Discrepancy mitigation techniques, especially CR, KMM, and few-shot learning, reduce performance differences—most notably between White American and Hispanic groups. These findings underscore the importance of representative training data and mitigation strategies in developing equitable AI models for dementia diagnosis.

---

## 🧪 Scripts Overview

### `train_ml_gridsearch_cv.py`

Train ML classifiers (XGBoost or SVM) with cross-validation and domain adaptation. Supports harmonization, correlation removal, and KMM.

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


### `train_ml_gridsearch_cv.py`

Train XGBoost models with custom objectives for:
- Few-shot learning (e.g., subtype '1B')
- Semi-supervised learning (e.g., subtype '2B')

#### Key Arguments:
```python
--save_model_path        Path to save models  
--save_result_path       Path to save results  
--train_approach         Strategy: 'few_shot' or 'semi_supervised'  
--objective_version      Version of custom objective: 'v1' or 'v2'  
--objective_subtype      Custom loss variant: '1A', '1B', '1C', '2A', or '2B'  

