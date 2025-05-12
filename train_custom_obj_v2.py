import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, dump
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import xgboost as xgb

# Import custom objectives and utilities.
sys.path.append("/home/Codes/ad_classification")
from custom_objectives import CustomObjective, CustomObjectiveV2, sigmoid
from train_utils import select_few_shot_target

def train_model_on_params(param_tuple, combined_X, combined_y, scaler,
                          X_target_eval, y_target_eval, total_steps,
                          n_source, m_target, train_approach, objective_version='v1', objective_subtype='1A'):
    (max_depth, eta, reg_lambda, reg_alpha, weight) = param_tuple
    params_current = {
        'max_depth': max_depth,
        'eta': eta,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'verbosity': 0,
        'lambda': reg_lambda,
        'alpha': reg_alpha,
        'scale_pos_weight': weight,
    }
    if objective_version == 'v1':
        custom_obj = CustomObjective(
            total_steps=total_steps,
            n_source=n_source,
            m_target=m_target,
            tau=1.0,
            eps=1e-7,
            approach=train_approach  # 'few_shot' or 'unsupervised'
        )
    elif objective_version == 'v2':
        custom_obj = CustomObjectiveV2(
            total_steps=total_steps,
            n_source=n_source,
            m_target=m_target,
            tau=1.5,
            eps=1e-7,
            source_focal_gamma=2.0,
            target_focal_gamma=2.0,
            use_target_reg=False,
            target_weight=1.0,
            approach=objective_subtype,  # One of '1A', '1B', '1C', '2A', '2B'
            X_combined=combined_X,
            y_source=combined_y[:n_source],
        )
    else:
        raise ValueError("Unsupported objective_version. Use 'v1' or 'v2'.")
    
    dtrain = xgb.DMatrix(combined_X, label=combined_y)
    X_target_eval_norm = scaler.transform(X_target_eval.values)
    X_eval_subset, _, y_eval_subset, _ = train_test_split(
        X_target_eval_norm, y_target_eval, test_size=0.9, random_state=42, stratify=y_target_eval
    )
    dtest = xgb.DMatrix(X_eval_subset, label=y_eval_subset)
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params_current, dtrain, total_steps, obj=custom_obj,
                      evals=evallist, early_stopping_rounds=100, verbose_eval=False)
    X_target_eval_norm = scaler.transform(X_target_eval.values)
    dtest_eval = xgb.DMatrix(X_target_eval_norm)
    preds_prob = model.predict(dtest_eval, iteration_range=(0, model.best_iteration + 1))
    y_pred = (sigmoid(preds_prob) >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_target_eval, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_target_eval, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    return (bal_acc, fpr, fnr, params_current, model, preds_prob)

def train_pipeline():
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    scenarios = ['hisp'] #['all', 'nhw', 'nha', 'hisp']
    folds = range(1, 11)
    
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'eta': [0.1, 0.3, 0.5, 0.7, 1],
        'lambda': [1, 3, 5, 7],
        'alpha': [0, 1, 3, 5],
        'scale_pos_weight': [1., 1.15, 1.3, 1.45]
    }
    param_combinations = list(itertools.product(
        param_grid['max_depth'],
        param_grid['eta'],
        param_grid['lambda'],
        param_grid['alpha'],
        param_grid['scale_pos_weight']
    ))
    
    total_steps = 5000
    save_model = args.save_model_path
    save_result = args.save_result_path
    
    # Objective choice.
    objective_version = args.objective_version  # 'v1' or 'v2'
    if objective_version == 'v1':
        train_approach = args.train_approach # 'few_shot' or 'unsupervised'
    else:
        train_approach = None  # Not used for v2; use objective_subtype instead.
    objective_subtype = args.objective_subtype  # For v2: '1A', '1B', '1C', '2A', or '2B'
    
    num_few_shot = 5  # number of target samples per group.
    
    # Load CSV data.
    train_file_prefix = "scenario_"
    # In your pipeline, files are inside data/split_fold_w_augmented.
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        for fold in tqdm(folds, desc="Folds", leave=False):
            print(f"\n--- Training for scenario: {scenario}, fold: {fold} ---")
            train_file = f"{train_file_prefix}{scenario}_fold_{fold}_train.csv"
            test_file = f"{train_file_prefix}{scenario}_fold_{fold}_test.csv"
            train_df = pd.read_csv(os.path.join(source_dir, "data", "split_fold_w_augmented", train_file))
            test_df = pd.read_csv(os.path.join(source_dir, "data", "split_fold_w_augmented", test_file))
            
            non_feature_cols = ["ID", "NACCADC", "NACCMRIA", "SEX", "HISPANIC", "RACE", "RACE_HISP", "NACCUDSD"]
            feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
            
            source_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            target_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            X_source = source_df[feature_cols]
            y_source = source_df["NACCUDSD"].values.astype(int)
            X_target_eval = target_df[feature_cols]
            y_target_eval = target_df["NACCUDSD"].values.astype(int)
            
            scaler = StandardScaler()
            X_source_norm = scaler.fit_transform(X_source.values)
            
            if objective_version == 'v1':
                if train_approach == 'few_shot':
                    target_few_shot = select_few_shot_target(target_df, scenario, num_few_shot)
                    X_target_train = target_few_shot[feature_cols]
                    y_target_train = target_few_shot["NACCUDSD"].values.astype(int)
                else:
                    X_target_train = target_df[feature_cols]
                    y_target_train = target_df["NACCUDSD"].values.astype(int)
            else:
                if objective_subtype in ['1A', '1B', '1C']:
                    target_few_shot = select_few_shot_target(target_df, scenario, num_few_shot)
                    X_target_train = target_few_shot[feature_cols]
                    y_target_train = target_few_shot["NACCUDSD"].values.astype(int)
                else:
                    X_target_train = target_df[feature_cols]
                    y_target_train = target_df["NACCUDSD"].values.astype(int)
            
            X_target_train_norm = scaler.transform(X_target_train.values)
            combined_X = np.vstack([X_source_norm, X_target_train_norm])
            combined_y = np.concatenate([y_source, y_target_train])
            
            n_source = X_source_norm.shape[0]
            m_target = X_target_train_norm.shape[0]
            
            results = Parallel(n_jobs=15, backend='loky')(
                delayed(train_model_on_params)(
                    params, combined_X, combined_y, scaler, X_target_eval, y_target_eval,
                    total_steps, n_source, m_target,
                    train_approach if objective_version == 'v1' else None,
                    objective_version, objective_subtype
                ) for params in tqdm(param_combinations, desc="Hyperparams", leave=False)
            )
            
            best_score = -np.inf
            best_params = None
            best_model = None
            best_preds_prob = None
            best_metrics = None
            for res in results:
                bal_acc, fpr, fnr, params_current, model, preds_prob = res
                if bal_acc > best_score:
                    best_score = bal_acc
                    best_params = params_current
                    best_model = model
                    best_preds_prob = preds_prob
                    best_metrics = (bal_acc, fpr, fnr)
            
            # Save best model and predictions.
            model_filename = f"scenario_{scenario}_fold_{fold}_model.json"
            model_save_dir = os.path.join(save_model, f"xgb_objective_{objective_version}_{train_approach if objective_version == 'v1' else objective_subtype}")
            os.makedirs(model_save_dir, exist_ok=True)
            best_model.save_model(os.path.join(model_save_dir, model_filename))
            
            pred_df = target_df[["ID", "RACE", "HISPANIC", "NACCUDSD"]].copy()
            pred_df["PREDICTION"] = best_preds_prob
            predict_filename = f"scenario_{scenario}_fold_{fold}_predict.csv"
            pred_save_dir = os.path.join(save_result, f"xgb_objective_{objective_version}_{train_approach if objective_version == 'v1' else objective_subtype}")
            os.makedirs(pred_save_dir, exist_ok=True)
            pred_df.to_csv(os.path.join(pred_save_dir, predict_filename), index=False)
            
            print(f"Overall Performance for Scenario: {scenario}, Fold: {fold}")
            print(f"Balanced Accuracy: {best_metrics[0]:.4f}")
            print(f"False Positive Rate: {best_metrics[1]:.4f}")
            print(f"False Negative Rate: {best_metrics[2]:.4f}")
            print(f"Best Hyperparameters: {best_params}\n")
            
            # (Optional) Compute subgroup performance...
            subgroups = {
                "NHW": (target_df["RACE"] == 1) & (target_df["HISPANIC"] == 0),
                "NHA": (target_df["RACE"] == 2) & (target_df["HISPANIC"] == 0),
                "Hispanic": (target_df["RACE"] == 1) & (target_df["HISPANIC"] == 1)
            }
            print("Subgroup Performance:")
            for name, condition in subgroups.items():
                idx = condition.values
                if np.sum(idx) == 0:
                    print(f"{name}: No samples available.")
                    continue
                y_true_sub = y_target_eval[idx]
                y_pred_sub = (sigmoid(best_preds_prob) >= 0.5).astype(int)[idx]
                sub_bal_acc = balanced_accuracy_score(y_true_sub, y_pred_sub)
                tn_sub, fp_sub, fn_sub, tp_sub = confusion_matrix(y_true_sub, y_pred_sub).ravel()
                sub_fpr = fp_sub / (fp_sub + tn_sub + 1e-6)
                sub_fnr = fn_sub / (fn_sub + tp_sub + 1e-6)
                print(f"{name}: Balanced Accuracy: {sub_bal_acc:.4f}, FPR: {sub_fpr:.4f}, FNR: {sub_fnr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training AD Classification with Domain Adaptation")
    parser.add_argument("--save_model_path", type=str, default='/home/Codes/ad_classification/models', help="Path to save models")
    parser.add_argument("--save_result_path", type=str, default='/home/Codes/ad_classification/results', help="Path to save results")
    parser.add_argument("--train_approach", type=str, default='few_shot', help="Training strategy")
    parser.add_argument("--objective_version", type=str, default='v1', help="Custom objective version: v1 or v2")
    parser.add_argument("--objective_subtype", type=str, default='1A', help="For v2: one of '1A', '1B', '1C', '2A', or '2B'")
    
    args = parser.parse_args()
    train_pipeline()
