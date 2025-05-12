import sys
sys.path.append("/home/Codes/ad_classification")

import os
import pickle
import numpy as np
import torch
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

import ops
import models as ML
from shap_utils import custom_tree_shap_batch_torch, get_xgboost_predictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUR_DIR = "/home/Codes/ad_classification/results/shap_values"

def load_pickle_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def load_csv_data(data_dir, scenario, fold):
    train_path = os.path.join(data_dir, f"scenario_{scenario}_fold_{fold}_train.csv")
    test_path = os.path.join(data_dir, f"scenario_{scenario}_fold_{fold}_test.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def load_xgb_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def compute_shap_for_fold(fold_idx, X_train_raw, X_test_raw, model, feature_columns, scaling_mode=4, batch_size=100):
    X_train = pd.DataFrame(ops.scaling_data(X_train_raw, None, scaling_mode), columns=feature_columns)
    X_test = pd.DataFrame(ops.scaling_data(X_test_raw, None, scaling_mode), columns=feature_columns)

    predict_fn = get_xgboost_predictor(model, internal_batch_size=batch_size)

    shap_vals = np.zeros((X_test.shape[0], len(feature_columns)))
    for start in tqdm(range(0, X_test.shape[0], batch_size), desc=f"Fold {fold_idx} SHAP", leave=False):
        end = min(start + batch_size, X_test.shape[0])
        shap_vals[start:end] = custom_tree_shap_batch_torch(
            predict_fn, X_test[start:end].values, X_train.values,
            filter_column_idx=list(range(len(feature_columns))),
            device=device
        )
    return np.nan_to_num(shap_vals, nan=0, posinf=1e10, neginf=-1e10)

def compute_shap_from_pickle(pickle_file, train_mode, model_type, scaling_mode=4, batch_size=100):
    train_data, nhw, nha, hwa, est = load_pickle_data(pickle_file)
    feature_columns = list(train_data[0]['data'].iloc[:, 1:].columns) if train_mode == "all" else list(train_data[0]['data_1'].iloc[:, 1:].columns)

    shap_values_all_folds = []
    for fold_idx in tqdm(range(len(train_data)), desc=f"[{model_type.upper()}] SHAP by Fold"):
        if train_mode == "all":
            X_train_raw = train_data[fold_idx]['data'].iloc[:, 1:]
            model = est[fold_idx].predictor
        else:
            X_train_raw = train_data[fold_idx]['data_1'].iloc[:, 1:]
            model = est[fold_idx][0].predictor

        X_nhw_raw = nhw[fold_idx]['data'].iloc[:, 1:]
        X_nha_raw = nha[fold_idx]['data'].iloc[:, 1:]
        X_hwa_raw = hwa[fold_idx]['data'].iloc[:, 1:]
        X_test_raw = pd.concat([X_nhw_raw, X_nha_raw, X_hwa_raw], axis=0)

        shap_vals = compute_shap_for_fold(fold_idx, X_train_raw, X_test_raw, model, feature_columns, scaling_mode, batch_size)
        shap_values_all_folds.append(shap_vals)

    save_dir = f"{OUR_DIR}/{model_type}/{train_mode}/"
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, os.path.basename(pickle_file).replace(".pkl", "_shap.pkl"))
    with open(out_file, "wb") as f:
        pickle.dump(shap_values_all_folds, f)
    print(f"SHAP values saved to {out_file}")

def compute_shap_from_csv(scenario, data_dir, model_dir, model_type, scaling_mode=4, batch_size=100):
    shap_values_all_folds = []
    for fold_idx in tqdm(range(1, 11), desc=f"[{model_type.upper()}] Scenario {scenario.upper()}"):
        train_df, test_df = load_csv_data(data_dir, scenario, fold_idx)
        X_train_raw = train_df.iloc[:, 8:]  # Features start from 9th column
        X_test_raw = test_df.iloc[:, 8:]
        feature_columns = list(X_train_raw.columns)

        model_path = os.path.join(model_dir, f"scenario_{scenario}_fold_{fold_idx}_model.xgb")
        model = load_xgb_model(model_path)

        shap_vals = compute_shap_for_fold(fold_idx, X_train_raw, X_test_raw, model, feature_columns, scaling_mode, batch_size)
        shap_values_all_folds.append(shap_vals)

    save_dir = f"{OUR_DIR}/{model_type}/"
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f"scenario_{scenario}_shap.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(shap_values_all_folds, f)
    print(f"SHAP values saved to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage for pickle mode: python shap_computation_direct.py <pickle_file> <train_mode> <model_type> [scaling_mode]")
        print("Usage for CSV mode: python shap_computation_direct.py <scenario> <data_dir> <model_dir> <model_type> [scaling_mode]")
        sys.exit(1)

    if sys.argv[1].endswith(".pkl"):
        pickle_file = sys.argv[1]
        train_mode = sys.argv[2]
        model_type = sys.argv[3]
        scaling_mode = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        compute_shap_from_pickle(pickle_file, train_mode, model_type, scaling_mode)
    else:
        scenario = sys.argv[1]
        data_dir = sys.argv[2]
        model_dir = sys.argv[3]
        model_type = sys.argv[4]
        scaling_mode = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        compute_shap_from_csv(scenario, data_dir, model_dir, model_type, scaling_mode)
