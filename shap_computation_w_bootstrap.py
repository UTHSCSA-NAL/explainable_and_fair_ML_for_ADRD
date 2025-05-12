import sys
sys.path.append("/home/Codes/ad_classification")

import pickle
import numpy as np
import torch # type: ignore
import pandas as pd
import os
from sklearn.base import clone
from tqdm import tqdm # type: ignore

# Import your functions (or include them in this file)
import ops
import models as ML
from shap_utils import custom_tree_shap_batch_torch, get_xgboost_predictor

# You might need to set up your device, feature indices, etc.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUR_DIR = "/home/Codes/ad_classification/shap_values"

# --- Helper Functions ---
def load_data(pickle_file):
    """
    Load training data and other objects.
    Expecting the pickle file to have: train_data, nhw, nha, hwa, est.
    """
    with open(pickle_file, 'rb') as f:
        train_data, nhw, nha, hwa, est = pickle.load(f)
    return train_data, nhw, nha, hwa, est

def bootstrap_resample_residuals(y_true, y_pred):
    """
    Compute residuals and resample them with replacement.
    """
    residuals = y_true - y_pred
    bootstrap_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
    return bootstrap_residuals

def retrain_model(trained_model, X_train, y_new):
    print("Re-training model ...")
    y_new = np.rint(y_new).astype(int)
    unique_vals = np.unique(y_new)
    # If the unique values are [-1, 0, 1, 2] (instead of [0, 1, 2, 3]), shift by 1.
    if np.array_equal(unique_vals, np.array([-1, 0, 1, 2])):
        y_new = y_new + 1
    if np.array_equal(unique_vals, np.array([-1, 0, 1])):
        y_new = y_new + 1
    
    params = trained_model.predictor.get_params()
    params["device"] = "cpu"
    print("Re-loading parameters: ", params)
    # Create a new model instance with the same hyperparameters
    new_model = clone(trained_model).set_params(**params)
    if trained_model.sample_weight is not None:
        new_model.fit(X_train, y_new, sample_weight=trained_model.sample_weight)
    else:
        new_model.fit(X_train, y_new)
        
    return new_model

def compute_bootstrap_shap_for_fold(fold_idx, train_mode, scaling_mode, feature_columns,
                                    train_data, nhw, nha, hwa, est, batch_size=100):
    """
    For a single fold, perform:
      - Data extraction and scaling,
      - Residual resampling and bootstrap target generation,
      - Retraining the model,
      - SHAP value computation on X_test (combined data of nhw, nha, and hwa),
      - Refining of NaN/infinite values.
    """
    # Select training features, labels, and model based on train_mode.
    if train_mode == "all":
        X_train_raw = train_data[fold_idx]['data'].iloc[:, 1:]
        y_train = train_data[fold_idx]['y']['DX']
        y_pred = np.asarray(train_data[fold_idx]['y']['PRED_DX']>=0.5, int)
        model = est[fold_idx]
        # Test data from each non-location dataset:
        X_nhw_raw = nhw[fold_idx]['data'].iloc[:, 1:]
        X_nha_raw = nha[fold_idx]['data'].iloc[:, 1:]
        X_hwa_raw = hwa[fold_idx]['data'].iloc[:, 1:]
    elif train_mode == "single":
        X_train_raw = train_data[fold_idx]['data_1'].iloc[:, 1:]
        y_train = train_data[fold_idx]['y_1']['DX']
        y_pred = np.asarray(train_data[fold_idx]['y_1']['PRED_DX']>=0.5, int)
        model = est[fold_idx][0]
        # Test data still come from nhw, nha, and hwa.
        X_nhw_raw = nhw[fold_idx]['data'].iloc[:, 1:]
        X_nha_raw = nha[fold_idx]['data'].iloc[:, 1:]
        X_hwa_raw = hwa[fold_idx]['data'].iloc[:, 1:]
    else:
        raise ValueError("Invalid train_mode. Must be 'all' or 'single'.")
    
    # Scale X_train using the provided scaling mode.
    X_train = pd.DataFrame(ops.scaling_data(X_train_raw, None, scaling_mode), columns=feature_columns)
    # For X_test, scale each dataset and then concatenate.
    X_nhw = pd.DataFrame(ops.scaling_data(X_nhw_raw, None, scaling_mode), columns=feature_columns)
    X_nha = pd.DataFrame(ops.scaling_data(X_nha_raw, None, scaling_mode), columns=feature_columns)
    X_hwa = pd.DataFrame(ops.scaling_data(X_hwa_raw, None, scaling_mode), columns=feature_columns)
    X_test = pd.concat([X_nhw, X_nha, X_hwa], axis=0).values  # as numpy array for prediction
    
    # Compute residuals and generate a bootstrapped target.
    resampled_residuals = bootstrap_resample_residuals(y_train, y_pred)
    y_bootstrap = y_pred + resampled_residuals
    
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_pred, return_counts=True))
    print(np.unique(y_bootstrap, return_counts=True))
    
    # Re-train the model with the bootstrapped target.
    new_model = retrain_model(model, X_train.values, y_bootstrap)
    
    # Build the predictor function for the new model.
    predict_fn = get_xgboost_predictor(new_model, internal_batch_size=batch_size)
    
    # Compute SHAP values on X_test using X_train as the background.
    n_instances = X_test.shape[0]
    shap_vals = np.zeros((n_instances, len(feature_columns)))
    for start in range(0, n_instances, batch_size):
        end = min(start + batch_size, n_instances)
        batch_x = X_test[start:end]
        shap_vals[start:end] = custom_tree_shap_batch_torch(
            predict_fn, batch_x, X_train.values,
            filter_column_idx=list(range(len(feature_columns))),
            device=device
        )
    # Refine any NaN/infinite values.
    shap_vals = np.nan_to_num(shap_vals, nan=0, posinf=1e10, neginf=-1e10)
    return shap_vals

def compute_bootstrap_shap(iteration, pickle_file, train_mode, model_type, scaling_mode, batch_size=100):
    """
    For a given bootstrap iteration, loop over all 10 folds, compute SHAP values
    on the test data for each fold, and store a list (one entry per fold).
    """
    # Load data from pickle.
    train_data, nhw, nha, hwa, est = load_data(pickle_file)
    
    # Determine feature columns from the first fold.
    if train_mode == "all":
        feature_columns = list(train_data[0]['data'].iloc[:, 1:].columns)
    else:
        feature_columns = list(train_data[0]['data_1'].iloc[:, 1:].columns)
    
    # Compute bootstrap SHAP values for each fold.
    shap_values_all_folds = []
    for fold_idx in range(len(train_data)):
        print(f"Processing fold {fold_idx}")
        shap_vals_fold = compute_bootstrap_shap_for_fold(
            fold_idx, train_mode, scaling_mode, feature_columns,
            train_data, nhw, nha, hwa, est,
            batch_size=batch_size
        )
        shap_values_all_folds.append(shap_vals_fold)
    
    # Save the list of SHAP values (one per fold) to a pickle file.
    if not os.path.exists(f"{OUR_DIR}/{model_type}_{train_mode}"):
        os.makedirs(f"{OUR_DIR}/{model_type}_{train_mode}")
    output_filename = f"{OUR_DIR}/{model_type}_{train_mode}/bootstrap_shap_iter_{iteration}.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(shap_values_all_folds, f)
    print(f"Iteration {iteration} complete. SHAP values saved to {output_filename}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python bootstrap_iteration.py <iteration> <pickle_file> <train_mode> [scaling_mode]")
        sys.exit(1)
    
    iteration = int(sys.argv[1])
    pickle_file = sys.argv[2]
    train_mode = sys.argv[3]  # "all" or "single"
    model_type = sys.argv[4]
    scaling_mode = int(sys.argv[5]) if len(sys.argv) > 5 else 4  # default scaling_mode = 4
    
    compute_bootstrap_shap(iteration, pickle_file, train_mode, model_type, scaling_mode, batch_size=100)