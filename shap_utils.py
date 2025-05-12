import torch # type: ignore
import numpy as np
import xgboost as xgb

##############################################
# Predictor Builder Functions
##############################################
def get_xgboost_predictor(xgb_model, internal_batch_size=100):
    """
    Returns a predictor function for an XGBoost model.
    The model should be configured for GPU acceleration (e.g., tree_method='gpu_hist')
    and output raw margin predictions (via output_margin=True).
    """
    def predict(x):
        n_samples = x.shape[0]
        predictions = []
        for start in range(0, n_samples, internal_batch_size):
            end = min(start + internal_batch_size, n_samples)
            x_batch = x[start:end]
            dmat = xgb.DMatrix(x_batch)
            pred_batch = xgb_model.predict(dmat, output_margin=True)
            predictions.append(pred_batch)
        return np.concatenate(predictions)
    return predict

##############################################
# SHAP Computation (GPU-accelerated using torch)
##############################################
def custom_tree_shap_batch_torch(predict_fn, x_batch, background_data, filter_column_idx=None, 
                                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                 target_class=0):
    """
    Computes Tree-based SHAP values for an XGBoost classifier using PyTorch.
    
    Args:
        predict_fn: Predictor function.
        x_batch (np.ndarray): Batch input, shape (n_instances, n_features).
        background_data (np.ndarray): Background data, shape (n_samples, n_features).
        filter_column_idx (list): List of feature indices.
        device (torch.device): Device for computation.
        target_class (int): Index of the class to compute SHAP values for (if predictions are vectors).
    
    Returns:
        np.ndarray: SHAP values (n_instances, n_filter_features).
    """
    n_instances, n_features = x_batch.shape
    n_filter_features = len(filter_column_idx)
    background_data_t = torch.tensor(background_data, dtype=torch.float32, device=device)
    x_batch_t = torch.tensor(x_batch, dtype=torch.float32, device=device)
    
    # Compute baseline prediction (average over background)
    f_x_baseline_np = predict_fn(background_data)  # Expected shape (n_samples,) or (n_samples, num_outputs)
    f_x_baseline = torch.tensor(f_x_baseline_np, dtype=torch.float32, device=device)
    if f_x_baseline.dim() > 1:
        f_x_baseline = f_x_baseline.mean(dim=0)
    
    # Compute predictions for batch inputs.
    f_x_batch_np = predict_fn(x_batch)  # Expected shape (n_instances,) or (n_instances, num_outputs)
    f_x_batch = torch.tensor(f_x_batch_np, dtype=torch.float32, device=device)
    
    shap_values_t = torch.zeros(n_instances, n_filter_features, device=device)
    
    for i in range(n_instances):
        instance = x_batch_t[i]
        contributions = torch.zeros(n_filter_features, device=device)
        
        for j, feature_idx in enumerate(filter_column_idx):
            # Generate a perturbed instance by replacing the feature with its background mean.
            perturbed_instance = instance.clone()
            perturbed_instance[feature_idx] = background_data_t[:, feature_idx].mean()
            
            d_perturbed = perturbed_instance.unsqueeze(0).cpu().numpy()
            f_x_perturbed_np = predict_fn(d_perturbed)[0]
            f_x_perturbed = torch.tensor(f_x_perturbed_np, dtype=torch.float32, device=device)
            
            # Check if the predictions are scalars or vectors.
            # print(f_x_batch[i].dim())
            # print(f_x_perturbed.dim())
            if f_x_batch[i].dim() == 0:
                # Scalar output: no indexing required.
                # print("Scalar output")
                diff = f_x_batch[i] - f_x_perturbed
            else:
                # Multi-dimensional output: select the target_class.
                # print("Multi-dimensional output")
                diff = f_x_batch[i][target_class] - f_x_perturbed[target_class]
            contributions[j] = diff
        shap_values_t[i] = contributions
    
    return shap_values_t.cpu().numpy()

