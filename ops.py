import fairlearn
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer

from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
from fairlearn.preprocessing import CorrelationRemover

from cvxopt import matrix, solvers
from itertools import product

def normalize_weight(weight):
    """Normalizes an array so that the sum of all elements is 1.

    Args:
        array: The input array.

    Returns:
        The normalized array.
    """

    if not isinstance(weight, np.ndarray):
        weight = np.array(weight)

    # Ensure the array is not empty
    if weight.size == 0:
        raise ValueError("Input array cannot be empty.")

    # Calculate the sum of the array
    weight_sum = np.sum(weight)

    # Check if the sum is zero
    if weight_sum == 0:
        raise ValueError("Sum of the array cannot be zero.")

    # Divide each element by the sum to normalize
    normalized_weight = weight / weight_sum

    return normalized_weight

def split_train_test_df(X, y, selected_column, feat_mode):
    def get_train_test_masks(s_column, f_mode):
        if s_column=='NACCAGE':
            if f_mode==0:
                train_mask = X[s_column] < 65.
                test_mask  = ~train_mask
            elif f_mode==1:
                train_mask = (X[s_column] >= 65.) & (X[s_column] < 75.)
                test_mask  = ~train_mask
            elif f_mode==2:
                train_mask = (X[s_column] >= 75.) & (X[s_column] < 85.)
                test_mask  = ~train_mask
            else:
                test_mask  = X[s_column] >= 85.
                train_mask = ~test_mask
        elif s_column=='HISPANIC':
            train_mask = X[s_column] == f_mode[0]
            test_mask_1  = X[s_column] == f_mode[1]
            test_mask_2  = X[s_column] == f_mode[2]
        elif s_column=='RACE':
            for i, fm in enumerate(f_mode):
                if i==0 and fm!=3:
                    train_mask = X[s_column] == fm
                elif i==0 and fm==3:
                    train_mask = X[s_column].isin([1, 2])
                if i==1 and fm!=3:
                    test_mask_1  = X[s_column] == fm
                elif i==1 and fm==3:
                    test_mask_1 = X[s_column].isin([1, 2])
                if i==2 and fm!=3:
                    test_mask_2  = X[s_column] == fm
                elif i==2 and fm==3:
                    test_mask_2 = X[s_column].isin([1, 2])
        return train_mask, test_mask_1, test_mask_2
    drop_columns = selected_column+['RACE_HISP']
    if len(selected_column)==1:
        m_train, m_test = get_train_test_masks(selected_column[0], feat_mode)
    else:
        if feat_mode:
            m_train_1, m_test_11, m_test_12 = get_train_test_masks(selected_column[0], feat_mode[0])
            m_train_2, m_test_21, m_test_22 = get_train_test_masks(selected_column[1], feat_mode[1])
            m_train, m_test_1, m_test_2 = (m_train_1) & (m_train_2), (m_test_11) & (m_test_21), (m_test_12) & (m_test_22)            
    if feat_mode:
        hisp_race_train = X[m_train].loc[:,selected_column]
        hisp_race_test_1 = X[m_test_1].loc[:,selected_column]
        hisp_race_test_2 = X[m_test_2].loc[:,selected_column]

        X_train = X[m_train].drop(columns=drop_columns, axis=1)
        X_test_1 = X[m_test_1].drop(columns=drop_columns, axis=1)
        X_test_2 = X[m_test_2].drop(columns=drop_columns, axis=1)
        y_train, y_test_1, y_test_2 = y[m_train], y[m_test_1], y[m_test_2]

        X_train  = X_train.reset_index(drop=True)
        X_test_1 = X_test_1.reset_index(drop=True)
        X_test_2 = X_test_2.reset_index(drop=True)
        y_train  = y_train.reset_index(drop=True)
        y_test_1 = y_test_1.reset_index(drop=True)
        y_test_2 = y_test_2.reset_index(drop=True)
        hisp_race_train = hisp_race_train.reset_index(drop=True)
        hisp_race_test_1 = hisp_race_test_1.reset_index(drop=True)
        hisp_race_test_2 = hisp_race_test_2.reset_index(drop=True)
        
        return X_train, [X_test_1, X_test_2], y_train, [y_test_1, y_test_2], [hisp_race_train, hisp_race_test_1, hisp_race_test_2]
    else:
        mask_1 = X[selected_column[0]].isin([0, 1])
        mask_2 = X[selected_column[1]].isin([1, 2])
        mask = (mask_1) & (mask_2)
        hisp_race = X[mask].loc[:,selected_column].reset_index(drop=True)
        X_train = X[mask].drop(columns=drop_columns, axis=1).reset_index(drop=True)
        y_train = y[mask].reset_index(drop=True)
        return X_train, y_train, hisp_race

def safe_log1p(x):
    """Calculates log(1+x) safely, handling negative and large values."""
    # Handle negative x
    x = np.where(x < 0, np.finfo(x.dtype).tiny, x)  # Replace negative with tiny positive

    # Handle large x
    return np.where(x > 1e8, np.log(x) + np.log1p(1/x), np.log1p(x))

def scaling_data(X_train, X_test=None, scaling_mode=0, return_scaler=False, scaler=None):
    scalers = {
        0: lambda: None,  # Logarithmic transformation is handled separately
        1: StandardScaler,
        2: RobustScaler,
        3: lambda: PowerTransformer(method='yeo-johnson'),
        4: lambda: QuantileTransformer(output_distribution='normal')
    }
    
    if scaling_mode == 0:
        # Logarithmic transformation
        X_train_ = safe_log1p(X_train.values)
        X_test_ = [safe_log1p(xt.values) for xt in X_test] if X_test else None
        scaler = None
    else:
        # Fetch the appropriate scaler
        if scaler:
            X_train_ = scaler.transform(X_train)
        else:
            scaler = scalers[scaling_mode]()
            X_train_ = scaler.fit_transform(X_train)
        X_test_ = [scaler.transform(xt) for xt in X_test] if X_test else None
    
    if return_scaler:
        return (X_train_, X_test_, scaler) if X_test else (X_train_, scaler)
    else:
        return (X_train_, X_test_) if X_test else X_train_

def compute_sample_weight_kmm(Xs, Xt):
    def _fit_weights(Xs, Xt, epsilon):
        n_s = len(Xs)
        n_t = len(Xt)
        
        if epsilon is None:
            epsilon = (np.sqrt(n_s) - 1)/np.sqrt(n_s)
        
        kernel_params = {'gamma': 1.0}
        
        # Compute Kernel Matrix
        K = pairwise.pairwise_kernels(Xs, Xs, metric=kernel,
                                      **kernel_params)
        K = (1/2) * (K + K.transpose())
        
        # Compute q
        kappa = pairwise.pairwise_kernels(Xs, Xt,
                                          metric=kernel,
                                          **kernel_params)
        kappa = (n_s/n_t) * np.dot(kappa, np.ones((n_t, 1)))
        
        P = matrix(K)
        q = -matrix(kappa)

        # Define constraints
        G = np.ones((2*n_s+2, n_s))
        G[1] = -G[1]
        G[2:n_s+2] = np.eye(n_s)
        G[n_s+2:n_s*2+2] = -np.eye(n_s)
        h = np.ones(2*n_s+2)
        h[0] = n_s*(1+epsilon)
        h[1] = n_s*(epsilon-1)
        h[2:n_s+2] = B
        h[n_s+2:] = 0
        
        G = matrix(G)
        h = matrix(h)
        
        solvers.options["show_progress"] = bool(verbose)
        solvers.options["maxiters"] = max_iter
        if tol is not None:
            solvers.options['abstol'] = tol
            solvers.options['reltol'] = tol
            solvers.options['feastol'] = tol
        else:
            solvers.options['abstol'] = 1e-7
            solvers.options['reltol'] = 1e-6
            solvers.options['feastol'] = 1e-7
        
        weights = solvers.qp(P, q, G, h)['x']
        return np.array(weights).ravel()
        
    kernel   = "rbf"
    B        = 512
    epsilon  = None
    max_size = 1000
    tol      = None
    verbose  = 0
    max_iter = 100
    random_state = 777
    
    Xs = check_array(Xs)
    Xt = check_array(Xt)
    np.random.seed(random_state)
    
    if len(Xs) > max_size:
        size = len(Xs)
        power = 0
        while size > max_size:
            size = size / 2
            power += 1
        split = int(len(Xs) / 2**power)
        shuffled_index = np.random.choice(len(Xs), len(Xs), replace=False)
        weights = np.zeros(len(Xs))
        for i in range(2**power):
            index = shuffled_index[split*i:split*(i+1)]
            weights[index] = _fit_weights(Xs[index], Xt, epsilon)
    else:
        weights = _fit_weights(Xs, Xt, epsilon)
    
    return weights

def correlation_remover(X, y=None, sen_features=None, alpha=0.5):
    cr = CorrelationRemover(sensitive_feature_ids=sen_features, alpha=alpha)
    cr.fit(X, y)
    X_transform = cr.transform(X)
    return X_transform

def ParameterGrid(param_grid):
    """
    Generate all combinations of parameter settings for each model in param_grid.
    
    Parameters
    ----------
    param_grid : dict
        Dictionary with model names as keys and dictionaries of parameters as values.
        Each parameter dictionary contains parameter names as keys and lists of parameter settings to try as values.
    
    Yields
    ------
    params : dict
        Dictionary with model names as keys and dictionaries of parameter settings as values.
    """
    # Extract model names and their parameter grids
    model_names = list(param_grid.keys())
    model_param_grids = [param_grid[model] for model in model_names]
    
    # Generate all combinations of parameter settings for each model
    param_combinations = [list(product(*[[(k, v) for v in values] for k, values in model_params.items()])) for model_params in model_param_grids]
    
    # Generate all combinations of parameter settings across models
    for combination in product(*param_combinations):
        params = {}
        for model_name, model_params in zip(model_names, combination):
            params[model_name] = dict(model_params)
        yield params
    