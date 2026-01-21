import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer

from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
try:
    # Optional; imported lazily in functions below if not present
    from fairlearn.preprocessing import CorrelationRemover  # type: ignore
except Exception:
    CorrelationRemover = None  # will guard at call sites

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

def safe_log1p(x, eps: float = 1e-12):
    """
    Numerically stable log1p:
      - Avoids evaluating both branches (no spurious warnings from np.log(0)).
      - Clips input at (-1 + eps) so log1p never sees <= -1.
      - Uses log(x) + log1p(1/x) only for very large x to reduce cancellation.
    """
    x = np.asarray(x, dtype=np.float64)
    # Guard against exact -1 or smaller (e.g., from rounding negative values)
    x = np.where(x <= -1.0, -1.0 + eps, x)
    out = np.log1p(x)              # safe baseline for all x > -1
    mask = x > 1e8                 # "very large" region
    if np.any(mask):
        xm = x[mask]
        out[mask] = np.log(xm) + np.log1p(1.0 / xm)
    return out

def _apply_same_scaling(X_raw, scaler, scale_mode, icv=None):
    if scale_mode == 0:
        # mirror ops.scaling_data behavior for mode 0
        return safe_log1p(np.asarray(X_raw))
    if scale_mode == 5:
        if icv is None:
            raise ValueError("ICV values required for scaling_mode=5")
        icv_arr = np.asarray(icv, dtype=np.float32).reshape(-1, 1)
        return np.asarray(X_raw, dtype=np.float32) / np.clip(icv_arr, 1e-6, None)
    if scaler is None:
        return np.asarray(X_raw)
    return scaler.transform(np.asarray(X_raw))

def scaling_data(X_train, X_test=None, scaling_mode=0, return_scaler=False, scaler=None,
                 icv_train=None, icv_test=None):
    scalers = {
        0: lambda: None,  # Logarithmic transformation is handled separately
        1: StandardScaler,
        2: RobustScaler,
        3: lambda: PowerTransformer(method='yeo-johnson'),
        4: lambda: QuantileTransformer(output_distribution='normal')
    }

    if scaling_mode == 0:
        # Logarithmic transformation
        X_train_np = np.asarray(X_train)
        X_train_ = safe_log1p(X_train_np)
        if X_test is None:
            X_test_ = None
        elif isinstance(X_test, (list, tuple)):
            # supports multiple test matrices: [X_test1, X_test2, ...]
            X_test_ = [safe_log1p(np.asarray(xt)) for xt in X_test]
        else:
            # single ndarray
            X_test_ = safe_log1p(np.asarray(X_test))
        scaler = None
    elif scaling_mode == 5:
        if icv_train is None:
            raise ValueError("icv_train required for scaling_mode=5")
        icv_arr = np.asarray(icv_train, dtype=np.float32).reshape(-1, 1)
        X_train_ = np.asarray(X_train, dtype=np.float32) / np.clip(icv_arr, 1e-6, None)
        if X_test is None:
            X_test_ = None
        elif isinstance(X_test, (list, tuple)):
            if not isinstance(icv_test, (list, tuple)):
                raise ValueError("icv_test list required for scaling_mode=5")
            X_test_ = []
            for xt, icv_te in zip(X_test, icv_test):
                icv_te_arr = np.asarray(icv_te, dtype=np.float32).reshape(-1, 1)
                X_test_.append(np.asarray(xt, dtype=np.float32) / np.clip(icv_te_arr, 1e-6, None))
        else:
            if icv_test is None:
                raise ValueError("icv_test required for scaling_mode=5")
            icv_te_arr = np.asarray(icv_test, dtype=np.float32).reshape(-1, 1)
            X_test_ = np.asarray(X_test, dtype=np.float32) / np.clip(icv_te_arr, 1e-6, None)
        scaler = None
    else:
        # Fetch the appropriate scaler
        X_train_np = np.asarray(X_train)
        if scaler:
            X_train_ = scaler.transform(X_train_np)
        else:
            scaler = scalers[scaling_mode]()
            X_train_ = scaler.fit_transform(X_train_np)
        if X_test is None:
            X_test_ = None
        elif isinstance(X_test, (list, tuple)):
            # supports multiple test matrices: [X_test1, X_test2, ...]
            X_test_ = [scaler.transform(np.asarray(xt)) for xt in X_test]
        else:
            # single ndarray
            X_test_ = scaler.transform(np.asarray(X_test))
    
    if return_scaler:
        return (X_train_, X_test_, scaler) if X_test is not None else (X_train_, scaler)
    return (X_train_, X_test_) if X_test is not None else X_train_

# def compute_sample_weight_kmm(Xs, Xt):
#     def _fit_weights(Xs, Xt, epsilon):
#         n_s = len(Xs)
#         n_t = len(Xt)
        
#         if epsilon is None:
#             epsilon = (np.sqrt(n_s) - 1)/np.sqrt(n_s)
        
#         kernel_params = {'gamma': 1.0}
        
#         # Compute Kernel Matrix
#         K = pairwise.pairwise_kernels(Xs, Xs, metric=kernel,
#                                       **kernel_params)
#         K = (1/2) * (K + K.transpose())
        
#         # Compute q
#         kappa = pairwise.pairwise_kernels(Xs, Xt,
#                                           metric=kernel,
#                                           **kernel_params)
#         kappa = (n_s/n_t) * np.dot(kappa, np.ones((n_t, 1)))
        
#         K = np.ascontiguousarray(K, dtype=np.float64)
#         kappa = np.ascontiguousarray(kappa, dtype=np.float64)
        
#         P = matrix(K)
#         q = -matrix(kappa)

#         # Define constraints
#         G = np.ones((2*n_s+2, n_s))
#         G[1] = -G[1]
#         G[2:n_s+2] = np.eye(n_s)
#         G[n_s+2:n_s*2+2] = -np.eye(n_s)
#         h = np.ones(2*n_s+2)
#         h[0] = n_s*(1+epsilon)
#         h[1] = n_s*(epsilon-1)
#         h[2:n_s+2] = B
#         h[n_s+2:] = 0
        
#         G = matrix(G)
#         h = matrix(h)
        
#         solvers.options["show_progress"] = bool(verbose)
#         solvers.options["maxiters"] = max_iter
#         if tol is not None:
#             solvers.options['abstol'] = tol
#             solvers.options['reltol'] = tol
#             solvers.options['feastol'] = tol
#         else:
#             solvers.options['abstol'] = 1e-7
#             solvers.options['reltol'] = 1e-6
#             solvers.options['feastol'] = 1e-7
        
#         weights = solvers.qp(P, q, G, h)['x']
#         return np.array(weights).ravel()
        
#     kernel   = "rbf"
#     B        = 512
#     epsilon  = None
#     max_size = 1000
#     tol      = None
#     verbose  = 0
#     max_iter = 100
#     random_state = 777
    
#     Xs = np.ascontiguousarray(check_array(Xs), dtype=np.float64)
#     Xt = np.ascontiguousarray(check_array(Xt), dtype=np.float64)
#     np.random.seed(random_state)
    
#     if len(Xs) > max_size:
#         size = len(Xs)
#         power = 0
#         while size > max_size:
#             size = size / 2
#             power += 1
#         split = int(len(Xs) / 2**power)
#         shuffled_index = np.random.choice(len(Xs), len(Xs), replace=False)
#         weights = np.zeros(len(Xs))
#         for i in range(2**power):
#             index = shuffled_index[split*i:split*(i+1)]
#             weights[index] = _fit_weights(Xs[index], Xt, epsilon)
#     else:
#         weights = _fit_weights(Xs, Xt, epsilon)
    
#     return weights

def compute_sample_weight_kmm(
    Xs,
    Xt,
    *,
    kernel: str = "rbf",
    B: float = 512.0,
    eps: float | None = None,
    gamma: float | None = 1.0,
    max_size: int = 1000,
    tol: float | None = None,
    verbose: int = 0,
    max_iter: int = 100,
    random_state: int = 777,
):
    """
    Kernel Mean Matching (KMM) weight solver.

    Parameters
    ----------
    Xs : array-like, shape (n_source, d)
        Source features (will be checked and cast to float64, C-contiguous).
    Xt : array-like, shape (n_target, d)
        Target features (will be checked and cast to float64, C-contiguous).
    kernel : {"rbf", "linear", ...}, default "rbf"
        Pairwise kernel name (sklearn.metrics.pairwise_kernels).
    B : float, default 512.0
        Upper bound on individual weights (beta_i <= B).
    eps : float or None, default None
        Constraint slack; if None, uses (sqrt(n_s) - 1)/sqrt(n_s).
    gamma : float or None, default 1.0
        Kernel parameter for RBF. Ignored for kernels that don’t use it.
    max_size : int, default 1000
        If n_source > max_size, chunk the problem for memory/CPU stability.
    tol : float or None
        cvxopt tolerance; if None, sensible defaults are used.
    verbose : int, default 0
        cvxopt verbosity flag (0/1).
    max_iter : int, default 100
        Max iterations for cvxopt.
    random_state : int, default 777
        RNG seed used when chunking.

    Returns
    -------
    weights : ndarray, shape (n_source,)
        KMM importance weights.
    """
    def _fit_weights(Xs_block, Xt_block, eps_block):
        n_s = len(Xs_block)
        n_t = len(Xt_block)

        # default epsilon
        if eps_block is None:
            eps_block = (np.sqrt(n_s) - 1.0) / np.sqrt(n_s)

        # kernel params
        kernel_params = {}
        if kernel == "rbf":
            kernel_params["gamma"] = 1.0 if gamma is None else float(gamma)

        # Compute kernel matrix K (source-source), symmetrize
        K = pairwise.pairwise_kernels(Xs_block, Xs_block, metric=kernel, **kernel_params)
        K = 0.5 * (K + K.T)

        # Compute kappa = (n_s / n_t) * sum_j k(x_i, z_j)
        kappa = pairwise.pairwise_kernels(Xs_block, Xt_block, metric=kernel, **kernel_params)
        kappa = (n_s / n_t) * np.dot(kappa, np.ones((n_t, 1)))

        # Cast to cvxopt types
        K = np.ascontiguousarray(K, dtype=np.float64)
        kappa = np.ascontiguousarray(kappa, dtype=np.float64)

        P = matrix(K)
        q = -matrix(kappa)

        # Constraints:
        #   sum_i beta_i <= n_s*(1+eps)
        #  -sum_i beta_i <= n_s*(eps-1)
        #         0 <= beta_i <= B
        G = np.ones((2 * n_s + 2, n_s), dtype=np.float64)
        G[1] = -G[1]
        G[2:n_s + 2] = np.eye(n_s, dtype=np.float64)
        G[n_s + 2:2 * n_s + 2] = -np.eye(n_s, dtype=np.float64)

        h = np.ones(2 * n_s + 2, dtype=np.float64)
        h[0] = n_s * (1.0 + eps_block)
        h[1] = n_s * (eps_block - 1.0)
        h[2:n_s + 2] = float(B)
        h[n_s + 2:] = 0.0

        G = matrix(G)
        h = matrix(h)

        # cvxopt options
        solvers.options["show_progress"] = bool(verbose)
        solvers.options["maxiters"] = int(max_iter)
        if tol is not None:
            solvers.options["abstol"] = float(tol)
            solvers.options["reltol"] = float(tol)
            solvers.options["feastol"] = float(tol)
        else:
            solvers.options["abstol"] = 1e-7
            solvers.options["reltol"] = 1e-6
            solvers.options["feastol"] = 1e-7

        sol = solvers.qp(P, q, G, h)
        return np.array(sol["x"]).ravel()

    # sanitize inputs
    Xs = np.ascontiguousarray(check_array(Xs), dtype=np.float64)
    Xt = np.ascontiguousarray(check_array(Xt), dtype=np.float64)
    np.random.seed(int(random_state))

    n_s = len(Xs)
    if n_s > int(max_size):
        # chunk Xs deterministically for stability, solve per chunk, and place back
        size = n_s
        power = 0
        while size > int(max_size):
            size /= 2.0
            power += 1
        split = int(n_s / (2 ** power))

        shuffled_index = np.random.choice(n_s, n_s, replace=False)
        weights = np.zeros(n_s, dtype=np.float64)
        for i in range(2 ** power):
            idx = shuffled_index[split * i: split * (i + 1)]
            weights[idx] = _fit_weights(Xs[idx], Xt, eps)
    else:
        weights = _fit_weights(Xs, Xt, eps)

    return weights

def correlation_remover(X, y=None, sen_features=None, alpha=0.5):
    if CorrelationRemover is None:
        raise ImportError("fairlearn is required for correlation_remover; please pip install fairlearn")
    cr = CorrelationRemover(sensitive_feature_ids=sen_features, alpha=alpha)
    cr.fit(X, y)
    X_transform = cr.transform(X)
    return X_transform

def correlation_remover_v2(X_train: np.ndarray,
                                  X_test:  np.ndarray,
                                  s_train: np.ndarray,
                                  s_test:  np.ndarray,
                                  alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Fairlearn CorrelationRemover when the sensitive attribute is provided
    as a vector (not a column in X). Handles both old and new Fairlearn APIs.

    Returns transformed X_train, X_test with the same number of columns as the
    original feature matrix (the temporary sensitive column is removed).
    """
    Xtr = np.asarray(X_train, dtype=np.float32, order="C")
    Xte = np.asarray(X_test,  dtype=np.float32, order="C")
    s_tr = np.asarray(s_train).reshape(-1, 1)
    s_te = np.asarray(s_test).reshape(-1, 1)

    # Build augmented matrices with sensitive column at the end
    Xtr_aug = np.concatenate([Xtr, s_tr.astype(np.float32)], axis=1)
    Xte_aug = np.concatenate([Xte, s_te.astype(np.float32)], axis=1)
    s_col_idx = Xtr_aug.shape[1] - 1  # last column

    # Try the *new* API first: fit/transform with `sensitive_features=...`
    try:
        if CorrelationRemover is None:
            raise TypeError("fairlearn not available")
        # Use DataFrames so we can pass sensitive column cleanly
        cols = [f"f{i}" for i in range(Xtr.shape[1])]
        Xtr_df = pd.DataFrame(Xtr, columns=cols)
        Xte_df = pd.DataFrame(Xte, columns=cols)
        s_tr_sr = pd.Series(s_train, name="__S__")
        s_te_sr = pd.Series(s_test,  name="__S__")

        cr = CorrelationRemover(alpha=float(alpha))
        Xtr_t = cr.fit_transform(Xtr_df, sensitive_features=s_tr_sr)
        Xte_t = cr.transform(Xte_df, sensitive_features=s_te_sr)

        return Xtr_t.to_numpy(dtype=np.float32), Xte_t.to_numpy(dtype=np.float32)

    except TypeError:
        # Old API path: must specify sensitive_feature_ids at init; no `sensitive_features` kwarg.
        if CorrelationRemover is None:
            raise ImportError("fairlearn is required for correlation_remover_v2; please pip install fairlearn")
        cr = CorrelationRemover(sensitive_feature_ids=[s_col_idx], alpha=float(alpha))
        Xtr_t = cr.fit_transform(Xtr_aug)  # returns array with sensitive column removed
        Xte_t = cr.transform(Xte_aug)
        # Shapes should match original feature count
        return Xtr_t.astype(np.float32, copy=False), Xte_t.astype(np.float32, copy=False)


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
    
