import sys
import json
import traceback


import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import parallel_backend
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import check_scoring
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score
from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from typing import Optional, Dict, List, Tuple, Any
from cvxopt import matrix, solvers

# Reuse shared preprocessing / KMM routines
from dementia_ai.core import ops
#import custom_objectives as CO
from . import objectives as CO

class _IdentityScaler:
    def transform(self, X):
        import numpy as np
        return np.asarray(X, dtype=np.float32)

def _ba_optimal_threshold(y_true, p1, grid=None) -> float:
    """Return the threshold that maximizes Balanced Accuracy on y_true."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    if grid is None:
        # dense grid around the middle; robust for imbalanced sets
        grid = np.unique(np.clip(np.r_[np.linspace(0.05,0.95,181), p1], 0, 1))
    best, thr = -1.0, 0.5
    for t in grid:
        yhat = (p1 >= t).astype(int)
        ba = balanced_accuracy_score(y_true, yhat)
        if ba > best:
            best, thr = ba, t
    return float(thr)

def _split_xgb_custom_objs(paras: dict):
    """Split custom-objective keys from plain XGB params."""
    p = dict(paras or {})
    focal = {
        "use":   bool(p.pop("use_focal_obj", False)),
        "alpha": float(p.pop("focal_alpha", 0.90)),
        "gamma": float(p.pop("focal_gamma", 2.0)),
    }
    cs = {
        "use":     bool(p.pop("use_cost_sensitive_obj", False)),
        "cost_fn": float(p.pop("cs_cost_fn", 5.0)),
        "cost_fp": float(p.pop("cs_cost_fp", 1.0)),
    }
    return p, focal, cs

# def ba_cost_scorer(est, X, y):
#     """Scorer used in grid search. Uses t_cost from est params if available."""
#     from sklearn.metrics import balanced_accuracy_score
#     proba = est.predict_proba(X)[:, 1]
#     # Parameters can be in est.paras (CustomML) or in get_params()
#     cs_fn = None
#     cs_fp = None
#     if hasattr(est, "paras"):
#         cs_fn = est.paras.get("cs_cost_fn", None)
#         cs_fp = est.paras.get("cs_cost_fp", None)
#     if (cs_fn is None or cs_fp is None) and hasattr(est, "get_params"):
#         p = est.get_params()
#         cs_fn = p.get("cs_cost_fn", cs_fn)
#         cs_fp = p.get("cs_cost_fp", cs_fp)
#     if cs_fn is not None and cs_fp is not None and (cs_fn + cs_fp) > 0:
#         t_cost = cs_fp / (cs_fp + cs_fn)
#     else:
#         t_cost = 0.5
#     yhat = (proba >= t_cost).astype(int)
#     return balanced_accuracy_score(y, yhat)

class BACostScorer:
    def __init__(self, default_threshold: float = 0.5):
        self._sign = 1  # sklearn-compatible
        self.default_threshold = float(default_threshold)

    def _threshold_from_params(self, est):
        # Look in both get_params() and .paras, if present
        cf = cp = None
        if hasattr(est, "get_params"):
            try:
                p = est.get_params()
                cf = p.get("cs_cost_fn", None)
                cp = p.get("cs_cost_fp", None)
            except Exception:
                pass
        if (cf is None or cp is None) and hasattr(est, "paras"):
            cf = est.paras.get("cs_cost_fn", cf)
            cp = est.paras.get("cs_cost_fp", cp)
        if cf is not None and cp is not None and (cf + cp) > 0:
            return float(cp) / float(cp + cf)
        return self.default_threshold

    def __call__(self, estimator, X, y):
        thr = self._threshold_from_params(estimator)

        # 1) Fast path for our custom XGB class that exposes .booster_ & .scaler_
        booster = getattr(estimator, "booster_", None)
        scaler  = getattr(estimator, "scaler_",  None)
        if booster is not None and scaler is not None:
            try:
                Xs = scaler.transform(np.asarray(X, dtype=np.float32))
                proba = booster.predict(xgb.DMatrix(Xs))
                yhat = (proba >= thr).astype(int)
                return balanced_accuracy_score(y, yhat)
            except Exception:
                pass  # fall through

        # 2) Fallback to predict_proba if available
        if hasattr(estimator, "predict_proba"):
            try:
                proba = estimator.predict_proba(X)[:, 1]
                yhat = (proba >= thr).astype(int)
                return balanced_accuracy_score(y, yhat)
            except Exception:
                pass

        # 3) Fallback to predict if available (already hard labels)
        if hasattr(estimator, "predict"):
            try:
                yhat = estimator.predict(X)
                # If predict returns floats, threshold them
                if yhat.dtype != np.int_ and yhat.dtype != np.bool_:
                    yhat = (np.asarray(yhat, dtype=float) >= thr).astype(int)
                return balanced_accuracy_score(y, yhat)
            except Exception:
                pass

        # 4) Last resort: return the worst possible score for this split
        #     so this candidate is ignored, but we don't crash the whole search
        return -np.inf

ba_cost_scorer = BACostScorer()

def custom_f1_score(y_true, y_pred):
    # Example: You can modify this function to compute any custom metric
    return f1_score(y_true, y_pred, average='weighted')

def custom_ba_score(y_true, y_pred):
    # Example: You can modify this function to compute any custom metric
    return balanced_accuracy_score(y_true, y_pred)

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
#     B        = 1000
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

class CustomML(BaseEstimator):
    def __init__(self, model_name, paras=None, sample_weight=None, **kwargs):
        self.model_name = model_name
        self.paras = paras if paras is not None else {}
        #print("Init:", self.paras)
        self.sample_weight = sample_weight
        self.kwargs = kwargs
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        import inspect
        def _filter_params(params, cls):
            if params is None:
                return {}
            valid = set(inspect.signature(cls.__init__).parameters.keys())
            # always ignore our internal dummy knob if present
            params = dict(params)
            params.pop("__dummy__", None)
            return {k: v for k, v in params.items() if k in valid}
        if self.model_name == 'svm':
            p = _filter_params(self.paras, SVC)
            k = _filter_params(self.kwargs, SVC)
            # guarantee probabilities unless explicitly overridden
            if 'probability' not in p and 'probability' not in k:
                p['probability'] = True
            self.predictor = SVC(**p, **k)
        elif self.model_name == 'xgboost_vanilla':
            # Pure, dependable XGB — no custom objective, no fancy knobs
            p = dict(self.paras or {})
            # normalize param aliases
            if 'lambda' in p and 'reg_lambda' not in p: p['reg_lambda'] = p.pop('lambda')
            if 'alpha'  in p and 'reg_alpha'  not in p: p['reg_alpha']  = p.pop('alpha')
            if 'eta' not in p and 'learning_rate' in p: p['eta'] = p.pop('learning_rate')

            # safe defaults if not provided
            p.setdefault('objective', 'binary:logistic')
            p.setdefault('eval_metric', 'aucpr')
            p.setdefault('tree_method', 'hist')
            p.setdefault('device', self.kwargs.get('device', 'cuda'))
            p.setdefault('max_depth', 4)
            p.setdefault('eta', 0.1)
            p.setdefault('subsample', 0.9)
            p.setdefault('colsample_bytree', 0.9)
            p.setdefault('reg_lambda', 1.0)
            p.setdefault('reg_alpha', 0.0)
            p.setdefault('n_estimators', self.kwargs.get('total_steps', 300))
            p.setdefault('verbosity', 0)
            p.setdefault('nthread', 8)

            self.predictor = xgb.XGBClassifier(**p)

            # vanilla keeps a fold-tuned BA threshold (optional; default 0.5)
            self.best_threshold_ = 0.5

            # wrap .fit to optionally accept (X_val, y_val) for early-stopping & BA threshold
            self._xgb_orig_fit = self.predictor.fit
            self._fit_with_es = self._xgb_vanilla_fit_with_es

            # helper: class predictions at tuned threshold
            self._xgb_predict_proba_fn = self._xgb_vanilla_predict_proba
        elif self.model_name == 'random_forest':
            self.predictor = RandomForestClassifier(
                **_filter_params(self.paras, RandomForestClassifier),
                **_filter_params(self.kwargs, RandomForestClassifier)
            )
        elif self.model_name == 'mlp':
            self.predictor = MLPClassifier(
                **_filter_params(self.paras, MLPClassifier),
                **_filter_params(self.kwargs, MLPClassifier)
            )
        
    def set_params(self, **params):
        p = dict(params or {})
        p.pop("__dummy__", None)
        # Preserve/override sample_weight explicitly
        if "sample_weight" in p:
            self.sample_weight = p.pop("sample_weight")
        # keep existing sample_weight if not provided
        # Normalize some xgboost aliases so grid params actually change the model
        if self.model_name == "xgboost_vanilla":
            if "lambda" in p and "reg_lambda" not in p:
                p["reg_lambda"] = p.pop("lambda")
            if "alpha" in p and "reg_alpha" not in p:
                p["reg_alpha"] = p.pop("alpha")
            if "learning_rate" in p and "eta" not in p:
                p["eta"] = p.pop("learning_rate")
        self.paras = p
        self._initialize_predictor()
        return self
    
    def fit(self, X, y, sample_weight=None):
        # ---------------- models (svm / rf / mlp / xgboost_vanilla) ----------------
        if sample_weight is not None:
            self.predictor.fit(X, y, sample_weight=sample_weight)
        else:
            self.predictor.fit(X, y)
        return self

    def _xgb_vanilla_fit_with_es(self, X, y, X_val=None, y_val=None, early_stopping_rounds=50, sample_weight=None):
        _orig_fit = getattr(self, "_xgb_orig_fit", None)
        if _orig_fit is None:
            raise AttributeError("_xgb_orig_fit not set; ensure xgboost_vanilla predictor initialized")

        fit_kwargs = {}
        if (X_val is not None) and (y_val is not None) and (early_stopping_rounds is not None):
            try:
                self.predictor.set_params(early_stopping_rounds=int(early_stopping_rounds))
            except Exception:
                pass
            fit_kwargs['eval_set'] = [(X, y), (X_val, y_val)]

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        _orig_fit(X, y, **fit_kwargs)

        if X_val is not None and y_val is not None:
            p1_val = self.predictor.predict_proba(X_val)[:, 1]
            self.best_threshold_ = _ba_optimal_threshold(y_val, p1_val)
        return self

    def _xgb_vanilla_predict_proba(self, X):
        if hasattr(self.predictor, "predict_proba"):
            return self.predictor.predict_proba(X)
        import numpy as np
        if hasattr(self.predictor, "decision_function"):
            s = self.predictor.decision_function(X)
            s = np.asarray(s)
            if s.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-s))
                return np.vstack([1.0 - p1, p1]).T
            else:
                e = np.exp(s - s.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                return p
        raise AttributeError("No predict_proba/decision_function available.")

    
    def predict(self, X):        
        if self.model_name == 'xgboost_vanilla' and hasattr(self, "best_threshold_"):
            p1 = self._xgb_vanilla_predict_proba(X)[:, 1]
            thr = getattr(self, "best_threshold_", 0.5)
            return (p1 >= thr).astype(int)
        return self.predictor.predict(X)
    
    def predict_proba(self, X):
        if self.model_name == 'xgboost_vanilla':
            return self._xgb_vanilla_predict_proba(X)
        # Native path if available
        if hasattr(self.predictor, "predict_proba"):
            return self.predictor.predict_proba(X)

        # Fallback: convert decision scores to [p0, p1] via logistic
        import numpy as np
        if hasattr(self.predictor, "decision_function"):
            s = self.predictor.decision_function(X)
            s = np.asarray(s)
            if s.ndim == 1:
                # binary: map to 2-column probs
                p1 = 1.0 / (1.0 + np.exp(-s))
                return np.vstack([1.0 - p1, p1]).T
            else:
                # multiclass one-vs-rest scores -> softmax
                e = np.exp(s - s.max(axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                return p

        # Last resort: hard predictions -> degenerate probabilities
        y = self.predictor.predict(X)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size == 2:
            p1 = (y == classes.max()).astype(float)
            return np.vstack([1.0 - p1, p1]).T
        else:
            # multiclass: one-hot
            K = classes.size
            idx = {c: i for i, c in enumerate(classes)}
            probs = np.zeros((y.shape[0], K), dtype=float)
            for i, yy in enumerate(y):
                probs[i, idx[yy]] = 1.0
            return probs
    
    def compute_shap_values(self, X_train, X_test, K=None):
        try:
            import shap
        except Exception:
            print("[SHAP] shap not installed; skipping SHAP explanation.", flush=True)
            return None
        
        def model_predict(X):
            return self.predict_proba(X)[:, 1]
        
        if K:
            X_train_summary = shap.sample(X_train, K)
        else:
            X_train_summary = X_train

        explainer = shap.KernelExplainer(model_predict, X_train_summary)
        shap_values = explainer.shap_values(X_test)
        
        return explainer, shap_values
    
    @staticmethod
    def hyperparameter_optimization(X_train, y_train, model_name, X_test=None, y_test=None, cv=None,
                                    init_paras=None, sample_weight=None, param_grid=None, n_jobs=-1):
        
        custom_scorer = make_scorer(custom_ba_score, greater_is_better=True)
        
        opt = MyGridSearchCV(estimator=CustomML(model_name, paras=init_paras, sample_weight=sample_weight),
                             param_grid=param_grid,
                             scoring=custom_scorer,
                             n_jobs=n_jobs,
                             cv=cv,
                             verbose=0,
                             refit=True)
        #opt.set_separated_test(X_test, y_test)
        with parallel_backend('loky', n_jobs=n_jobs):
            opt.fit(X_train, y_train)
        #print("Best Hyperparameters: ", opt.best_params_)
        
        return opt

def fit_and_score(estimator, X, y, train, test, parameters, scorer, X_test=None, y_test=None):
    #est = clone(estimator).set_params(**parameters)
    est = clone(estimator)
    # preserve sample_weight attribute across clone if present
    if hasattr(estimator, "sample_weight"):
        try:
            est.sample_weight = getattr(estimator, "sample_weight")
        except Exception:
            pass
    if parameters:
        # Pass all grid parameters through; CustomML.set_params will normalize/ignore as needed.
        try:
            est.set_params(**parameters)
        except Exception:
            # Fallback to legal subset if something goes wrong
            legal = est.get_params(deep=True).keys()
            clean = {k: v for k, v in parameters.items() if k in legal}
            if clean:
                est.set_params(**clean)

    # slice
    X_tr = X.iloc[train] if hasattr(X, "iloc") else X[train]
    y_tr = y.iloc[train] if hasattr(y, "iloc") else y[train]

    # pass group ids / sample_weight when present (kept from your version)
    if hasattr(est, "group_ids_source") and est.group_ids_source is not None:
        est.group_ids_source = np.asarray(est.group_ids_source)[train]
    sw = getattr(est, "sample_weight", None)
    if sw is not None:
        sw_tr = sw[train]
        est.fit(X_tr, y_tr, sample_weight=sw_tr)
    else:
        est.fit(X_tr, y_tr)

    # --- robust scoring ---
    try:
        if X_test is not None and y_test is not None:
            score_list = [scorer(est, xt, yt) for xt, yt in zip(X_test, y_test)]
            score = sum(score_list) / len(score_list)
        else:
            X_te = X.iloc[test] if hasattr(X, "iloc") else X[test]
            y_te = y.iloc[test] if hasattr(y, "iloc") else y[test]
            score = scorer(est, X_te, y_te)
    except Exception:
        # If anything blows up, give this parameter set a terrible score
        score = -np.inf

    return parameters, score, est
    
class MyGridSearchCV(GridSearchCV):
    def __init__(self, *args, **kwargs):
        super(MyGridSearchCV, self).__init__(*args, **kwargs)
        self.all_best_parameters_ = {}
        self.split_indices = {
            'train': [],
            'test': []
        }
        self.test_score_fold  = []
        self.best_test_scores = []
        self.X_test = None
        self.y_test = None

    @staticmethod
    def _param_key(params: dict):
        return tuple(sorted(params.items()))
    
    def set_separated_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        
    def scorer_single(self, estimator, X, y_true):
        y_pred = estimator.predict(X)
        return balanced_accuracy_score(y_true, y_pred)
        
    def fit(self, X, y=None):
        self.scorer_ = self.scoring
        # Wrap integer cv into StratifiedKFold
        cv = self.cv if not isinstance(self.cv, int) else StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=42
        )

        if self.verbose > 0:
            print(f"Fitting {len(cv)} folds for each of {len(self.param_grid)} candidates, "
                  f"totalling {len(cv) * len(self.param_grid)} fits")

        # materialize the full list of parameter settings
        if not self.param_grid:
            self.param_iter = [dict()]
        else:
            self.param_iter = list(ParameterGrid(self.param_grid))

        # for sensitivity: collect scores per param set
        score_map = {self._param_key(p): [] for p in self.param_iter}
        param_map = {self._param_key(p): p for p in self.param_iter}

        for split_idx, (train, test) in enumerate(cv.split(X, y)):
            n_tr = len(train)
            n_va = len(test)
            print(f"        =========== Training Split {split_idx} - train {n_tr} samples - val {n_va} samples ===========        ")
            # reset per-fold best trackers
            self.best_score_fold = -np.inf if self.scorer_._sign > 0 else np.inf
            self.best_params_fold = None

            # grid search with progress bar
            with tqdm(total=len(self.param_iter),
                      desc="Grid Search Progress",
                      bar_format='{desc:<10}{percentage:3.0f}%|{bar:30}{r_bar}') as pbar:

                # sequential evaluation of each parameter setting
                for parameters in self.param_iter:
                    parameters, score, est = fit_and_score(
                        self.estimator, X, y, train, test,
                        parameters, self.scorer_, self.X_test, self.y_test
                    )

                    # update global best if refit=True
                    if self.refit and score > getattr(self, 'best_score_', -np.inf):
                        self.best_score_     = score
                        self.best_params_    = parameters
                        self.best_estimator_ = est

                    # update this fold’s best
                    if score > self.best_score_fold:
                        self.best_score_fold  = score
                        self.best_params_fold = parameters

                    # record fold‐test score & advance bar
                    score_map[self._param_key(parameters)].append(score)
                    self.test_score_fold.append(score)
                    try:
                        pbar.set_postfix_str(f"val_score:{score:.4f}")
                    except Exception:
                        pass
                    pbar.update(1)

            # after finishing this fold
            self.best_test_scores.append(self.best_score_fold)
            self.all_best_parameters_[f'split{split_idx}_params'] = self.best_params_fold
            self.split_indices['train'].append(train)
            self.split_indices['test'].append(test)

        # Build cv_results_ summary (mean/std per param set)
        param_keys = list(score_map.keys())
        param_names = sorted({k for p in param_map.values() for k in p.keys()})
        self.cv_results_ = {
            "mean_test_score": [np.mean(score_map[k]) if score_map[k] else -np.inf for k in param_keys],
            "std_test_score":  [np.std(score_map[k], ddof=1) if len(score_map[k]) > 1 else 0.0 for k in param_keys],
        }
        for name in param_names:
            self.cv_results_[f"param_{name}"] = [param_map[k].get(name) for k in param_keys]

        return self
    
    def predict(self, X):
        if hasattr(self.best_estimator_, "predict"):
            self._check_is_fitted('predict')
            return self.best_estimator_.predict(X)
        else:
            raise AttributeError("The best estimator does not have a 'predict' method.")

    def predict_proba(self, X):
        if hasattr(self.best_estimator_, "predict_proba"):
            self._check_is_fitted('predict_proba')
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("The best estimator does not have a 'predict_proba' method.")

    def score(self, X, y=None):
        if hasattr(self.best_estimator_, "score"):
            self._check_is_fitted('score')
            return self.best_estimator_.score(X, y)
        else:
            raise AttributeError("The best estimator does not have a 'score' method.")



class XGBCustomObjectiveEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 # Custom objective selection
                 objective_version: str = 'v3',
                 objective_subtype: str = '1C',   # '1C' (RegAlign) or '2B' (SSDA)
                 total_steps: int = 500,
                 eval_metric: str = 'auc',
                 loss_mode: str = None,  # 'ce','align','ce+align' (1C) | 'entropy','align','entropy+align' (2B)
                 # target domain (few-shot/regalign or unlabeled/ssda)
                 target_X: Optional[np.ndarray] = None,
                 target_y: Optional[np.ndarray] = None,
                 # validation for logging
                 X_valid: Optional[np.ndarray] = None,
                 y_valid: Optional[np.ndarray] = None,
                 callbacks: Optional[list] = None,
                 # --- method hyper-params (tuned by grid) ---
                 tau: float = 1.0,     # target CE / entropy weight
                 eps: float = 1.0,     # alignment/GBA weight
                 source_focal_gamma: float = 0.0,
                 target_pos_weight: float = 1.0,
                 update_every: int = 25,
                 fair_grad_alpha: float = 0.0, fair_grad_ema: float = 0.9, eq_lambda: float = 0.25, cal_lambda: float = 0.2,
                 gap_kind: str = 'ba',
                 class1_boost: float = 1.0,
                 lambda_align: float = 0.1,
                 # ----- NEW knobs for V3 -----
                 target_focal_gamma: float = 0.0,
                 use_target_reg: bool = False,
                 target_weight: float = 1.0,
                 # (ALL-groups homogenization)
                 bary_align: bool = False,
                 var_align: bool = False,
                 center_on: str = 'barycenter',   # or 'source'|'target'
                 group_ids_source: Optional[np.ndarray] = None,
                 group_ids_target: Optional[np.ndarray] = None,
                 # --- plain xgb params (also tuned by grid) ---
                 **xgb_params):
        self.objective_version = objective_version
        self.objective_subtype = objective_subtype
        self.total_steps = total_steps
        self.eval_metric = eval_metric
        self.target_X = target_X
        self.target_y = target_y
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.callbacks = callbacks
        self.loss_mode = loss_mode

        # method HPs
        self.tau = tau
        self.eps = eps
        self.source_focal_gamma = source_focal_gamma
        self.target_pos_weight  = target_pos_weight
        self.update_every       = update_every
        self.class1_boost       = class1_boost
        self.lambda_align       = float(lambda_align)

        # ---- NEW: V3 knobs ----
        self.target_focal_gamma = target_focal_gamma
        self.use_target_reg     = use_target_reg
        self.target_weight      = target_weight

        self.bary_align     = bary_align
        self.var_align      = var_align
        self.center_on      = center_on
        self.group_ids_source = group_ids_source
        self.group_ids_target = group_ids_target
        
        self.fair_grad_alpha = fair_grad_alpha
        self.fair_grad_ema   = fair_grad_ema
        self.eq_lambda       = eq_lambda
        self.cal_lambda      = cal_lambda
        self.gap_kind        = gap_kind

        # defaults + alias normalization (unchanged)
        defaults = dict(
            max_depth=3, eta=0.3, reg_lambda=1.0, reg_alpha=0.0,
            subsample=1.0, colsample_bytree=1.0,
            tree_method="hist", device=xgb_params.get("device", "cuda"),
            objective="binary:logistic",
            scale_pos_weight=1.0, nthread=8, verbosity=0,
            eval_metric=self.eval_metric
        )
        defaults.update(xgb_params)
        if 'lambda' in defaults and 'reg_lambda' not in defaults:
            defaults['reg_lambda'] = defaults.pop('lambda')
        if 'alpha' in defaults and 'reg_alpha' not in defaults:
            defaults['reg_alpha'] = defaults.pop('alpha')
        if 'eta' not in defaults and 'learning_rate' in defaults:
            defaults['eta'] = defaults.pop('learning_rate')
        defaults['objective'] = 'binary:logistic'
        defaults.setdefault('eval_metric', self.eval_metric)
        defaults.setdefault('max_delta_step', 1)
        self.xgb_params: Dict = defaults

        # fitted state
        self.booster_ = None
        self.classes_ = np.array([0, 1])

    def get_params(self, deep=True):
        p = dict(self.xgb_params)
        p.update(dict(
            objective_version=self.objective_version,
            objective_subtype=self.objective_subtype,
            total_steps=self.total_steps,
            eval_metric=self.eval_metric,
            target_X=self.target_X,
            target_y=self.target_y,
            X_valid=self.X_valid,
            y_valid=self.y_valid,
            callbacks=self.callbacks,
            # method HPs
            tau=self.tau, eps=self.eps,
            source_focal_gamma=self.source_focal_gamma,
            target_pos_weight=self.target_pos_weight,
            update_every=self.update_every,
            class1_boost=self.class1_boost,
            loss_mode=self.loss_mode,
            # NEW V3 knobs
            target_focal_gamma=self.target_focal_gamma,
            use_target_reg=self.use_target_reg,
            target_weight=self.target_weight,
            lambda_align=self.lambda_align,
            bary_align=self.bary_align,
            var_align=self.var_align,
            center_on=self.center_on,
            group_ids_source=self.group_ids_source,
            group_ids_target=self.group_ids_target,
            gap_lambda=float(getattr(self, "gap_lambda", 1.0)),
            ba_tau=float(getattr(self, "ba_tau", 2.0)),
            fair_grad_alpha=self.fair_grad_alpha,
            fair_grad_ema=self.fair_grad_ema,
            eq_lambda=self.eq_lambda,
            cal_lambda=self.cal_lambda,
            gap_kind=self.gap_kind
        ))
        return p

    def set_params(self, **params):
        known = {
             'objective_version','objective_subtype','total_steps','eval_metric',
            'target_X','target_y','X_valid','y_valid','callbacks',
            'tau','eps','source_focal_gamma','target_pos_weight','update_every',
            'class1_boost','loss_mode','target_focal_gamma','use_target_reg','lambda_align',
            'target_weight','bary_align','var_align','center_on',
            'group_ids_source','group_ids_target','gap_lambda','ba_tau','fair_grad_alpha',
            'fair_grad_ema','eq_lambda','cal_lambda','gap_kind'
        }
        for k, v in params.items():
            if k in known:
                setattr(self, k, v)
            else:
                self.xgb_params[k] = v
        return self

    def _build_custom_obj(self, X_src, y_src, X_tgt, y_tgt):
        n_source = X_src.shape[0]
        m_target = 0 if X_tgt is None else X_tgt.shape[0]
        X_comb   = X_src if m_target == 0 else np.vstack([X_src, X_tgt])

        kwargs = dict(
            total_steps=self.total_steps,
            n_source=n_source,
            m_target=m_target,
            tau=float(self.tau),
            eps=float(self.eps),
            approach=self.objective_subtype,            # '1C' or '2B'
            loss_mode=self.loss_mode,                   # same modes as v2
            source_focal_gamma=float(self.source_focal_gamma),
            target_pos_weight=float(self.target_pos_weight),
            update_every=int(self.update_every),
            gap_kind=self.gap_kind,
            fair_grad_alpha=float(self.fair_grad_alpha),
            fair_grad_ema=float(self.fair_grad_ema),
            eq_lambda=float(self.eq_lambda),
            cal_lambda=float(self.cal_lambda),
            class1_boost=float(self.class1_boost),
            lambda_align=float(self.lambda_align),
            # v1-style options (restored)
            target_focal_gamma=float(self.target_focal_gamma),
            use_target_reg=bool(self.use_target_reg),
            target_weight=float(self.target_weight),
            # ALL-groups homogenization
            bary_align=bool(self.bary_align),
            var_align=bool(self.var_align),
            center_on=str(self.center_on),
            group_ids_source=self.group_ids_source,
            group_ids_target=self.group_ids_target,
            gap_lambda=float(getattr(self, "gap_lambda", 1.0)),
            ba_tau=float(getattr(self, "ba_tau", 2.0)),
            X_combined=X_comb,
            y_source=y_src
        )
        try:
            method_keys = [
                #"tau","eps","source_focal_gamma","target_focal_gamma","target_pos_weight",
                #"class1_boost","lambda_align",
                #"target_weight","eq_lambda","cal_lambda",
                "fair_grad_alpha","fair_grad_ema","gap_lambda","ba_tau","update_every",
                #"bary_align","var_align","center_on"
            ]
            meth = {k: kwargs.get(k) for k in method_keys if k in kwargs}
            #print("[DEBUG][CustomObjV3] method_params:", json.dumps(meth, default=str))
            #print("[DEBUG][CustomObjV3] xgb_params:", json.dumps(self.xgb_params, default=str))
        except Exception:
            pass
        # ---- V3 objective with GBA + smooth BA-gap + v1/v2-compat knobs ----
        return CO.CustomObjectiveV3(**kwargs)

    def fit(self, X, y):
        X_src = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.pos_rate_ = float(np.mean(y)) if y.size else 0.5
        
        X_tgt = None if self.target_X is None else np.asarray(self.target_X, dtype=np.float32)
        y_tgt = None if self.target_y is None else np.asarray(self.target_y, dtype=np.float32).ravel()

        # Align group ids lengths with source/target
        gid_src = None
        if getattr(self, "group_ids_source", None) is not None:
            gid_src = np.asarray(self.group_ids_source).astype(int)
            if gid_src.shape[0] != X_src.shape[0]:
                print(f"[XGB-CUSTOM] WARN: group_ids_source len {len(gid_src)} != X_src len {len(X_src)}; resizing.")
                if gid_src.shape[0] == 0:
                    gid_src = np.zeros(X_src.shape[0], dtype=int)
                elif gid_src.shape[0] > X_src.shape[0]:
                    gid_src = gid_src[:X_src.shape[0]]
                else:
                    gid_src = np.resize(gid_src, X_src.shape[0])
        gid_tgt = None
        if X_tgt is not None and getattr(self, "group_ids_target", None) is not None:
            gid_tgt = np.asarray(self.group_ids_target).astype(int)
            if gid_tgt.shape[0] != X_tgt.shape[0]:
                print(f"[XGB-CUSTOM] WARN: group_ids_target len {len(gid_tgt)} != X_tgt len {len(X_tgt)}; resizing.")
                if gid_tgt.shape[0] == 0:
                    gid_tgt = np.zeros(X_tgt.shape[0], dtype=int)
                elif gid_tgt.shape[0] > X_tgt.shape[0]:
                    gid_tgt = gid_tgt[:X_tgt.shape[0]]
                else:
                    gid_tgt = np.resize(gid_tgt, X_tgt.shape[0])

        if X_tgt is None:
            dtrain = xgb.DMatrix(X_src, label=y)
        else:
            if y_tgt is None:
                y_comb = np.concatenate([y, np.zeros(X_tgt.shape[0], dtype=np.float32)], axis=0)
            else:
                y_comb = np.concatenate([y, y_tgt], axis=0)
            dtrain = xgb.DMatrix(np.vstack([X_src, X_tgt]), label=y_comb.ravel())

        # optional validation for logging
        evals = [(dtrain, 'train')]
        if self.X_valid is not None and self.y_valid is not None:
            Xv = np.asarray(self.X_valid, dtype=np.float32)
            yv = np.asarray(self.y_valid, dtype=np.float32).ravel()
            dvalid = xgb.DMatrix(Xv, label=yv)
            evals.append((dvalid, 'valid'))

        # Override gids for this fit
        if gid_src is not None:
            self.group_ids_source = gid_src
        if gid_tgt is not None:
            self.group_ids_target = gid_tgt

        custom_obj = self._build_custom_obj(X_src, y, X_tgt, y_tgt)

        try:
            # # Collect eval names safely
            # eval_names = [name for (_, name) in (evals or [])]

            # # Method HPs we care to see (only include those present on self)
            # hp_keys = [
            #     "tau", "eps", "target_weight",
            #     "source_focal_gamma", "target_focal_gamma",
            #     "target_pos_weight", "class1_boost",
            #     "gap_lambda", "ba_tau", "update_every"
            # ]
            # method_hparams = {k: getattr(self, k) for k in hp_keys if hasattr(self, k)}

            # debug_info = {
            #     "branch": "custom-xgboost-train",
            #     "loss_mode": getattr(self, "loss_mode", None),
            #     "approach": getattr(self, "approach", None),
            #     "total_steps": int(self.total_steps),
            #     "xgb_params": self.xgb_params,
            #     "method_hparams": method_hparams,
            #     "train_rows": getattr(dtrain, "num_row", lambda: None)(),
            #     "train_cols": getattr(dtrain, "num_col", lambda: None)(),
            #     "evals": eval_names,
            #     "callbacks": [type(cb).__name__ for cb in (self.callbacks or [])],
            # }
            # print("[XGB-CUSTOM] Starting training with:", flush=True)
            # print(json.dumps(debug_info, indent=2, default=str), flush=True)
            
            for bad in ("n_jobs", "n_estimators"):
                if bad in self.xgb_params:
                    self.xgb_params.pop(bad, None)
            
            self.booster_ = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain,
                num_boost_round=int(self.total_steps),
                obj=custom_obj,
                evals=evals,
                callbacks=(self.callbacks if self.callbacks is not None else []),
                verbose_eval=False
            )
            # # Success log
            # btype = type(self.booster_).__name__
            # try:
            #     attrs = self.booster_.attributes()
            # except Exception:
            #     attrs = {}
            # print(f"[XGB-CUSTOM] Training finished. booster={btype}, attrs_keys={list(attrs.keys())}", flush=True)
        except Exception as e:
             # Failure path → DummyBooster
            print("[XGB-CUSTOM] TRAINING FAILED — falling back to _DummyBooster.", flush=True)
            print("[XGB-CUSTOM] Exception:", repr(e), flush=True)
            traceback.print_exc()
            class _DummyBooster:
                def predict(self, dm):
                    n = dm.num_row()
                    p = np.full(n, fill_value=max(1e-6, min(1-1e-6, getattr(self, "p_", 0.5))), dtype=np.float32)
                    return p
            db = _DummyBooster()
            db.p_ = self.pos_rate_
            self.booster_ = db
            self.used_dummy_booster_ = True
            print(f"[XGB-CUSTOM] Using _DummyBooster with constant p_={self.pos_rate_:.6f}", flush=True)

        return self

    def predict_proba(self, X):
        if getattr(self, "used_dummy_booster_", False) and not getattr(self, "_warned_dummy_", False):
            print("[XGB-CUSTOM] WARNING: predictions come from _DummyBooster (constant probs).", flush=True)
            self._warned_dummy_ = True
        
        if self.booster_ is None:
            # constant fallback
            n = len(X)
            p = np.full(n, fill_value=max(1e-6, min(1-1e-6, getattr(self, "pos_rate_", 0.5))), dtype=np.float32)
            return np.vstack([1 - p, p]).T
        Xs = np.asarray(X, dtype=np.float32)
        p = self.booster_.predict(xgb.DMatrix(Xs))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        if getattr(self, "used_dummy_booster_", False) and not getattr(self, "_warned_dummy_", False):
            print("[XGB-CUSTOM] WARNING: predictions come from _DummyBooster (constant probs).", flush=True)
            self._warned_dummy_ = True
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        
    def _xgb_vanilla_predict_proba(self, X):
        if hasattr(self.predictor, "predict_proba"):
            return self.predictor.predict_proba(X)
        import numpy as np
        if hasattr(self.predictor, "decision_function"):
            s = np.asarray(self.predictor.decision_function(X))
            if s.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-s))
                return np.vstack([1.0 - p1, p1]).T
        raise AttributeError("No predict_proba/decision_function available.")

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        thr = getattr(self, "best_threshold_", 0.5)
        return (p1 >= thr).astype(int)
