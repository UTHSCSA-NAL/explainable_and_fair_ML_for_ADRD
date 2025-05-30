import sys
sys.path.append("/home/Codes/ad_classification")

import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score

from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS

from cvxopt import matrix, solvers

def custom_f1_score(y_true, y_pred):
    # Example: You can modify this function to compute any custom metric
    return f1_score(y_true, y_pred, average='weighted')

def custom_ba_score(y_true, y_pred):
    # Example: You can modify this function to compute any custom metric
    return balanced_accuracy_score(y_true, y_pred)


def safe_log1p(x):
    """Calculates log(1+x) safely, handling negative and large values."""
    # Handle negative x
    x = np.where(x < 0, np.finfo(x.dtype).tiny, x)  # Replace negative with tiny positive

    # Handle large x
    return np.where(x > 1e8, np.log(x) + np.log1p(1/x), np.log1p(x))

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
    B        = 1000
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

class CustomML(BaseEstimator):
    def __init__(self, model_name, paras=None, sample_weight=None, **kwargs):
        self.model_name = model_name
        self.paras = paras if paras is not None else {}
        #print("Init:", self.paras)
        self.sample_weight = sample_weight
        self.kwargs = kwargs
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        # Initialize the SVC with current parameters
        if self.model_name=='svm':
            self.predictor = SVC(
                **self.paras,
                #probability=True, 
                #class_weight='balanced', 
                **self.kwargs
            )
        elif self.model_name=='xgboost':
            self.predictor = xgb.XGBClassifier(
                **self.paras,
                **self.kwargs
            )
        elif self.model_name=='random_forest':
            self.predictor = RandomForestClassifier(
                **self.paras,
                **self.kwargs
            )
        elif self.model_name=='mlp':
            self.predictor = MLPClassifier(
                **self.paras,
                **self.kwargs
            )
        
    def set_params(self, **params):
        self.paras = params
        self._initialize_predictor()
        return self
    
    def fit(self, X, y, sample_weight=None):
        self.predictor.fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):        
        return self.predictor.predict(X)
    
    def predict_proba(self, X):
        return self.predictor.predict_proba(X)
    
    def compute_shap_values(self, X_train, X_test, K=None):
        if self.model_name=='xgboost':
            # data = xgb.DMatrix(X_train)
            # shap_values = self.predictor.predict(data, pred_contribs=True)
            if self.sample_weight:
                shap_values = self.predictor._Booster.predict(xgb.DMatrix(X_train, weight=self.sample_weight), pred_contribs=True)
            else:
                shap_values = self.predictor._Booster.predict(xgb.DMatrix(X_train), pred_contribs=True)
            return shap_values
        else:
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
    est = clone(estimator).set_params(**parameters)

    # --- slice training data safely ---
    if hasattr(X, "iloc"):
        X_tr = X.iloc[train]
        y_tr = y.iloc[train]
    else:
        X_tr = X[train]
        y_tr = y[train]

    if est.sample_weight is not None:
        sw = est.sample_weight[train]
        est.fit(X_tr, y_tr, sample_weight=sw)
    else:
        est.fit(X_tr, y_tr)

    # --- compute score ---
    if X_test is not None and y_test is not None:
        score_list = [scorer(est, xt, yt) for xt, yt in zip(X_test, y_test)]
        score = sum(score_list) / len(score_list)
    else:
        if hasattr(X, "iloc"):
            X_te = X.iloc[test]
            y_te = y.iloc[test]
        else:
            X_te = X[test]
            y_te = y[test]
        score = scorer(est, X_te, y_te)

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
        self.param_iter = list(ParameterGrid(self.param_grid))

        for split_idx, (train, test) in enumerate(cv.split(X, y)):
            print(f"        =========== Training Split {split_idx} ===========        ")
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
                        self.best_estimator_ = clone(est)

                    # update this fold’s best
                    if score > self.best_score_fold:
                        self.best_score_fold  = score
                        self.best_params_fold = parameters

                    # record fold‐test score & advance bar
                    self.test_score_fold.append(score)
                    pbar.update(1)

            # after finishing this fold
            self.best_test_scores.append(self.best_score_fold)
            self.all_best_parameters_[f'split{split_idx}_params'] = self.best_params_fold
            self.split_indices['train'].append(train)
            self.split_indices['test'].append(test)

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

        