import sys
sys.path.append("/home/Codes/ad_classification")

import numpy as np
import cupy as cp # type: ignore
import shap # type: ignore
from tqdm import tqdm # type: ignore
from multiprocessing import Pool
import xgboost as xgb # type: ignore
from sklearn.model_selection import StratifiedKFold, ParameterGrid, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
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

from cvxopt import matrix, solvers # type: ignore

import scipy.optimize as optim
import ops
#import losses as L

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

class CustomEnsemble(BaseEstimator):
    def __init__(self, init_paras=None, **kwargs):
        self.init_paras = init_paras
        self.initial_base_models(self.init_paras)
    
    def initial_base_models(self, params):
        svm_model = SVC(**params['SVM'])
        #lr_model  = LinearRegression(**params['LR'])
        mlp_model = MLPClassifier(**params['MLP'])
        xgb_model = xgb.XGBClassifier(**params['XGB'])
        
        self.predictors = [svm_model, mlp_model, xgb_model]
    
    def set_params(self, **params):
        self.params = params
        self.initial_base_models(self.params)
        return self
    
    def fit(self, X, y):
        for i in range(len(self.predictors)):
            self.predictors[i].fit(X, y)
    
    def compute_relative_weight(self, X_val, y_val):
        X_val_cu = cp.array(X_val)
        model_predictions = list()
        for i in range(len(self.predictors)):
            # if i==0:
            #     y_pred = L.sigmoid(self.predictors[i].predict(X_val))
            #     y_pred_2d = np.vstack((1-y_pred, y_pred)).T
            #     model_predictions.append(y_pred_2d)
            if i==(len(self.predictors)-1):
                self.predictors[i].set_params(device="cuda")
                model_predictions.append(self.predictors[i].predict_proba(X_val_cu))
            else:
                model_predictions.append(self.predictors[i].predict_proba(X_val))

        validation_losses = np.array([L.binary_cross_entropy(y_val, y_pred[:,1]) for y_pred in model_predictions])
        epsilon = 1e-8
        normalized_losses = 1 / (validation_losses + epsilon)
        relative_weights = normalized_losses / np.sum(normalized_losses)
        self.relative_weights = relative_weights        
        
    def predict(self, X):
        X_cu = cp.array(X)
        model_predictions = list()
        for i in range(len(self.predictors)):
            # if i==0:
            #     y_pred = L.sigmoid(self.predictors[i].predict(X_val))
            #     y_pred_2d = np.vstack((1-y_pred, y_pred)).T
            #     model_predictions.append(y_pred_2d)
            if i==(len(self.predictors)-1):
                self.predictors[i].set_params(device="cuda")
                model_predictions.append(self.predictors[i].predict_proba(X_cu))
            else:
                model_predictions.append(self.predictors[i].predict_proba(X))
        
        final_prediction = np.sum([weight * pred for weight, pred in zip(self.relative_weights, model_predictions)], axis=0)

        return np.argmax(final_prediction, axis=1)
    
    def predict_proba(self, X):
        X_cu = cp.array(X)
        model_predictions = list()
        for i in range(len(self.predictors)):
            # if i==0:
            #     y_pred = L.sigmoid(self.predictors[i].predict(X_val))
            #     y_pred_2d = np.vstack((1-y_pred, y_pred)).T
            #     model_predictions.append(y_pred_2d)
            if i==(len(self.predictors)-1):
                self.predictors[i].set_params(device="cuda")
                model_predictions.append(self.predictors[i].predict_proba(X_cu))
            else:
                model_predictions.append(self.predictors[i].predict_proba(X))
        
        final_prediction = np.sum([weight * pred for weight, pred in zip(self.relative_weights, model_predictions)], axis=0)

        return final_prediction
    
    @staticmethod
    def hyperparameter_optimization(X_train, y_train, X_test=None, y_test=None, cv=None,
                                    init_paras=None, param_grid=None, n_jobs=-1):
        
        custom_scorer = make_scorer(custom_ba_score, greater_is_better=True)
        
        opt = EnsembleGridSearchCV(estimator=CustomEnsemble(init_paras=init_paras),
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

def fit_and_score_ensemble(estimator, X, y, train, test, parameters, scorer, X_test=None, y_test=None):
    est = clone(estimator).set_params(**parameters)
    est.fit(X[train], y[train])
    est.compute_relative_weight(X[test], y[test])
        
    if X_test is not None and y_test is not None:
        score = [scorer(est, xt, yt) for xt, yt in zip(X_test, y_test)]
        score = sum(score) / len(score)
    else:
        score = scorer(est, X[test], y[test])
    
    return parameters, score, est

class EnsembleGridSearchCV(GridSearchCV):
    def __init__(self, *args, **kwargs):
        super(EnsembleGridSearchCV, self).__init__(*args, **kwargs)
        self.all_best_parameters_ = {}
        self.split_indices = {
            'train': [],
            'test': []
        }
        self.test_score_fold  = []
        self.best_test_scores = []
        self.relative_weights = []
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
        self.best_score_ = -np.inf if self.scorer_._sign > 0 else np.inf
        self.best_params_ = None
        self.best_estimator_ = None
        
        cv = self.cv if self.cv is not None else 5
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling {2} fits"
                  .format(len(cv), len(self.param_grid),
                          len(cv) * len(self.param_grid)))
        
        self.param_iter = list(ops.ParameterGrid(self.param_grid))
        for split_idx, (train, test) in enumerate(cv.split(X, y)):
            print(f"        =========== Training Split {split_idx} ===========        ")
            self.best_score_fold = -np.inf if self.scorer_._sign > 0 else np.inf
            self.best_params_fold = None
            self.best_estimator_fold = None
            
            with tqdm(total=len(self.param_iter), desc="Grid Search Progress",
                      bar_format='{desc:<10}{percentage:3.0f}%|{bar:30}{r_bar}') as pbar:
                for parameters in self.param_iter:
                    parameters, score, est = fit_and_score_ensemble(self.estimator,
                                                                X, y, train, test, 
                                                                parameters, self.scorer_,
                                                                self.X_test, self.y_test)
                    if self.refit:
                        if (self.best_score_ is None or score > self.best_score_):
                            self.best_score_ = score
                            self.best_params_ = parameters
                            self.best_estimator_ = clone(est)
                        if (self.best_score_fold is None or score > self.best_score_fold):
                            self.best_score_fold = score
                            self.best_params_fold = parameters
                            self.best_estimator_fold = clone(est)
                            self.best_weights_fold = est.relative_weights
                    
                    self.test_score_fold.append(score)
                    pbar.update(1)
            
            self.best_test_scores.append(self.best_score_fold)
            self.all_best_parameters_[f'split{split_idx}_params'] = self.best_params_fold            
            self.split_indices['train'].append(train)
            self.split_indices['test'].append(test)
            self.relative_weights.append(self.best_weights_fold)
        
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
        
        # if self.model_name=='xgboost':
        #     print("Train: ", self.paras)
        #     dtrain = xgb.DMatrix(X, y)
        #     self.predictor = xgb.train(self.paras, dtrain, 50)
        # else:
        #     self.predictor.fit(X, y)
    
    def predict(self, X):        
        # if self.model_name=='xgboost':
        #     data = xgb.DMatrix(X)
        #     return np.asarray(self.predictor.predict(data) > 0.5, dtype=np.int64)
        # else:
        return self.predictor.predict(X)
    
    def predict_proba(self, X):
        # if self.model_name=='xgboost':
        #     data = xgb.DMatrix(X)
        #     return self.predictor.predict(data)
        # else:
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
    if est.sample_weight is not None:
        est.fit(X[train], y[train], sample_weight=est.sample_weight[train])
    else:
        est.fit(X[train], y[train])
    
    if X_test is not None and y_test is not None:
        score = [scorer(est, xt, yt) for xt, yt in zip(X_test, y_test)]
        score = sum(score) / len(score)
    else:
        score = scorer(est, X[test], y[test])
    
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
        estimator = self.estimator
        self.scorer_ = self.scoring
        self.best_score_ = -np.inf if self.scorer_._sign > 0 else np.inf
        self.best_params_ = None
        self.best_estimator_ = None
        
        cv = self.cv if self.cv is not None else 5
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling {2} fits"
                  .format(len(cv), len(self.param_grid),
                          len(cv) * len(self.param_grid)))
            
        self.param_iter = list(ParameterGrid(self.param_grid))
        for split_idx, (train, test) in enumerate(cv.split(X, y)):
            print(f"        =========== Training Split {split_idx} ===========        ")
            self.best_score_fold = -np.inf if self.scorer_._sign > 0 else np.inf
            self.best_params_fold = None
            self.best_estimator_fold = None
            
            with tqdm(total=len(self.param_iter), desc="Grid Search Progress",
                      bar_format='{desc:<10}{percentage:3.0f}%|{bar:30}{r_bar}') as pbar:
                # with Pool(processes=4) as pool:
                #     results = [
                #         pool.apply_async(
                #             fit_and_score, 
                #             args=(
                #                 self.estimator, X, y, train, test, parameters, 
                #                 self.scorer_, self.X_test, self.y_test
                #             )
                #         )
                #         for parameters in self.param_iter
                #     ]
                for parameters in self.param_iter:
                    parameters, score, est = fit_and_score(self.estimator,
                                                            X, y, train, test, 
                                                            parameters, self.scorer_,
                                                            self.X_test, self.y_test)
                    if self.refit:
                        if (self.best_score_ is None or score > self.best_score_):
                            self.best_score_ = score
                            self.best_params_ = parameters
                            self.best_estimator_ = clone(est)
                        if (self.best_score_fold is None or score > self.best_score_fold):
                            self.best_score_fold = score
                            self.best_params_fold = parameters
                            self.best_estimator_fold = clone(est)
                    
                    self.test_score_fold.append(score)
                    pbar.update(1)
            
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


# class LFR(Transformer):
#     def __init__(self,
#                  unprivileged_groups,
#                  privileged_groups,
#                  k=5,
#                  Ax=0.01,
#                  Ay=1.0,
#                  Az=50.0,
#                  print_interval=250,
#                  verbose=0,
#                  seed=None):
#         super(LFR, self).__init__(
#             unprivileged_groups=unprivileged_groups,
#             privileged_groups=privileged_groups)
        
#         self.seed = seed

#         self.unprivileged_groups = unprivileged_groups
#         self.privileged_groups = privileged_groups
        
#         if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
#             raise ValueError("Only one unprivileged_group or privileged_group supported.")
        
        