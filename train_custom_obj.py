# train.py
import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

sys.path.append("/home/Codes/ad_classification")
from custom_objectives import CustomObjective, sigmoid
from train_utils import select_few_shot_target

def evaluate_param(param_tuple,
                   combined_X, combined_y,
                   n_source, total_steps,
                   inner_cv, approach):
    """
    Inner‐CV evaluation of one hyperparameter tuple on the combined training set.
    Returns mean balanced‐accuracy across the inner folds.
    """
    max_depth, eta, reg_lambda, reg_alpha, weight = param_tuple

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

    scores = []
    for tr_idx, val_idx in inner_cv.split(combined_X, combined_y):
        # --- reorder train‐fold so source come first, then target few‐shot ---
        src_idx = [i for i in tr_idx if i < n_source]
        tgt_idx = [i for i in tr_idx if i >= n_source]
        train_order = src_idx + tgt_idx

        X_tr = combined_X[train_order]
        y_tr = combined_y[train_order]
        n_s  = len(src_idx)
        m_t  = len(tgt_idx)

        obj = CustomObjective(
            total_steps=total_steps,
            n_source=n_s,
            m_target=m_t,
            tau=1.0,
            eps=1e-7,
            approach=approach
        )
        dtrain = xgb.DMatrix(X_tr, label=y_tr)

        # validation‐fold (mixed source+few‐shot)
        X_val = combined_X[val_idx]
        y_val = combined_y[val_idx]
        dval  = xgb.DMatrix(X_val, label=y_val)

        ev = [(dtrain, 'train'), (dval, 'eval')]
        bst = xgb.train(
            params_current, dtrain, total_steps,
            obj=obj, evals=ev,
            early_stopping_rounds=100,
            verbose_eval=False
        )

        raw = bst.predict(dval, iteration_range=(0, bst.best_iteration+1))
        pred = (sigmoid(raw) >= 0.5).astype(int)
        scores.append(balanced_accuracy_score(y_val, pred))

    return float(np.mean(scores)) if scores else 0.0

def nested_train_and_eval(train_df, test_df, feature_cols, args):
    """
    1) Build few-shot target from test_df.
    2) Normalize on source.
    3) Inner CV over combined (source + few-shot) to pick best hyperparams.
    4) Retrain on full combined, then evaluate on full test_df.
    """
    # --- prepare source / few-shot target / full-target-eval ---
    X_src  = train_df[feature_cols].values
    y_src  = train_df["NACCUDSD"].values.astype(int)
    tgt_few= select_few_shot_target(test_df, args.scenario, args.num_few_shot)
    X_tgt_train = tgt_few[feature_cols].values
    y_tgt_train = tgt_few["NACCUDSD"].values.astype(int)
    X_tgt_eval  = test_df[feature_cols].values
    y_tgt_eval  = test_df["NACCUDSD"].values.astype(int)

    # --- scale on source, apply to target train+eval ---
    scaler = StandardScaler().fit(X_src)
    X_src_norm     = scaler.transform(X_src)
    X_tgt_train_nm = scaler.transform(X_tgt_train)
    X_tgt_eval_nm  = scaler.transform(X_tgt_eval)

    # --- combined training set ---
    combined_X = np.vstack([X_src_norm, X_tgt_train_nm])
    combined_y = np.concatenate([y_src, y_tgt_train])
    n_source   = X_src_norm.shape[0]
    total_steps= args.total_steps

    # --- inner CV splitter ---
    inner_cv = StratifiedKFold(
        n_splits=args.inner_splits,
        shuffle=True,
        random_state=42
    )

    # --- hyperparameter grid (no approach dimension; approach fixed by args) ---
    param_grid = {
        'max_depth':       [2,3,4,5,6],
        'eta':             [0.1,0.3,0.5,0.7,1],
        'lambda':          [1,3,5,7],
        'alpha':           [0,1,3,5],
        'scale_pos_weight':[1.0,1.15,1.3,1.45]
    }
    param_combos = list(itertools.product(
        param_grid['max_depth'],
        param_grid['eta'],
        param_grid['lambda'],
        param_grid['alpha'],
        param_grid['scale_pos_weight']
    ))

    # --- inner CV search ---
    inner_scores = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(evaluate_param)(
            p, combined_X, combined_y,
            n_source, total_steps,
            inner_cv, args.objective_subtype
        )
        for p in tqdm(param_combos, desc=" Inner grid ")
    )
    best_idx    = int(np.argmax(inner_scores))
    best_params = param_combos[best_idx]

    # --- final train on full combined set and eval on full test_df ---
    max_depth, eta, reg_lambda, reg_alpha, weight = best_params
    params_final = {
        'max_depth': max_depth,
        'eta': eta,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'auc',
        'verbosity': 0,
        'lambda': reg_lambda,
        'alpha': reg_alpha,
        'scale_pos_weight': weight
    }
    # one CustomObjective on full combined
    obj_final = CustomObjective(
        total_steps=total_steps,
        n_source=n_source,
        m_target=X_tgt_train_nm.shape[0],
        tau=1.0,
        eps=1e-7,
        approach=args.objective_subtype
    )
    dtrain_full = xgb.DMatrix(combined_X, label=combined_y)
    dtest_full  = xgb.DMatrix(X_tgt_eval_nm)

    bst = xgb.train(
        params_final, dtrain_full, total_steps,
        obj=obj_final,
        evals=[(dtrain_full,'train')],
        verbose_eval=False
    )

    raw_preds = bst.predict(dtest_full, iteration_range=(0, bst.best_iteration+1))
    prob_preds= sigmoid(raw_preds)
    bin_preds = (prob_preds >= 0.5).astype(int)

    # compute metrics
    ba  = balanced_accuracy_score(y_tgt_eval, bin_preds)
    tn, fp, fn, tp = confusion_matrix(y_tgt_eval, bin_preds).ravel()
    fpr = fp/(fp+tn+1e-6)
    fnr = fn/(fn+tp+1e-6)

    return bst, raw_preds, (ba,fpr,fnr), best_params

def train_pipeline(args):
    src        = os.path.dirname(os.path.abspath(__file__))
    scenarios  = ['all', 'nhw', 'nha', 'hisp']
    folds      = range(1, args.outer_splits+1)

    for scenario in scenarios:
        args.scenario = scenario
        for fold in folds:
            print(f"\n=== Scenario {scenario}, Fold {fold} ===")
            train_df = pd.read_csv(os.path.join(
                src, "data", "split_fold_w_augmented",
                f"scenario_{scenario}_fold_{fold}_train.csv"
            ))
            test_df = pd.read_csv(os.path.join(
                src, "data", "split_fold_w_augmented",
                f"scenario_{scenario}_fold_{fold}_test.csv"
            ))

            non_feat    = ["ID","NACCADC","NACCMRIA","SEX","HISPANIC","RACE",
                           "RACE_HISP","NACCUDSD"]
            feature_cols= [c for c in train_df.columns if c not in non_feat]

            # nested CV + final eval
            model, raw_preds, (ba,fpr,fnr), best_params = nested_train_and_eval(
                train_df, test_df, feature_cols, args
            )

            # --- Save model ---
            model_dir = os.path.join(
                args.save_model_path,
                f"xgb_{args.objective_subtype}"
            )
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"scenario_{scenario}_fold_{fold}_model.json"
            model.save_model(os.path.join(model_dir, model_filename))

            # --- Save predictions ---
            pred_df = test_df[["ID","RACE","HISPANIC","NACCUDSD"]].copy()
            pred_df["PREDICTION"] = raw_preds
            pred_dir = os.path.join(
                args.save_result_path,
                f"xgb_{args.objective_subtype}"
            )
            os.makedirs(pred_dir, exist_ok=True)
            pred_filename = f"scenario_{scenario}_fold_{fold}_predict.csv"
            pred_df.to_csv(os.path.join(pred_dir, pred_filename), index=False)

            # --- Print overall performance ---
            print("Overall Performance:")
            print(f"  Balanced Accuracy: {ba:.4f}")
            print(f"  False Positive Rate: {fpr:.4f}")
            print(f"  False Negative Rate: {fnr:.4f}")
            print("Best Hyperparameters:", best_params)

            # --- Subgroup performance ---
            print("Subgroup Performance:")
            sig = sigmoid(raw_preds)
            binp = (sig >= 0.5).astype(int)
            subs = {
                "NHW":      (test_df["RACE"]==1)&(test_df["HISPANIC"]==0),
                "NHA":      (test_df["RACE"]==2)&(test_df["HISPANIC"]==0),
                "Hispanic": (test_df["RACE"]==1)&(test_df["HISPANIC"]==1)
            }
            for name, cond in subs.items():
                idx = cond.values
                if idx.sum()==0:
                    print(f"  {name}: no samples")
                    continue
                y_true = test_df["NACCUDSD"].values[idx]
                y_pr   = binp[idx]
                ba_s   = balanced_accuracy_score(y_true, y_pr)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pr).ravel()
                fpr_s = fp/(fp+tn+1e-6)
                fnr_s = fn/(fn+tp+1e-6)
                print(f"  {name}: BA={ba_s:.4f}, FPR={fpr_s:.4f}, FNR={fnr_s:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested‐CV train with custom objective 1C or 2B"
    )
    parser.add_argument("--scenario",         type=str, default="all")
    parser.add_argument("--save_model_path",  type=str,
                        default="/home/Codes/ad_classification/models")
    parser.add_argument("--save_result_path", type=str,
                        default="/home/Codes/ad_classification/results")
    parser.add_argument("--objective_subtype",type=str, choices=['1C','2B'],
                        default='1C',
                        help="Use only '1C' or '2B'")
    parser.add_argument("--num_few_shot",     type=int, default=5)
    parser.add_argument("--total_steps",      type=int, default=5000)
    parser.add_argument("--inner_splits",     type=int, default=10)
    parser.add_argument("--outer_splits",     type=int, default=10)
    parser.add_argument("--n_jobs",           type=int, default=8)
    args = parser.parse_args()
    start = time.time()
    train_pipeline(args)
    print(f"\nFinished in {time.time()-start:.1f}s")
