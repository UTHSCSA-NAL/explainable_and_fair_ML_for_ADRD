import sys

import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from joblib import dump

import ops
import models as ML

LOW_LEVEL_FEATURES = ['23','30','31','32','36','37','38','39','47','48','55','56','57','58','59','60',\
    '71','72','73','75','76','100','101','102','103','104','105','106','107','108','109','112','113','114','115','116','117','118','119','120','121',\
    '122','123','124','125','128','129','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149',\
    '150','151','152','153','154','155','156','157','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175',\
    '176','177','178','179','180','181','182','183','184','185','186','187','190','191','192','193','194','195','196','197','198','199','200','201',\
    '202','203','204','205','206','207', \
    '4','11','40','41','49','50','51','52','61','62','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95'
]

# Function to filter data based on race and ethnicity
def filter_data_by_group(df_roi, train_type, train_drop_rate, threshold=1):
    selected_column = ['HISPANIC', 'RACE']
    feat_mode = {
        'NHW': [[0, 0, 1], [1, 2, 3]],
        'NHA': [[0, 0, 1], [2, 1, 3]],
        'Hispanic': [[1, 0, 0], [3, 1, 2]]
    }.get(train_type, None)
    
    df = df_roi[df_roi['DX'].isin([1, 4])]
    df = df[(df['HISPANIC'].isin([0, 1])) & (df['RACE'].isin([1, 2]))].reset_index(drop=True)

    # Drop mixed group, then optionally re-sample
    df_to_drop = df[(df['HISPANIC']==1)&(df['RACE']==2)]
    df = df.drop(df_to_drop.index).reset_index(drop=True)

    if train_drop_rate > 0:
        num_drop = int(len(df_to_drop[df_to_drop['DX']==4]) * train_drop_rate/100)
        drop_i   = df_to_drop[df_to_drop['DX']==4].sample(n=num_drop).index
        df2      = df_to_drop.drop(drop_i)
        keep_i   = df2[df2['DX']==1].sample(
                        n=int(len(df2[df2['DX']==4])*1.2)
                   ).index
        df2      = df2.loc[keep_i.union(df2[df2['DX']==4].index)]
        df       = pd.concat([df, df2], ignore_index=True).reset_index(drop=True)

    if threshold > 0:
        df = get_similarity_population(df, train_type, threshold)

    X = df.drop(['NACCADC','NACCMRIA','SEX','DX'], axis=1)
    y = df['DX'].map({1:0,4:1}).astype(int)
    return X, y, selected_column, feat_mode

def get_similarity_population(df_roi, train_type, threshold=1):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression

    # ── Impute missing values ─────────────────────────────────────────
    df_roi['NACCMRIA'] = df_roi['NACCMRIA'].fillna(df_roi['NACCMRIA'].median())
    df_roi['SEX']      = df_roi['SEX'].fillna(df_roi['SEX'].mode()[0]).astype(int)

    # ── Identify primary vs. secondary groups ──────────────────────────
    if train_type in ['All','NHW']:
        prim = df_roi[(df_roi['RACE']==1)&(df_roi['HISPANIC']==0)]
    elif train_type=='NHA':
        prim = df_roi[(df_roi['RACE']==2)&(df_roi['HISPANIC']==0)]
    else:  # Hispanic
        prim = df_roi[(df_roi['RACE']==1)&(df_roi['HISPANIC']==1)]

    # ── Encode and scale ───────────────────────────────────────────────
    le = LabelEncoder()
    df_roi['sex_encoded'] = le.fit_transform(df_roi['SEX'])
    df_roi['age_scaled']  = StandardScaler().fit_transform(df_roi[['NACCMRIA']])

    # ── Build train/test for similarity model ─────────────────────────
    primary   = prim.copy(); primary['label']=1
    secondary = df_roi[~df_roi.index.isin(prim.index)].copy(); secondary['label']=0

    combined = pd.concat([primary, secondary], ignore_index=True)
    Xs = combined[['age_scaled','sex_encoded']]
    ys = combined['label']

    # ── Fit logistic regression and filter secondaries ────────────────
    lr = LogisticRegression().fit(Xs, ys)
    secondary['similarity'] = lr.predict_proba(secondary[['age_scaled','sex_encoded']])[:,1]
    thresh = secondary['similarity'].min() * threshold
    filtered_secondary = secondary[secondary['similarity'] >= thresh]

    # ── Return only ID-level columns for downstream splitting ─────────
    primary = primary.drop(['age_scaled','sex_encoded','label'], axis=1)
    filtered_secondary = filtered_secondary.drop(
        ['age_scaled','sex_encoded','label','similarity'], axis=1
    )

    return pd.concat([primary, filtered_secondary], axis=0).reset_index(drop=True)

def load_data(df_roi_path: str):
    df = pd.read_csv(df_roi_path, low_memory=False)
    df = df[df['702']>1e5].rename(columns={'NACCUDSD':'DX'})
    return df

# Function to set classifier parameters
def get_default_params(model_name):
    if model_name == 'xgboost':
        return {
            'max_depth': 2, 
            'learning_rate': 1,
            'lamda': 1,
            'alpha': 0,
            'objective': 'binary:logistic',
            'nthread': 8,
            'verbosity': 0,
            'n_jobs': 8,
            'eval_metric': 'auc',
            'device': 'cuda',
            'tree_method': 'hist',
            'scale_pos_weight': 1.
        }
    elif model_name == 'svm':
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True
        }
    elif model_name == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'n_jobs': -1
        }
    elif model_name == 'mlp':
        return {
            'hidden_layer_sizes': 32,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'shuffle': True,
            'random_state': 42
        }

# Function to set grid search parameters
def get_param_grid(model_name):
    if model_name == 'svm':
        return {
            'C': [0.1, 1.0, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale'],
            'class_weight': ['balanced'],
            'probability': [True],
        }
    elif model_name == 'xgboost':
        return {
            'max_depth': [2, 3, 4, 5, 6], 
            'learning_rate': [0.1, 0.3, 0.5, 0.7, 1], 
            'lambda': [1, 3, 5, 7],
            'alpha': [0, 1, 3, 5],
            'objective': ['binary:logistic'],
            'nthread': [8],
            'verbosity': [0],
            'n_jobs': [8],
            'eval_metric': ['auc'],
            'device': ['cuda'],
            'tree_method': ['hist'],
            'scale_pos_weight': [1.,1.15,1.3,1.45]#[1.,1.44,1.96,2.56,3.24,4.]
        }
    elif model_name == 'random_forest':
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 2, 6, 12],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': ['balanced'],
            'n_jobs': [-1]
        }
    elif model_name == 'mlp':
        return {
            'hidden_layer_sizes': [(32,), (64,), (128,), (256,), (512,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'batch_size': ['auto'],
            'learning_rate': ['constant'],
            'learning_rate_init': [0.001],
            'max_iter': [1000],
            'early_stopping': [True],
            'shuffle': [True],
            'random_state': [42]
        }

def main(args):
    currtime = time.strftime("%I%M%p_%b%d%Y")
    starttime = time.time()

    # --- Filter & preprocess ---
    df_roi = load_data(args.df_roi)
    df_roi = df_roi[list(df_roi.columns[:8]) + LOW_LEVEL_FEATURES]
    X, y, sel_col, feat_mode = filter_data_by_group(
        df_roi, args.train_type, args.train_drop_rate, args.sim_threshold
    )

    if args.corr_remove:
        X_tr = ops.correlation_remover(
            X.drop(['ID','HISPANIC','RACE'],axis=1),
            None, sen_features=['RACE_HISP'], alpha=args.corr_alpha
        )
        X.iloc[:,4:] = X_tr

    paras      = get_default_params(args.model_name)
    param_grid = get_param_grid(args.model_name)
    classes    = (
        f"NCvsDem_train-{args.train_type}_rate_{args.train_drop_rate}"
        f"_alpha_{args.corr_alpha}_{args.sample_weight_method}_{currtime}"
    )

    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.save_result_path, exist_ok=True)

    # --- split data for nested CV ---
    if args.train_type=="All":
        X_in, y_in, hr_in = ops.split_train_test_df(X, y, sel_col, feat_mode)
        others = None
    else:
        X_in, X_out_list, y_in, y_out_list, hr_list = ops.split_train_test_df(
            X, y, sel_col, feat_mode
        )
        hr_in = hr_list[0]
        others = {
            "names": [n if n!="Hispanic" else "HWA"
                      for n in {"NHW":["NHA","HWA"],
                                "NHA":["NHW","HWA"],
                                "Hispanic":["NHW","NHA"]}[args.train_type]],
            "X": X_out_list,
            "y": y_out_list
        }

    outer_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    train_df, NHW_df, NHA_df, HWA_df, estimators = [], [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_in, y_in)):
        print(f"\n--- Outer fold {fold+1} ---")
        X_tr = X_in.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_in.iloc[tr_idx].reset_index(drop=True)
        hr_tr= hr_in.iloc[tr_idx].reset_index(drop=True)

        X_te = X_in.iloc[te_idx].reset_index(drop=True)
        y_te = y_in.iloc[te_idx].reset_index(drop=True)
        hr_te= hr_in.iloc[te_idx].reset_index(drop=True)

        # sample‐weights
        if args.sample_weight_method=='kmm':
            Xs       = ops.scaling_data(X_tr.iloc[:,1:], scaling_mode=args.data_scaling)
            idx_src  = hr_tr.index[(hr_tr['HISPANIC']==0)&(hr_tr['RACE']==1)]
            sw       = ops.compute_sample_weight_kmm(Xs[idx_src], Xs[idx_src])
            sample_weight = ops.normalize_weight(sw)
        else:
            sample_weight = None

        # --- inner CV grid search ---
        base = ML.CustomML(args.model_name, paras=paras, sample_weight=sample_weight)
        grid = base.hyperparameter_optimization(
            X_tr.iloc[:,1:], y_tr,
            args.model_name,
            cv=inner_cv,
            init_paras=paras,
            sample_weight=sample_weight,
            param_grid=param_grid,
            n_jobs=-1
        )
        best = grid.best_params_
        print("  Best params:", best)

        # --- retrain on full outer train ---
        model = clone(base).set_params(**best)
        model.fit(
            ops.scaling_data(X_tr.iloc[:,1:], scaling_mode=args.data_scaling),
            y_tr,
            sample_weight=sample_weight
        )

        # Save training predictions
        p_tr   = model.predict_proba(ops.scaling_data(X_tr.iloc[:,1:], scaling_mode=args.data_scaling))[:,1]
        ytr_df = y_tr.to_frame(); ytr_df['PRED_DX']=p_tr; ytr_df['ID']=X_tr['ID']
        train_df.append({'data':X_tr, 'y':ytr_df[["ID","DX","PRED_DX"]]})

        # Evaluate each group on outer-test
        p_te   = model.predict_proba(ops.scaling_data(X_te.iloc[:,1:], scaling_mode=args.data_scaling))[:,1]
        yte_df = y_te.to_frame(); yte_df['PRED_DX']=p_te; yte_df['ID']=X_te['ID']

        def collect(label, container):
            if args.train_type=="All":
                mask = (
                    (hr_te['HISPANIC']== (1 if label=="HWA" else 0))
                    & (hr_te['RACE']== (1 if label in ["NHW","HWA"] else 2))
                )
                idxs = np.where(mask)[0]
                sub = yte_df[yte_df['ID'].isin(X_te.iloc[idxs]['ID'])]
            else:
                if label==args.train_type or (args.train_type=="Hispanic" and label=="HWA"):
                    sub = yte_df
                else:
                    pos   = others["names"].index(label)
                    X_o   = others["X"][pos]
                    y_o   = others["y"][pos]
                    p_o   = model.predict_proba(
                                ops.scaling_data(X_o.iloc[:,1:], scaling_mode=args.data_scaling)
                            )[:,1]
                    sub = y_o.to_frame(); sub['PRED_DX']=p_o; sub['ID']=X_o['ID']
            container.append({'data':X_te if label in ["NHW","NHA","HWA"] and args.train_type=="All" else X_o,
                              'y': sub[["ID","DX","PRED_DX"]]})

        for lab, cont in [("NHW",NHW_df),("NHA",NHA_df),("HWA",HWA_df)]:
            collect(lab, cont)

        # save model for this fold
        dump(model, f"{args.save_model_path}/{args.model_name}_fold{fold+1}_{classes}.joblib")
        estimators.append(model)

    # Final metrics print
    for i in range(args.n_splits):
        print(f"\n=== Fold {i+1} ===")
        for lab, dfc in [("NHW",NHW_df),("NHA",NHA_df),("HWA",HWA_df)]:
            y_true = dfc[i]["y"]["DX"].values
            y_pred = (dfc[i]["y"]["PRED_DX"]>0.5).astype(int)
            print(f"{lab} — BA: {balanced_accuracy_score(y_true,y_pred):.3f}, "
                  f"F1: {f1_score(y_true,y_pred):.3f}")

    # save all results
    with open(f"{args.save_result_path}/{classes}.pkl","wb") as f:
        pickle.dump([train_df, NHW_df, NHA_df, HWA_df, estimators], f)

    print("All done in", time.time()-starttime, "seconds.")
    

if __name__ == "__main__" and os.getcwd().endswith("dementia_classification"):
    sys.path.insert(0, os.path.abspath(".."))
    parser = argparse.ArgumentParser(description="Training AD Classification")
    # Data Arguments
    parser.add_argument("--save_model_path", type=str, default=r'./models/xgb', help="Saved model path")
    parser.add_argument("--save_result_path", type=str, default=r'./results/xgb', help="Saved result path")
    parser.add_argument("--df_roi", type=str, default=r'./data/filtered_data_09182024.csv', help="Path to CSV data file of MUSE ROI")
    parser.add_argument("--train_type", type=str, default='All', help="Type of training dataset: All, NHW, NHA, or Hispanic")
    parser.add_argument("--train_drop_rate", type=int, default=0, help="Rate to drop training sample")
    parser.add_argument("--model_name", type=str, default='xgboost', help="Type of ML classifier: xgboost, svm")
    parser.add_argument("--sample_weight_method", type=str, default='kmm', help="Method for domain adaptation")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of splitting folds for cross validation")
    parser.add_argument("--data_scaling", type=int, default=0, help="Normalize data")
    parser.add_argument("--corr_remove", type=int, default=0, help="Remove feature correlation")
    parser.add_argument("--corr_alpha", type=float, default=0.05, help="Correlation factor")
    parser.add_argument("--harmonize", type=int, default=0, help="whether using harmonization")
    parser.add_argument("--sim_threshold", type=float, default=1., help="similarity threshold")
    
    args = parser.parse_args()
    main(args)
    