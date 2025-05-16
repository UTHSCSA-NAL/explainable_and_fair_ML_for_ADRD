import sys
sys.path.append("/home/Codes/ad_classification")

import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib
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
    
    df_roi_filtered = df_roi[df_roi['DX'].isin([1, 4])]  # Binary classification: 1 (NC), 4 (Dementia)
    df_roi_filtered = df_roi_filtered[(df_roi_filtered['HISPANIC'].isin([0, 1])) & (df_roi_filtered['RACE'].isin([1, 2]))]
    df_roi_filtered.reset_index(drop=True, inplace=True)
    
    # Filter by HISPANIC and RACE based on feat_mode
    df_to_drop = df_roi_filtered[(df_roi_filtered['HISPANIC'] == 1) & (df_roi_filtered['RACE'] == 2)]
    df_roi_filtered = df_roi_filtered.drop(df_to_drop.index)
    df_roi_filtered.reset_index(drop=True, inplace=True)
    
    if train_drop_rate > 0:
        num_samples_to_drop = int(len(df_to_drop[df_to_drop['DX'] == 4]) * train_drop_rate / 100)
        indices_to_drop = df_to_drop[df_to_drop['DX'] == 4].sample(n=num_samples_to_drop).index
        df_to_drop = df_to_drop.drop(indices_to_drop)
        num_samples_to_keep = int(len(df_to_drop[df_to_drop['DX'] == 4]) * 1.2)
        indices_to_keep = df_to_drop[df_to_drop['DX'] == 1].sample(n=num_samples_to_keep).index
        df_to_drop = df_to_drop.loc[indices_to_keep.union(df_to_drop[df_to_drop['DX'] == 4].index)]
        df_roi_filtered = pd.concat([df_roi_filtered, df_to_drop], ignore_index=True)
    
    df_roi_filtered.reset_index(drop=True, inplace=True)
    if threshold > 0:
        df_roi_filtered = get_similarity_population(df_roi_filtered, train_type, threshold)

    X = df_roi_filtered.drop(['NACCADC', 'NACCMRIA', 'SEX', 'DX'], axis=1)
    y = df_roi_filtered['DX'].map({1: 0, 4: 1}).astype(int)

    return X, y, selected_column, feat_mode

def get_similarity_population(df_roi, train_type, threshold=1):
    df_roi['SEX'] = df_roi['SEX'].astype(int)
    if train_type == 'All' or train_type == 'NHW':
        primary_ids = list(df_roi[(df_roi['RACE'] == 1) & (df_roi['HISPANIC'] == 0)]['ID'].values)
    elif train_type == 'NHA':
        primary_ids = list(df_roi[(df_roi['RACE'] == 2) & (df_roi['HISPANIC'] == 0)]['ID'].values)
    elif train_type == 'Hispanic':
        primary_ids = list(df_roi[(df_roi['RACE'] == 1) & (df_roi['HISPANIC'] == 1)]['ID'].values)
    
    le_sex = LabelEncoder()
    df_roi['sex_encoded'] = le_sex.fit_transform(df_roi['SEX'])
    
    scaler = StandardScaler()
    df_roi['age_scaled'] = scaler.fit_transform(df_roi[['NACCMRIA']])
    
    primary = df_roi[df_roi['ID'].isin(primary_ids)].copy()
    secondary = df_roi[~df_roi['ID'].isin(primary_ids)].copy()
    
    primary['label'] = 1
    secondary['label'] = 0
    
    combined = pd.concat([primary, secondary], ignore_index=True)
    X = combined[['age_scaled', 'sex_encoded']]
    y = combined['label']
    
    model = LogisticRegression()
    model.fit(X, y)

    X_secondary = secondary[['age_scaled', 'sex_encoded']]
    secondary['similarity'] = model.predict_proba(X_secondary)[:, 1]
    threshold = secondary['similarity'].min() * threshold
    filtered_secondary = secondary[secondary['similarity'] >= threshold]
    
    primary = primary.drop(['age_scaled','sex_encoded','label'], axis=1)
    filtered_secondary = filtered_secondary.drop(['age_scaled','sex_encoded','label','similarity'], axis=1)
    
    filter_df = pd.concat([primary, filtered_secondary], axis=0)
    
    return filter_df.reset_index(drop=True)

def load_data(df_roi_path: str):
    df_roi = pd.read_csv(df_roi_path, low_memory=False)
    df_roi = df_roi[df_roi['702'] > 1e5]  # Filter ROI by size
    df_roi = df_roi.rename(columns={'NACCUDSD': 'DX'})
    
    return df_roi

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
    print("=====================================================================")
    print("===================== Filtering Data ================================")
    print("=====================================================================")
    
    df_roi = load_data(args.df_roi)
    df_roi = df_roi[list(df_roi.columns[:8])+LOW_LEVEL_FEATURES]
    X, y, selected_column, feat_mode = filter_data_by_group(df_roi, args.train_type, args.train_drop_rate, args.sim_threshold)
    if args.train_drop_rate > 0:
        print(f"   Drop rate = {args.train_drop_rate}")
    print(f"   Samples of data : {len(X)} samples  ")
    print(X.head(3))
    print(y.head(3))
    
    print("=====================================================================")
    print("===================== Preprocessing Data ============================")
    print("=====================================================================")
    if args.corr_remove==1:
        print("   Processing correlation remover...   ")
        X_transform = ops.correlation_remover(X.drop(['ID','HISPANIC','RACE'], axis=1), None, sen_features=['RACE_HISP'], alpha=args.corr_alpha)
        X.iloc[:,4:] = X_transform
    
    print("=====================================================================")
    print("===================== Parameter Initialization ======================")
    print("=====================================================================")
    print(f"   Classifier Model: {args.model_name}")
    paras = get_default_params(args.model_name)
    print(f"Initialize paremeters: {paras}")

    param_grid = get_param_grid(args.model_name)
    
    print("=====================================================================")
    print("========================= Training Model ============================")
    print("=====================================================================")
    classes = f'NCvsDem_train-{args.train_type}_rate_{args.train_drop_rate}_alpha_{args.corr_alpha}_{args.sample_weight_method}'
    
    if args.harmonize==1:
        classes = classes + '_harm'
    
    if args.corr_remove==1:
        classes = classes + '_cr'
    
    classes = classes + f'_{currtime}'
    
    if args.train_type=="All":
        print(" Training on all groups: ")
        print("")
        
        X_train, y_train, hisp_race_ = ops.split_train_test_df(X, y, selected_column, feat_mode)
        xtrain = X_train.iloc[:,1:].copy()
        ytrain = y_train.copy()
        X_train_, trained_scaler = ops.scaling_data(xtrain, scaling_mode=args.data_scaling, return_scaler=True)
        if args.sample_weight_method=='kmm':
            print("   Using KMM to compute sample weights   ")
            sample_weight = np.ones((X_train_.shape[0],))
            idx_src = hisp_race_.index[(hisp_race_['HISPANIC']==0) & (hisp_race_['RACE']==1)].tolist()
            sample_weight[idx_src] = ops.compute_sample_weight_kmm(X_train_[idx_src], X_train_[idx_src])
            sample_weight = ops.normalize_weight(sample_weight)
        elif args.sample_weight_method=='none':
            print("   No sample weights computed   ")
            sample_weight = None
        
        trained_model = ML.CustomML(model_name=args.model_name, paras=paras, sample_weight=sample_weight)
        hyper_optimizer = trained_model.hyperparameter_optimization(X_train_, ytrain, args.model_name,
                                                                    init_paras=paras, sample_weight=sample_weight, param_grid=param_grid,
                                                                    cv=StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42), n_jobs=-1)
        print("   Saving optimizer ...   ")
        if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path, exist_ok=True)
            print(f"   Create saved model folder: {args.save_model_path}")
        dump(hyper_optimizer, f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}.joblib')
    else:
        print(f" Training on {args.train_type} groups: ")
        print("")
        
        X_train, X_test, y_train, y_test, hisp_race_ = ops.split_train_test_df(X, y, selected_column, feat_mode)
        xtrain = X_train.iloc[:,1:].copy()
        xtest1  = X_test[0].iloc[:,1:].copy()
        xtest2  = X_test[1].iloc[:,1:].copy()
        X_train_, X_test_, trained_scaler = ops.scaling_data(xtrain, X_test=[xtest1, xtest2], scaling_mode=args.data_scaling, return_scaler=True)
        
        if args.sample_weight_method=='kmm':
            print("   Using KMM to compute sample weights   ")
            sample_weights = [ops.compute_sample_weight_kmm(X_train_, Xt) for Xt in X_test_]
        else:
            print("   No sample weights computed   ")
            sample_weights = [None, None]
        
        trained_model_1 = ML.CustomML(model_name=args.model_name, paras=paras, sample_weight=sample_weights[0])
        trained_model_2 = ML.CustomML(model_name=args.model_name, paras=paras, sample_weight=sample_weights[1])
        hyper_optimizer_1 = trained_model_1.hyperparameter_optimization(X_train_, y_train, args.model_name, X_test_[0], y_test[0],
                                                                    init_paras=paras, sample_weight=sample_weights[0], param_grid=param_grid,
                                                                    cv=StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42), n_jobs=-1)
        hyper_optimizer_2 = trained_model_2.hyperparameter_optimization(X_train_, y_train, args.model_name, X_test_[1], y_test[1],
                                                                    init_paras=paras, sample_weight=sample_weights[1], param_grid=param_grid,
                                                                    cv=StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42), n_jobs=-1)
        
        print("   Saving optimizer ...   ")
        if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path, exist_ok=True)
            print(f"   Create saved model folder: {args.save_model_path}")
        dump(hyper_optimizer_1, f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}_1.joblib')
        dump(hyper_optimizer_2, f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}_2.joblib')
        
    print(" Finish training! ")
    
    
    print("=====================================================================")
    print("============================ Prediction =============================")
    print("=====================================================================")
    estimator = []
    if args.train_type=="All":
        train_df, NHW_df, NHA_df, HWA_df = [], [], [], []
        print("   Load trained model   ")
        trained_hyper_optimizer = joblib.load(f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}.joblib')
        for idx in range(args.n_splits):
            train_idx = trained_hyper_optimizer.split_indices['train'][idx]
            test_idx  = trained_hyper_optimizer.split_indices['test'][idx]
            fold_params = trained_hyper_optimizer.all_best_parameters_[f'split{idx}_params']
            print(f"      Load model {idx+1}      ")
            print(f"      Load best parameter : {fold_params}        ")
            est = clone(trained_hyper_optimizer.estimator).set_params(**fold_params)
            
            xtrain, ytrain = X_train.iloc[train_idx,1:], y_train.iloc[train_idx] 
            xtrain = ops.scaling_data(xtrain, scaling_mode=args.data_scaling, scaler=trained_scaler)
            xtrain = pd.DataFrame(xtrain, columns=X_train.iloc[train_idx,1:].columns)
            print(f"      Refit model {idx+1}      ")
            est.fit(xtrain, ytrain)
            
            ytrain = ytrain.to_frame()
            ytrain['PRED_DX'] = est.predict_proba(pd.DataFrame(xtrain, columns=X_train.iloc[train_idx,1:].columns))[:,1]
            ytrain['ID'] = X_train['ID'].values[train_idx]
            ytrain = ytrain[["ID","DX","PRED_DX"]]
            train_df.append({'data':X_train.iloc[train_idx,:], 'y': ytrain})
            
            test_hr = hisp_race_.iloc[test_idx,:]

            NHW_idx = test_hr.index[(test_hr['HISPANIC']==0) & (test_hr['RACE']==1)].tolist()
            NHA_idx = test_hr.index[(test_hr['HISPANIC']==0) & (test_hr['RACE']==2)].tolist()
            HWA_idx = test_hr.index[(test_hr['HISPANIC']==1) & (test_hr['RACE']==1)].tolist()

            X_NHW = X_train.iloc[NHW_idx,:]
            y_NHW = y_train.iloc[NHW_idx]
            X_NHA = X_train.iloc[NHA_idx,:]
            y_NHA = y_train.iloc[NHA_idx]
            X_HWA = X_train.iloc[HWA_idx,:]
            y_HWA = y_train.iloc[HWA_idx]

            print(f"     Estimate prediction of {idx+1}      ")
            pred_NHW = est.predict_proba(pd.DataFrame(ops.scaling_data(X_NHW.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHW.iloc[:,1:].columns))[:,1]
            pred_NHA = est.predict_proba(pd.DataFrame(ops.scaling_data(X_NHA.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHA.iloc[:,1:].columns))[:,1]
            pred_HWA = est.predict_proba(pd.DataFrame(ops.scaling_data(X_HWA.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_HWA.iloc[:,1:].columns))[:,1]
            
            y_NHW = y_NHW.to_frame()
            y_NHA = y_NHA.to_frame()
            y_HWA = y_HWA.to_frame()
            
            y_NHW['PRED_DX'] = pred_NHW
            y_NHA['PRED_DX'] = pred_NHA
            y_HWA['PRED_DX'] = pred_HWA
            
            y_NHW['ID'] = X_NHW['ID'].values
            y_NHA['ID'] = X_NHA['ID'].values
            y_HWA['ID'] = X_HWA['ID'].values
            
            y_NHW = y_NHW[["ID","DX","PRED_DX"]]
            y_NHA = y_NHA[["ID","DX","PRED_DX"]]
            y_HWA = y_HWA[["ID","DX","PRED_DX"]]
            
            NHW_df.append({'data':X_NHW, 'y': y_NHW})
            NHA_df.append({'data':X_NHA, 'y': y_NHA})
            HWA_df.append({'data':X_HWA, 'y': y_HWA})
            
            estimator.append(est)
    else:
        train_df, NHW_df, NHA_df, HWA_df = [], [], [], []
        print("   Load trained model   ")
        trained_hyper_optimizer_1 = joblib.load(f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}_1.joblib')
        trained_hyper_optimizer_2 = joblib.load(f'{args.save_model_path}/{args.model_name}_grid_search_{args.data_scaling}_{classes}_2.joblib')
        for idx in range(args.n_splits):
            print(f"      Load model {idx+1}      ")
            train_idx_1 = trained_hyper_optimizer_1.split_indices['train'][idx]
            test_idx_1 = trained_hyper_optimizer_1.split_indices['test'][idx]
            fold_params_1 = trained_hyper_optimizer_1.all_best_parameters_[f'split{idx}_params']
            est_1 = clone(trained_hyper_optimizer_1.estimator).set_params(**fold_params_1)
            print(f"      Load best parameter 1: {fold_params_1}        ")
            
            train_idx_2 = trained_hyper_optimizer_2.split_indices['train'][idx]
            test_idx_2 = trained_hyper_optimizer_2.split_indices['test'][idx]
            fold_params_2 = trained_hyper_optimizer_2.all_best_parameters_[f'split{idx}_params']
            est_2 = clone(trained_hyper_optimizer_2.estimator).set_params(**fold_params_2)
            print(f"      Load best parameter 2: {fold_params_2}        ")

            print(f"      Refit model {idx+1}      ")
            xtrain_1, ytrain_1 = X_train.iloc[train_idx_1,1:], y_train.iloc[train_idx_1]
            xtrain_1 = ops.scaling_data(xtrain_1, scaling_mode=args.data_scaling, scaler=trained_scaler)
            xtrain_1 = pd.DataFrame(xtrain_1, columns=X_train.iloc[train_idx_1,1:].columns)
            if est_1.sample_weight is not None:
                est_1.fit(xtrain_1, ytrain_1, sample_weight=est_1.sample_weight[train_idx_1])
            else:
                est_1.fit(xtrain_1, ytrain_1)
            
            xtrain_2, ytrain_2 = X_train.iloc[train_idx_2,1:], y_train.iloc[train_idx_2]
            xtrain_2 = ops.scaling_data(xtrain_2, scaling_mode=args.data_scaling, scaler=trained_scaler)
            xtrain_2 = pd.DataFrame(xtrain_2, columns=X_train.iloc[train_idx_2,1:].columns)
            if est_2.sample_weight is not None:
                est_2.fit(xtrain_2, ytrain_2, sample_weight=est_2.sample_weight[train_idx_2])
            else:
                est_2.fit(xtrain_2, ytrain_2)
            
            ytrain_1 = ytrain_1.to_frame()
            ytrain_1['PRED_DX'] = est_1.predict_proba(pd.DataFrame(xtrain_1, columns=X_train.iloc[train_idx_1,1:].columns))[:,1]
            ytrain_1['ID'] = X_train['ID'].values[train_idx_1]
            ytrain_1 = ytrain_1[["ID","DX","PRED_DX"]]
            ytrain_2 = ytrain_2.to_frame()
            ytrain_2['PRED_DX'] = est_2.predict_proba(pd.DataFrame(xtrain_2, columns=X_train.iloc[train_idx_2,1:].columns))[:,1]
            ytrain_2['ID'] = X_train['ID'].values[train_idx_2]
            ytrain_2 = ytrain_2[["ID","DX","PRED_DX"]]
            train_df.append({'data_1':X_train.iloc[train_idx_1,:], 'y_1': ytrain_1, 'data_2':X_train.iloc[train_idx_2,:], 'y_2': ytrain_2})
            
            if args.train_type=="NHW": 
                HWA_idx = hisp_race_[2].index[(hisp_race_[2]['HISPANIC']==1) & (hisp_race_[2]['RACE']==1)].tolist()

                X_NHW_1 = X_train.iloc[test_idx_1,:].copy()
                y_NHW_1 = y_train.iloc[test_idx_1].copy()
                X_NHW_2 = X_train.iloc[test_idx_2,:].copy()
                y_NHW_2 = y_train.iloc[test_idx_2].copy()
                X_NHW = X_NHW_1.copy()
                y_NHW = y_NHW_1.copy()
                X_NHA = X_test[0].copy()
                y_NHA = y_test[0].copy()
                X_HWA = X_test[1].iloc[HWA_idx,:].copy()
                y_HWA = y_test[1].iloc[HWA_idx].copy()
                
                pred_NHW_0 = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_NHW_1.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHW_1.iloc[:,1:].columns))[:,1]
                pred_NHW_1 = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_NHW_2.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHW_2.iloc[:,1:].columns))[:,1]
                pred_NHW = (pred_NHW_0 + pred_NHW_1) / 2.
                pred_NHA = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_NHA.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHA.iloc[:,1:].columns))[:,1]
                pred_HWA = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_HWA.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_HWA.iloc[:,1:].columns))[:,1]
                
                
            elif args.train_type=="NHA":
                HWA_idx = hisp_race_[2].index[(hisp_race_[2]['HISPANIC']==1) & (hisp_race_[2]['RACE']==1)].tolist()
                
                X_NHW = X_test[0].copy()
                y_NHW = y_test[0].copy()
                X_NHA_1 = X_train.iloc[test_idx_1,:].copy()
                y_NHA_1 = y_train.iloc[test_idx_1].copy()
                X_NHA_2 = X_train.iloc[test_idx_2,:].copy()
                y_NHA_2 = y_train.iloc[test_idx_2].copy()
                X_NHA = X_NHA_1.copy()
                y_NHA = y_NHA_1.copy()
                X_HWA = X_test[1].iloc[HWA_idx,:].copy()
                y_HWA = y_test[1].iloc[HWA_idx].copy()
                
                pred_NHA_0 = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_NHA_1.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHA_1.iloc[:,1:].columns))[:,1]
                pred_NHA_1 = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_NHA_2.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHA_2.iloc[:,1:].columns))[:,1]
                pred_NHA = (pred_NHA_0 + pred_NHA_1) / 2.
                pred_NHW = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_NHW.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHW.iloc[:,1:].columns))[:,1]
                pred_HWA = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_HWA.iloc[:,1:], scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_HWA.iloc[:,1:].columns))[:,1]
                
            elif args.train_type=="Hispanic":
                HWA_idx_1 = hisp_race_[0].iloc[test_idx_1,:].index[(hisp_race_[0].iloc[test_idx_1,:]['HISPANIC']==1) & (hisp_race_[0].iloc[test_idx_1,:]['RACE']==1)].tolist()
                HWA_idx_2 = hisp_race_[0].iloc[test_idx_2,:].index[(hisp_race_[0].iloc[test_idx_2,:]['HISPANIC']==1) & (hisp_race_[0].iloc[test_idx_2,:]['RACE']==1)].tolist()
                
                X_NHW = X_test[0].copy()
                y_NHW = y_test[0].copy()
                X_NHA = X_test[1].copy()
                y_NHA = y_test[1].copy()
                X_HWA_1 = X_train.iloc[HWA_idx_1,:].copy()
                y_HWA_1 = y_train.iloc[HWA_idx_1].copy()
                X_HWA_2 = X_train.iloc[HWA_idx_2,:].copy()
                y_HWA_2 = y_train.iloc[HWA_idx_2].copy()
                X_HWA = X_HWA_1.copy()
                y_HWA = y_HWA_1.copy()
                
                pred_HWA_0 = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_HWA_1.iloc[:,1:], None, scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_HWA_1.iloc[:,1:].columns))[:,1]
                pred_HWA_1 = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_HWA_2.iloc[:,1:], None, scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_HWA_2.iloc[:,1:].columns))[:,1]
                pred_HWA = (pred_HWA_0 + pred_HWA_1) / 2.
                pred_NHW = est_1.predict_proba(pd.DataFrame(ops.scaling_data(X_NHW.iloc[:,1:], None, scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHW.iloc[:,1:].columns))[:,1]
                pred_NHA = est_2.predict_proba(pd.DataFrame(ops.scaling_data(X_NHA.iloc[:,1:], None, scaling_mode=args.data_scaling, scaler=trained_scaler), columns=X_NHA.iloc[:,1:].columns))[:,1]
            
            y_NHW = y_NHW.to_frame()
            y_NHA = y_NHA.to_frame()
            y_HWA = y_HWA.to_frame()
            
            print(f"     Estimate prediction of {idx+1}      ")
            y_NHW['PRED_DX'] = pred_NHW
            y_NHA['PRED_DX'] = pred_NHA
            y_HWA['PRED_DX'] = pred_HWA
            
            y_NHW['ID'] = X_NHW['ID'].values
            y_NHA['ID'] = X_NHA['ID'].values
            y_HWA['ID'] = X_HWA['ID'].values
            
            y_NHW = y_NHW[["ID","DX","PRED_DX"]]
            y_NHA = y_NHA[["ID","DX","PRED_DX"]]
            y_HWA = y_HWA[["ID","DX","PRED_DX"]]
            
            NHW_df.append({'data':X_NHW, 'y': y_NHW})
            NHA_df.append({'data':X_NHA, 'y': y_NHA})
            HWA_df.append({'data':X_HWA, 'y': y_HWA})
            
            estimator.append((est_1, est_2))

    for i in range(args.n_splits):
        print(f"======================= FOLD {i+1} ==============================")
        print("   NHW balanced accuracy :  ", balanced_accuracy_score(NHW_df[i]["y"]["DX"].values, np.array(NHW_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
        print("   NHA balanced accuracy :  ", balanced_accuracy_score(NHA_df[i]["y"]["DX"].values, np.array(NHA_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
        print("   HWA balanced accuracy :  ", balanced_accuracy_score(HWA_df[i]["y"]["DX"].values, np.array(HWA_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
        print("   +  NHW F1 score :  ", f1_score(NHW_df[i]["y"]["DX"].values, np.array(NHW_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
        print("   +  NHA F1 score :  ", f1_score(NHA_df[i]["y"]["DX"].values, np.array(NHA_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
        print("   +  HWA F1 score :  ", f1_score(HWA_df[i]["y"]["DX"].values, np.array(HWA_df[i]["y"]["PRED_DX"].values>0.5, dtype=np.int32)))
    
    if not os.path.exists(args.save_result_path):
        os.makedirs(args.save_result_path, exist_ok=True)
        print(f"   Create result folder: {args.save_result_path}")

    with open(f'{args.save_result_path}/{classes}.pkl', 'wb') as f:
        print(f" Save prediction! Time : {time.time() - starttime}")
        pickle.dump([train_df, NHW_df, NHA_df, HWA_df, estimator], f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training AD Classification")
    # Data Arguments
    parser.add_argument("--save_model_path", type=str, default=r'/home/Codes/ad_classification/models/xgb_cr_new', help="Saved model path")
    parser.add_argument("--save_result_path", type=str, default=r'/home/Codes/ad_classification/results/xgb_cr_new', help="Saved result path")
    parser.add_argument("--df_roi", type=str, default=r'/home/Codes/ad_classification/data/filtered_data_09182024.csv', help="Path to CSV data file of MUSE ROI")
    #parser.add_argument("--df_roi", type=str, default=r'/home/Codes/ad_classification/data/filtered_data_09182024.csv', help="Path to CSV data file of MUSE ROI")
    #parser.add_argument("--df_dem", type=str, default=r'/henryho/ad_classification/data/mri_ID_w_diag.csv', help="Path to CSV data file of demographic")
    parser.add_argument("--train_type", type=str, default='All', help="Type of training dataset: All, NHW, NHA, or Hispanic")
    parser.add_argument("--train_drop_rate", type=int, default=0, help="Rate to drop training sample")
    parser.add_argument("--model_name", type=str, default='xgboost', help="Type of ML classifier: xgboost, svm")
    parser.add_argument("--sample_weight_method", type=str, default='kmm', help="Method for domain adaptation")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of splitting folds for cross validation")
    #parser.add_argument("--scale_pos", type=float, default=1.0, help="Scale ratio of negative and positive samples")
    parser.add_argument("--data_scaling", type=int, default=0, help="Normalize data")
    parser.add_argument("--corr_remove", type=int, default=0, help="Remove feature correlation")
    parser.add_argument("--corr_alpha", type=float, default=0.05, help="Correlation factor")
    parser.add_argument("--harmonize", type=int, default=0, help="whether using harmonization")
    parser.add_argument("--sim_threshold", type=float, default=1., help="similarity threshold")
    
    args = parser.parse_args()
    main(args)
    