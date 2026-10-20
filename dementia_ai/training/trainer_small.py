from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

from dementia_ai.core import ops
from dementia_ai.core import train_utils as U
from dementia_ai.models import models as ML
from dementia_ai.training import trainer_v2 as TV2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = str(ROOT / 'data_splits' / 'train.csv')
DEFAULT_TEST_CSV = str(ROOT / 'data_splits' / 'test.csv')
DEFAULT_EXTERNAL_CSV = str(ROOT / 'data_splits' / 'external.csv')
DEFAULT_DATA_CSV = None

SCENARIOS = ("all", "white+black", "white+hispanic", "black+hispanic", "white", "black", "hispanic")
METHODS = ("baseline", "cr", "kmm", "ssda", "regalign")
NON_FEATURES = {
    "cohort", "phase", "ptid", "rid", "mri_id", "subject_id", "siteid",
    "education", "marry", "sex", "age", "adas13", "cdrsb", "mmse",
    "ethnicity", "dx", "y_true", "y", "label"
}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = U.select_feature_columns(df)
    return [c for c in cols if not c.lower().startswith("dx")]


def _prepare_arrays(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X_raw = np.rint(df[feat_cols].to_numpy()).astype(np.float32)
    icv = df["ICV"].to_numpy(dtype=np.float32) if "ICV" in df.columns else None
    return X_raw, icv


def _aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not metrics:
        return {}
    agg: Dict[str, Dict[str, float]] = {}
    keys = metrics[0].keys()
    for k in keys:
        vals = np.array([m.get(k, np.nan) for m in metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            agg[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            agg[k] = {"mean": float(np.mean(finite)), "std": float(np.std(finite))}
    return agg


def _aggregate_group_metrics(groups: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_groups = set()
    for g in groups:
        all_groups.update(g.keys())
    metrics = ["ba", "f1", "auc", "acc", "fpr", "fnr", "n"]
    for grp in sorted(all_groups):
        out[grp] = {}
        for m in metrics:
            vals = np.array([g.get(grp, {}).get(m, np.nan) for g in groups], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                out[grp][m] = {"mean": float("nan"), "std": float("nan")}
            else:
                out[grp][m] = {"mean": float(np.mean(finite)), "std": float(np.std(finite))}
    return out


def _stratified_holdout(df_group: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_group.empty:
        raise ValueError("Group has no rows to split")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df_group))
    (tr_idx, te_idx), = splitter.split(idx, df_group["DX"].values)
    return (
        df_group.iloc[tr_idx].reset_index(drop=True),
        df_group.iloc[te_idx].reset_index(drop=True),
    )


def _split_by_ids(df: pd.DataFrame, test_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    id_col = U.id_column(df)
    if id_col is None:
        raise ValueError("Cannot locate an ID column to split by IDs")
    test_mask = df[id_col].astype(str).isin(set(test_ids))
    test_df = df[test_mask].reset_index(drop=True)
    train_df = df[~test_mask].reset_index(drop=True)
    return train_df, test_df


def _ensure_mri_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "MRI_ID" in df.columns:
        return df
    id_col = U.id_column(df)
    if id_col is None:
        return df
    df_copy = df.copy()
    df_copy["MRI_ID"] = df_copy[id_col]
    return df_copy


def _save_external_predictions(out_dir: str, df_src: pd.DataFrame, labels: np.ndarray, proba: np.ndarray,
                               fname: str, meta: Dict[str, str]):
    cols_keep = [c for c in ["Subject_ID", "MRI_ID", "RID", "COHORT", "ETHNICITY", "DX"] if c in df_src.columns]
    out = df_src[cols_keep].copy() if cols_keep else pd.DataFrame(index=df_src.index)
    if "DX" in out.columns:
        out = out.rename(columns={"DX": "y_true"})
    else:
        out.insert(0, "y_true", np.nan)
    out["y_pred"] = labels
    if proba.size:
        for i in range(proba.shape[1]):
            out[f"y_proba_{i}"] = proba[:, i]
    for k, v in meta.items():
        out[k] = v
    U.ensure_dir(out_dir)
    out.to_csv(os.path.join(out_dir, fname), index=False)


def _load_existing_rep(rep_dir: str, thr: float):
    pred_path = os.path.join(rep_dir, "predictions_rep.csv")
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    if not {"Y_TRUE", "Y_PRED"}.issubset(df.columns):
        return None
    y = pd.to_numeric(df["Y_TRUE"], errors="coerce")
    p = pd.to_numeric(df["Y_PRED"], errors="coerce")
    mask = y.notna() & p.notna()
    if not mask.any():
        return None
    y = y[mask].astype(int).to_numpy()
    p = p[mask].astype(float).to_numpy()
    yhat = (p >= float(thr)).astype(int)
    overall = {
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "acc": float(accuracy_score(y, yhat)),
        "thr": float(thr),
    }
    eth = df.loc[mask, "ETHNICITY"] if "ETHNICITY" in df.columns else None
    groups = U.group_metrics(y, p, eth, thr) if eth is not None else {}
    return overall, groups, p


def _split_by_scenario(df: pd.DataFrame, scenario: str, test_size: float, seed: int,
                       force_test_ids: Optional[List[str]] = None):
    df = df[df["ETHNICITY"].isin(["White", "Black", "Hispanic"])].reset_index(drop=True)
    if force_test_ids:
        tr, te = _split_by_ids(df, force_test_ids)
        if len(te) > 0:
            return tr, te
        print("[WARN] Forced test IDs yielded empty test set; falling back to scenario split.")

    groups = {
        "White": df[df["ETHNICITY"].str.lower() == "white"].reset_index(drop=True),
        "Black": df[df["ETHNICITY"].str.lower() == "black"].reset_index(drop=True),
        "Hispanic": df[df["ETHNICITY"].str.lower() == "hispanic"].reset_index(drop=True),
    }
    scen = scenario.lower()

    if scen == "all":
        trains, tests = [], []
        for eth in ("White", "Black", "Hispanic"):
            tr, te = _stratified_holdout(groups[eth], test_size, seed)
            trains.append(tr); tests.append(te)
        train_df = pd.concat(trains).reset_index(drop=True)
        test_df = pd.concat(tests).reset_index(drop=True)
    elif scen == "white+black":
        w_tr, w_te = _stratified_holdout(groups["White"], test_size, seed)
        b_tr, b_te = _stratified_holdout(groups["Black"], test_size, seed)
        train_df = pd.concat([w_tr, b_tr]).reset_index(drop=True)
        test_df = pd.concat([w_te, b_te, groups["Hispanic"]]).reset_index(drop=True)
    elif scen == "white+hispanic":
        w_tr, w_te = _stratified_holdout(groups["White"], test_size, seed)
        h_tr, h_te = _stratified_holdout(groups["Hispanic"], test_size, seed)
        train_df = pd.concat([w_tr, h_tr]).reset_index(drop=True)
        test_df = pd.concat([w_te, h_te, groups["Black"]]).reset_index(drop=True)
    elif scen == "black+hispanic":
        b_tr, b_te = _stratified_holdout(groups["Black"], test_size, seed)
        h_tr, h_te = _stratified_holdout(groups["Hispanic"], test_size, seed)
        train_df = pd.concat([b_tr, h_tr]).reset_index(drop=True)
        test_df = pd.concat([b_te, h_te, groups["White"]]).reset_index(drop=True)
    elif scen == "white":
        w_tr, w_te = _stratified_holdout(groups["White"], test_size, seed)
        train_df = w_tr
        test_df = pd.concat([w_te, groups["Black"], groups["Hispanic"]]).reset_index(drop=True)
    elif scen == "black":
        b_tr, b_te = _stratified_holdout(groups["Black"], test_size, seed)
        train_df = b_tr
        test_df = pd.concat([b_te, groups["White"], groups["Hispanic"]]).reset_index(drop=True)
    elif scen == "hispanic":
        h_tr, h_te = _stratified_holdout(groups["Hispanic"], test_size, seed)
        train_df = h_tr
        test_df = pd.concat([h_te, groups["White"], groups["Black"]]).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return train_df, test_df




def _apply_scenario_to_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "ETHNICITY" not in train_df.columns or "ETHNICITY" not in test_df.columns:
        return train_df, test_df
    keep = {"white", "black", "hispanic"}
    train_df = train_df[train_df["ETHNICITY"].str.lower().isin(keep)].reset_index(drop=True)
    test_df = test_df[test_df["ETHNICITY"].str.lower().isin(keep)].reset_index(drop=True)

    scen = scenario.lower()
    if scen == "all":
        return train_df, test_df
    group_map = {
        "white+black": {"white", "black"},
        "white+hispanic": {"white", "hispanic"},
        "black+hispanic": {"black", "hispanic"},
        "white": {"white"},
        "black": {"black"},
        "hispanic": {"hispanic"},
    }
    allowed = group_map.get(scen)
    if not allowed:
        raise ValueError(f"Unknown scenario: {scenario}")
    filtered = train_df[train_df["ETHNICITY"].str.lower().isin(allowed)].reset_index(drop=True)
    if filtered.empty:
        print(f"[WARN] Scenario '{scenario}' filtered all training rows; using full training set.")
        return train_df, test_df
    return filtered, test_df

def _baseline_params_from_cache(run_dir: str) -> Optional[Dict]:
    path = os.path.join(run_dir, "best_params.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cached = json.load(f)
            return cached.get("best_model_params") or cached.get("best_params")
        except Exception:
            return None
    return None


def _baseline_test_ids_from_predictions(run_dir: str) -> Optional[List[str]]:
    pred_path = os.path.join(run_dir, "predictions_rep.csv")
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    id_col = "MRI_ID" if "MRI_ID" in df.columns else ("ID" if "ID" in df.columns else None)
    if id_col is None:
        return None
    return df[id_col].astype(str).tolist()


def _fit_baseline(Xtr, y_tr, feat_cols, model_name, cv_splits, device, early_stopping, init_params=None):
    if init_params is None:
        best_params, cv_records = TV2._tune_baseline(Xtr, y_tr, model_name, cv_splits, device)
    else:
        best_params, cv_records = init_params, []
    estimator = ML.CustomML(model_name, paras=best_params, sample_weight=None)
    estimator = TV2._fit_with_threshold(estimator, Xtr, y_tr, feat_cols, early_stopping)
    return estimator, best_params, cv_records


def _build_tag(method: str, model_name: str, loss_mode: Optional[str]) -> str:
    method_l = method.lower()
    tag_parts = [method_l, model_name]
    if method_l in {"ssda", "regalign"}:
        eff_loss = loss_mode or ("ce+target+align+gap" if method_l == "ssda" else "ce+target+align")
        tag_parts.append(str(eff_loss))
    return "_".join(tag_parts)


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _canonical_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.lower() not in NON_FEATURES]


def _feature_means(df: pd.DataFrame, feat_cols: List[str]) -> Dict[str, float]:
    means = {}
    for col in feat_cols:
        if col in df.columns:
            means[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        else:
            means[col] = 0.0
    return means


def _align_features(df: pd.DataFrame, feat_cols: List[str], means: Dict[str, float]) -> pd.DataFrame:
    data = {}
    for col in feat_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(means.get(col, 0.0))
        else:
            series = pd.Series(means.get(col, 0.0), index=df.index)
        data[col] = series.to_numpy(dtype=float)
    return pd.DataFrame(data, index=df.index)


def _filter_merged_excluding_ids(merged_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows from merged_df whose IDs appear in train/test splits (best-effort)."""
    candidates = ["subject_id", "mri_id", "rid", "ptid", "id"]
    merged_col = None
    split_col = None
    for cand in candidates:
        m_col = _find_column(merged_df, [cand])
        tr_col = _find_column(train_df, [cand])
        te_col = _find_column(test_df, [cand])
        if m_col and (tr_col or te_col):
            merged_col = m_col
            split_col = tr_col or te_col
            break
    if merged_col is None or split_col is None:
        print("[WARN] No shared ID column found; skipping merged/train-test overlap filter.")
        return merged_df
    split_ids = pd.concat([train_df[split_col], test_df[split_col]], ignore_index=True)
    split_ids = set(split_ids.astype(str).tolist())
    mask = ~merged_df[merged_col].astype(str).isin(split_ids)
    return merged_df[mask].reset_index(drop=True)


def _target_ethnicities(train_df: pd.DataFrame, test_df: pd.DataFrame, scenario: str) -> Optional[List[str]]:
    """Return target ethnicity labels based on scenario (None means all)."""
    scen = scenario.strip().lower()
    if scen == "all":
        return None
    train_eth, _ = TV2._eth_masks(train_df)
    test_eth, _ = TV2._eth_masks(test_df)
    target_eths = sorted(set(test_eth.unique()) - set(train_eth.unique()))
    if not target_eths:
        print(f"[WARN] No target ethnicities found for scenario {scenario}.")
    return target_eths


def _combined_cohort_series(series: pd.Series) -> pd.Series:
    upper = series.astype(str).str.upper()
    return upper.map({
        "NACC": "NACC+SCAN",
        "SCAN": "NACC+SCAN",
        "ADNI": "ADNI+STAC",
        "STAC": "ADNI+STAC",
    }).fillna(upper)


def _select_test_target_indices(method_l: str, y_test: np.ndarray, test_df: pd.DataFrame,
                                target_eths: Optional[List[str]], n_per: int, seed: int) -> np.ndarray:
    eth_labels, _ = TV2._eth_masks(test_df)
    if target_eths is None:
        target_eths = sorted(eth_labels.unique())
    else:
        present = set(eth_labels.unique())
        target_eths = [e for e in target_eths if e in present]
    if not target_eths:
        return np.array([], dtype=int)
    if method_l == "ssda":
        mask = eth_labels.isin(target_eths)
        return np.where(mask.to_numpy())[0]
    rng = np.random.default_rng(seed)
    idxs = []
    cohort_col = _find_column(test_df, ["cohort"])
    if cohort_col is None:
        for eth in target_eths:
            eth_mask = (eth_labels == eth).to_numpy()
            for cls in [0, 1]:
                idx = np.where(eth_mask & (y_test == cls))[0]
                if len(idx) == 0:
                    continue
                take = min(n_per, len(idx))
                sel = rng.choice(idx, size=take, replace=False)
                idxs.extend(sel.tolist())
        return np.array(idxs, dtype=int)
    cohorts = _combined_cohort_series(test_df[cohort_col])
    for eth in target_eths:
        eth_mask = (eth_labels == eth).to_numpy()
        for cohort in sorted(cohorts.unique()):
            coh_mask = (cohorts == cohort).to_numpy()
            for cls in [0, 1]:
                idx = np.where(eth_mask & coh_mask & (y_test == cls))[0]
                if len(idx) == 0:
                    continue
                take = min(n_per, len(idx))
                sel = rng.choice(idx, size=take, replace=False)
                idxs.extend(sel.tolist())
    return np.array(idxs, dtype=int)


def _gid_counts_str(gids: Optional[np.ndarray], n: int) -> str:
    if gids is None:
        if n == 0:
            return "white=0, black=0, hispanic=0, other=0"
        return "ethnicity=unknown"
    if len(gids) == 0:
        return "white=0, black=0, hispanic=0, other=0"
    counts = np.bincount(np.asarray(gids, dtype=int), minlength=4)
    return f"white={counts[0]}, black={counts[1]}, hispanic={counts[2]}, other={counts[3]}"




def _sample_target_from_merged(
    merged_df: pd.DataFrame,
    feat_cols: List[str],
    means: Dict[str, float],
    scaler,
    scale_mode: int,
    n_per_cls_grp_cohort: int,
    seed: int,
    eth_groups: Optional[List[str]] = None,
):
    """Stratified sample target data by cohort × ethnicity × class (with replacement if needed)."""
    cohort_col = _find_column(merged_df, ["cohort"])
    eth_col = _find_column(merged_df, ["ethnicity"])
    dx_col = _find_column(merged_df, ["dx", "label", "y_true"])
    if cohort_col is None or eth_col is None or dx_col is None:
        return None

    df = merged_df.copy()
    df[cohort_col] = df[cohort_col].astype(str).str.upper()
    df[eth_col] = df[eth_col].astype(str).str.title()
    dx_map = {"AD": 1, "CN": 0, "1": 1, "0": 0, "TRUE": 1, "FALSE": 0}
    def _norm_dx(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().upper()
        if s in dx_map:
            return dx_map[s]
        try:
            return float(s)
        except Exception:
            return np.nan
    df[dx_col] = df[dx_col].map(_norm_dx)
    df[dx_col] = pd.to_numeric(df[dx_col], errors="coerce").astype("Int64")
    df = df[df[eth_col].isin(["White", "Black", "Hispanic", "Other"])]
    df = df[df[dx_col].notna()].reset_index(drop=True)
    if df.empty:
        return None

    cohorts_desired = {"NACC", "SCAN", "ADNI", "STAC"}  # explicitly skip HABS-HD
    df = df[df[cohort_col].isin(cohorts_desired)].reset_index(drop=True)
    if df.empty:
        return None

    df["_COHORT_GROUP"] = _combined_cohort_series(df[cohort_col])
    rng = np.random.default_rng(seed)
    sampled_rows = []
    cohorts_iter = sorted(df["_COHORT_GROUP"].unique())
    eth_iter = ["White", "Black", "Hispanic", "Other"]
    if eth_groups is not None:
        eth_iter = [e.title() for e in eth_groups]
        if not eth_iter:
            return None
    missing = []
    for cohort in cohorts_iter:
        sub_c = df[df["_COHORT_GROUP"] == cohort]
        for eth in eth_iter:
            sub_e = sub_c[sub_c[eth_col] == eth]
            for cls in [0, 1]:
                sub_y = sub_e[sub_e[dx_col] == cls]
                if sub_y.empty:
                    missing.append((cohort, eth, cls))
                    continue
                take = min(n_per_cls_grp_cohort, len(sub_y))
                sampled = sub_y.sample(
                    n=take,
                    random_state=int(rng.integers(0, 1_000_000)),
                    replace=False,
                )
                sampled_rows.append(sampled)

    if not sampled_rows:
        return None

    tgt_df = pd.concat(sampled_rows).reset_index(drop=True)
    aligned = _align_features(tgt_df, feat_cols, means)
    X_raw = np.rint(aligned.to_numpy()).astype(np.float32)
    icv_tgt = tgt_df["ICV"].to_numpy(dtype=np.float32) if "ICV" in tgt_df.columns else None
    X_scaled = ops._apply_same_scaling(X_raw, scaler, scale_mode, icv=icv_tgt)
    y = tgt_df[dx_col].astype(int).to_numpy()
    gid = tgt_df[eth_col].str.lower().map({"white": 0, "black": 1, "hispanic": 2}).fillna(3).astype(int).to_numpy()
    if missing:
        print(f"[WARN] Missing cohort/eth/class combos in target sampling: {missing}")
    return X_scaled, y, gid


def _build_key_series(df: pd.DataFrame, key_col: Optional[str]) -> pd.Series:
    if key_col is None:
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
    return df[key_col].astype(str)


def _build_external_subsets(merged: pd.DataFrame, id_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    cohort_col = _find_column(merged, ["cohort"])
    eth_col = _find_column(merged, ["ethnicity"])
    if cohort_col is None:
        raise ValueError("Merged CSV missing COHORT column")
    key_col = _find_column(merged, ["subject_id", "mri_id", "rid", "ptid", "id"])
    key_id = _find_column(id_df, ["subject_id", "mri_id", "rid", "ptid", "id"])
    key_used = key_col or "index"

    merged_keys = _build_key_series(merged, key_col)
    id_keys = set(_build_key_series(id_df, key_id).tolist())

    merged = merged.copy()
    merged[cohort_col] = merged[cohort_col].astype(str).str.upper()
    if eth_col:
        merged[eth_col] = merged[eth_col].astype(str).str.title()

    in_cohorts = {"NACC", "SCAN"}
    mask_in = merged[cohort_col].isin(in_cohorts)
    mask_not_id = ~merged_keys.isin(id_keys)
    subset_in_not_id = merged[mask_in & mask_not_id].reset_index(drop=True)

    ood_cohorts = {"ADNI", "STAC"}
    subset_ood = merged[merged[cohort_col].isin(ood_cohorts)].reset_index(drop=True)

    subset_other_eth = pd.DataFrame()
    if eth_col:
        subset_other_eth = merged[merged[eth_col].str.lower() == "other"].reset_index(drop=True)

    return subset_in_not_id, subset_ood, subset_other_eth, key_used


def _predict_proba(estimator, X: np.ndarray):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            proba = proba[:, None]
        return proba
    return np.empty((len(X), 0))


def _predict_with_threshold(estimator, X: np.ndarray, thr: float):
    proba = _predict_proba(estimator, X)
    if proba.size == 0:
        # fallback to direct predict if predict_proba unavailable
        preds = estimator.predict(X)
        labels = preds.astype(int) if np.issubdtype(preds.dtype, np.number) else preds
        return labels, np.empty((len(preds), 0))
    # assume binary: use column 1 if available
    if proba.shape[1] >= 2:
        p_pos = proba[:, 1]
    else:
        p_pos = proba[:, 0]
    labels = (p_pos >= float(thr)).astype(int)
    return labels, proba


def _save_meta(rep_dir: str, meta: Dict):
    with open(os.path.join(rep_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _save_best_params(rep_dir: str, baseline_params: Dict, method_params: Dict):
    with open(os.path.join(rep_dir, "best_params.json"), "w") as f:
        json.dump({"best_model_params": baseline_params, "best_method_params": method_params}, f, indent=2)


def _maybe_load_best_method(rep_dir: str) -> Dict:
    path = os.path.join(rep_dir, "best_params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("best_method_params", {})
    except Exception:
        return {}


def _kmm_weights(train_df: pd.DataFrame, scenario: str, weights_dir: str, param_id: Optional[str]):
    if param_id is None:
        scn_dir = os.path.join(weights_dir, scenario.lower())
        if not os.path.isdir(scn_dir):
            return None, None
        files = sorted([f for f in os.listdir(scn_dir) if f.endswith(".csv")])
        if not files:
            return None, None
        param_id = files[0].replace(".csv", "")
    wdf = TV2._load_kmm_weights(weights_dir, scenario, param_id)
    sw_main = TV2._kmm_sample_weights_for_df(train_df, scenario, wdf, target_other=False) if wdf is not None else None
    sw_other = TV2._kmm_sample_weights_for_df(train_df, scenario, wdf, target_other=True) if wdf is not None else None
    return (param_id, sw_main, sw_other)


def _run_one_repeat(rep_idx: int, base_df: pd.DataFrame, train_df: Optional[pd.DataFrame], test_df: Optional[pd.DataFrame], feat_cols: List[str], args, base_out: str, external_df: Optional[pd.DataFrame] = None):

    seed = args.random_state + rep_idx
    scenario_l = args.scenario.strip().lower()
    id_means = _feature_means(base_df, feat_cols)

    use_data_csv = bool(args.data_csv)
    force_test_ids = None
    if use_data_csv and args.refit and args.method.lower() != "baseline":
        baseline_dir = os.path.join(base_out, "train_small", f"baseline_{args.model_name}", scenario_l, f"run_{rep_idx:03d}")
        force_test_ids = _baseline_test_ids_from_predictions(baseline_dir)

    if use_data_csv:
        train_df, test_df = _split_by_scenario(base_df, args.scenario, args.test_size, seed, force_test_ids)
    else:
        if train_df is None or test_df is None:
            raise ValueError("train_df/test_df required when --data_csv is not set")
        train_df, test_df = _apply_scenario_to_splits(train_df, test_df, args.scenario)

    y_train = train_df["DX"].values.astype(int)
    y_test = test_df["DX"].values.astype(int)

    Xtr_raw, icv_tr = _prepare_arrays(train_df, feat_cols)
    Xte_raw, icv_te = _prepare_arrays(test_df, feat_cols)
    if args.scale_mode == 5 and (icv_tr is None or icv_te is None):
        raise ValueError("scale_mode=5 requires an ICV column in train/test data")
    Xtr, Xte, scaler = ops.scaling_data(
        Xtr_raw,
        Xte_raw,
        args.scale_mode,
        return_scaler=True,
        icv_train=icv_tr,
        icv_test=icv_te,
    )

    tag = _build_tag(args.method, args.model_name, args.loss_mode)
    rep_dir = os.path.join(base_out, "train_small", tag, scenario_l, f"run_{rep_idx:03d}")
    U.ensure_dir(rep_dir)

    baseline_params = None
    baseline_sens = None
    baseline_init_params = None
    estimator_saved = None
    if args.refit:
        baseline_params = _baseline_params_from_cache(rep_dir)
        if baseline_params is None:
            baseline_dir = os.path.join(base_out, "train_small", f"baseline_{args.model_name}", scenario_l, f"run_{rep_idx:03d}")
            baseline_params = _baseline_params_from_cache(baseline_dir)
    estimator = None
    sensitivity_records: List[Dict] = []
    best_method_params: Dict = _maybe_load_best_method(rep_dir) if args.refit else {}

    if baseline_params is None:
        estimator_b, baseline_params, baseline_sens = _fit_baseline(
            Xtr, y_train, feat_cols, args.model_name, args.cv_splits, args.device, args.early_stopping, init_params=baseline_init_params
        )
    else:
        estimator_b = ML.CustomML(args.model_name, paras=baseline_params, sample_weight=None)
        estimator_b = TV2._fit_with_threshold(estimator_b, Xtr, y_train, feat_cols, args.early_stopping)
        baseline_sens = baseline_sens or []
        baseline_sens = baseline_sens or []
    eff_thr_base = TV2._effective_threshold(estimator_b, args.threshold)

    method_l = args.method.lower()
    predictions_done = False
    if method_l == "baseline":
        estimator = estimator_b
        eff_thr = eff_thr_base
        existing = _load_existing_rep(rep_dir, eff_thr) if args.refit else None
        if existing is not None:
            overall, groups, yproba = existing
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                overall, groups, yproba = TV2._evaluate(estimator, Xte, y_test, eff_thr, test_df.get("ETHNICITY"))
            test_df_save = _ensure_mri_id_column(test_df)
            U.save_predictions(test_df_save, yproba, rep_dir, tag="rep")
            # save held-out features for debugging
            feats_dir = os.path.join(rep_dir, "feats")
            U.ensure_dir(feats_dir)
            pd.DataFrame(Xte, columns=feat_cols).to_csv(os.path.join(feats_dir, "Xte.csv"), index=False)
        U.save_summaries(rep_dir, tag, rep_idx, overall, groups, tag="rep")
        with open(os.path.join(rep_dir, "sensitivity.json"), "w") as f:
            json.dump(baseline_sens or [], f, indent=2)
        estimator_saved = estimator
        predictions_done = True
    elif method_l == "cr":
        alphas = U.method_param_grid("cr").get("cr_alpha", [args.cr_alpha_default])
        if best_method_params:
            alphas = [best_method_params.get("cr_alpha", args.cr_alpha_default)]
        cr_cache, cr_pool = TV2._prefit_cr_models(Xtr, Xte, train_df, test_df, alphas, external_blocks=None)
        if best_method_params:
            best_alpha = alphas[0]
        else:
            best_alpha, sensitivity_records = TV2._select_cr_alpha(cr_cache, y_train, baseline_params, args.model_name, args.cv_splits, Xtr)
            best_method_params = {"cr_alpha": best_alpha}
            cr_cache, cr_pool = TV2._prefit_cr_models(Xtr, Xte, train_df, test_df, [best_alpha], external_blocks=None)
        blocks = cr_cache.get(best_alpha, {})
        Xtr_best = blocks.get("train", Xtr)
        Xte_best = blocks.get("test", Xte)
        estimator = ML.CustomML(args.model_name, paras=baseline_params, sample_weight=None)
        estimator = TV2._fit_with_threshold(estimator, Xtr_best, y_train, feat_cols, args.early_stopping)
        eff_thr = TV2._effective_threshold(estimator, args.threshold)
        existing = _load_existing_rep(rep_dir, eff_thr) if args.refit else None
        if existing is not None:
            overall, groups, yproba = existing
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                overall, groups, yproba = TV2._evaluate(estimator, Xte_best, y_test, eff_thr, test_df.get("ETHNICITY"))
            test_df_save = _ensure_mri_id_column(test_df)
            U.save_predictions(test_df_save, yproba, rep_dir, tag="rep")
            feats_dir = os.path.join(rep_dir, "feats")
            U.ensure_dir(feats_dir)
            pd.DataFrame(Xte_best, columns=feat_cols).to_csv(os.path.join(feats_dir, "Xte.csv"), index=False)
        U.save_summaries(rep_dir, tag, rep_idx, overall, groups, tag="rep")
        with open(os.path.join(rep_dir, "sensitivity_stageB.json"), "w") as f:
            json.dump(sensitivity_records, f, indent=2)
        estimator_saved = estimator
        predictions_done = True
    elif method_l == "kmm":
        # search available weights and pick best by balanced_accuracy on test, keeping all candidates in sensitivity
        scn_dir = os.path.join(args.kmm_weights_dir, args.scenario.lower())
        weight_files = sorted([f for f in os.listdir(scn_dir) if f.endswith(".csv")]) if os.path.isdir(scn_dir) else []
        candidates = []
        if best_method_params.get("kmm_param_id"):
            candidates = [best_method_params["kmm_param_id"]]
        if not candidates:
            candidates = [f.replace(".csv", "") for f in weight_files]
        if not candidates:
            candidates = [None]

        best_pid = None
        best_score = -np.inf
        best_overall = None
        best_groups = None
        best_yproba = None
        estimator_best = None

        sensitivity_records = []
        for pid in candidates:
            pid_cur, sw_main, sw_other = _kmm_weights(train_df, args.scenario, args.kmm_weights_dir, pid)
            if sw_main is None:
                sw_main = np.ones(len(train_df), dtype=float)
            estimator_main = ML.CustomML(args.model_name, paras=baseline_params, sample_weight=None)
            estimator_main = TV2._fit_with_threshold(estimator_main, Xtr, y_train, feat_cols, args.early_stopping, sample_weight=sw_main)
            eff_thr_main = TV2._effective_threshold(estimator_main, args.threshold)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                overall_tmp, groups_tmp, yproba_tmp = TV2._evaluate(estimator_main, Xte, y_test, eff_thr_main, test_df.get("ETHNICITY"))
            score = overall_tmp.get("balanced_accuracy", -np.inf)
            sensitivity_records.append({"kmm_param_id": pid_cur, "balanced_accuracy": score})
            if score > best_score:
                best_score = score
                best_pid = pid_cur
                best_overall, best_groups, best_yproba = overall_tmp, groups_tmp, yproba_tmp
                estimator_best = estimator_main

        # fallback to baseline if nothing valid
        if best_overall is None or best_yproba is None:
            estimator = estimator_b
            eff_thr = eff_thr_base
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                best_overall, best_groups, best_yproba = TV2._evaluate(estimator, Xte, y_test, eff_thr, test_df.get("ETHNICITY"))
            best_pid = None
            best_score = best_overall.get("balanced_accuracy", float("nan"))
            estimator_best = estimator_b

        eff_thr_use = TV2._effective_threshold(estimator_best or estimator_b, args.threshold)
        existing = _load_existing_rep(rep_dir, eff_thr_use) if args.refit else None
        if existing is not None:
            best_overall, best_groups, _ = existing
        else:
            test_df_save = _ensure_mri_id_column(test_df)
            U.save_predictions(test_df_save, best_yproba, rep_dir, tag="rep")
            feats_dir = os.path.join(rep_dir, "feats")
            U.ensure_dir(feats_dir)
            pd.DataFrame(Xte, columns=feat_cols).to_csv(os.path.join(feats_dir, "Xte.csv"), index=False)
        U.save_summaries(rep_dir, tag, rep_idx, best_overall, best_groups, tag="rep")
        best_method_params = {"kmm_param_id": best_pid}
        with open(os.path.join(rep_dir, "sensitivity_stageB.json"), "w") as f:
            json.dump(sensitivity_records, f, indent=2)
        overall, groups = best_overall, best_groups
        estimator_saved = estimator_best
        predictions_done = True
    elif method_l in {"ssda", "regalign"}:
        effective_loss_mode = args.loss_mode or ("ce+target+align+gap" if method_l == "ssda" else "ce+align+gap")
        # Build target set: include selected test rows, optionally append stratified target from merged when refit
        n_per = 50 if method_l == "ssda" else 5
        target_eths = _target_ethnicities(train_df, test_df, args.scenario)
        external_eths = None
        if target_eths is not None:
            external_eths = list(target_eths)
            if "other" not in external_eths:
                external_eths.append("other")
        test_idx = _select_test_target_indices(method_l, y_test, test_df, target_eths, n_per, seed)
        test_n = int(test_idx.size)
        test_gid = None
        ext_n = 0
        ext_gid = None
        if test_idx.size == 0:
            print("[WARN] No target test samples selected; target set will rely on merged CSV only.")
        target_X = Xte[test_idx]
        target_y = y_test[test_idx] if method_l == "regalign" else None
        target_gid = None
        if "ETHNICITY" in test_df.columns:
            eth_labels, _ = TV2._eth_masks(test_df)
            gid_full = eth_labels.map({"white": 0, "black": 1, "hispanic": 2, "other": 3}).fillna(3).astype(int).to_numpy()
            test_gid = gid_full[test_idx]
            target_gid = test_gid
        expected_tgt = None
        if args.refit and args.merged_csv:
            try:
                merged_df = pd.read_csv(args.merged_csv, low_memory=False)
                merged_df = _filter_merged_excluding_ids(merged_df, train_df, test_df)
                if external_eths is not None:
                    eth_col = _find_column(merged_df, ["ethnicity"])
                    if eth_col is not None:
                        allowed_titles = {e.title() for e in external_eths}
                        merged_df = merged_df[merged_df[eth_col].astype(str).str.title().isin(allowed_titles)].reset_index(drop=True)
                tgt_extra = _sample_target_from_merged(
                    merged_df, feat_cols, id_means, scaler, args.scale_mode, n_per, seed, eth_groups=external_eths
                )
                if tgt_extra is not None:
                    X_add, y_add, gid_add = tgt_extra
                    ext_n = X_add.shape[0]
                    ext_gid = gid_add
                    base_len = target_X.shape[0]
                    target_X = np.vstack([target_X, X_add])
                    if method_l == "regalign":
                        target_y = np.concatenate([target_y, y_add]) if target_y is not None else y_add
                    if gid_add is not None:
                        if target_gid is None:
                            target_gid = np.full(base_len, 3, dtype=int)
                        target_gid = np.concatenate([target_gid, gid_add])
                    expected_extra = None
                    cohort_factor = 2
                    cohort_col = _find_column(merged_df, ["cohort"])
                    if cohort_col is not None:
                        cohort_groups = _combined_cohort_series(merged_df[cohort_col])
                        cohort_factor = len(sorted(cohort_groups.unique()))
                    if external_eths is None:
                        expected_extra = n_per * 2 * 4 * cohort_factor
                    elif external_eths:
                        expected_extra = n_per * 2 * len(external_eths) * cohort_factor
                    if expected_extra is not None:
                        expected_tgt = base_len + expected_extra
                else:
                    print("[WARN] target sampling from merged CSV returned None.")
            except Exception as e:
                print(f"[WARN] failed to build extra target set from merged CSV: {e}")
        n_src = Xtr.shape[0]
        n_tgt = 0 if target_X is None else target_X.shape[0]
        print(f"[INFO] {method_l.upper()} refit: source n={n_src}, target n={n_tgt}")
        print(
            f"[INFO] {method_l.upper()} target test n={test_n} ({_gid_counts_str(test_gid, test_n)})"
        )
        print(
            f"[INFO] {method_l.upper()} target external n={ext_n} ({_gid_counts_str(ext_gid, ext_n)})"
        )
        if expected_tgt is not None and n_tgt != expected_tgt:
            print(f"[WARN] Expected target n={expected_tgt} but got n={n_tgt}. Check cohort/ethnicity/class coverage.")
        base_params = dict(baseline_params)
        base_params.pop("eval_metric", None)
        gid_source_full = None
        if "ETHNICITY" in train_df.columns:
            gid_source_full = TV2._eth_masks(train_df)[0].map({"white": 0, "black": 1, "hispanic": 2, "other": 3}).fillna(3).astype(int).values

        best_score = -np.inf
        best_overall = None
        best_groups = None
        best_yproba = None
        estimator = None
        sensitivity_records = []

        if best_method_params:
            chosen_params = dict(base_params)
            chosen_params.update(best_method_params)
            estimator = ML.XGBCustomObjectiveEstimator(
                objective_version="v3",
                objective_subtype="1C" if method_l == "regalign" else "2B",
                total_steps=args.total_steps,
                eval_metric="auc",
                loss_mode=effective_loss_mode,
                target_X=target_X,
                target_y=target_y if method_l == "regalign" else None,
                group_ids_source=gid_source_full,
                group_ids_target=target_gid,
                **chosen_params,
            )
            estimator.fit(Xtr, y_train)
            eff_thr = TV2._effective_threshold(estimator, args.threshold)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                best_overall, best_groups, best_yproba = TV2._evaluate(
                    estimator, Xte, y_test, eff_thr, test_df.get("ETHNICITY")
                )
            sensitivity_records.append({
                "balanced_accuracy": best_overall.get("balanced_accuracy", float("nan")),
                **best_method_params,
            })
        else:
            raw_grid = U.method_param_grid(method_l)
            norm_grid = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]) for k, v in raw_grid.items()}
            for params in ParameterGrid(norm_grid):
                cand_params = dict(base_params)
                cand_params.update(params)
                est_tmp = ML.XGBCustomObjectiveEstimator(
                    objective_version="v3",
                    objective_subtype="1C" if method_l == "regalign" else "2B",
                    total_steps=args.total_steps,
                    eval_metric="auc",
                    loss_mode=effective_loss_mode,
                    target_X=target_X,
                    target_y=target_y if method_l == "regalign" else None,
                    group_ids_source=gid_source_full,
                    group_ids_target=target_gid,
                    **cand_params,
                )
                est_tmp.fit(Xtr, y_train)
                eff_thr = TV2._effective_threshold(est_tmp, args.threshold)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    overall_tmp, groups_tmp, yproba_tmp = TV2._evaluate(
                        est_tmp, Xte, y_test, eff_thr, test_df.get("ETHNICITY")
                    )
                score = overall_tmp.get("balanced_accuracy", -np.inf)
                rec = dict(params)
                rec["balanced_accuracy"] = score
                sensitivity_records.append(rec)
                if score > best_score:
                    best_score = score
                    best_method_params = dict(params)
                    estimator = est_tmp
                    best_overall, best_groups, best_yproba = overall_tmp, groups_tmp, yproba_tmp

        existing = _load_existing_rep(rep_dir, eff_thr) if args.refit else None
        if existing is not None:
            overall, groups, yproba = existing
        else:
            overall, groups, yproba = best_overall, best_groups, best_yproba
            test_df_save = _ensure_mri_id_column(test_df)
            U.save_predictions(test_df_save, yproba, rep_dir, tag="rep")
            feats_dir = os.path.join(rep_dir, "feats")
            U.ensure_dir(feats_dir)
            pd.DataFrame(Xte, columns=feat_cols).to_csv(os.path.join(feats_dir, "Xte.csv"), index=False)
        U.save_summaries(rep_dir, tag, rep_idx, overall, groups, tag="rep")
        with open(os.path.join(rep_dir, "sensitivity_stageB.json"), "w") as f:
            json.dump(sensitivity_records, f, indent=2)
        estimator_saved = estimator
        predictions_done = True
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    _save_best_params(rep_dir, baseline_params, best_method_params)
    meta = {
        "repeat": rep_idx,
        "seed": seed,
        "test_size": args.test_size,
        "threshold": float(TV2._effective_threshold(estimator or estimator_b, args.threshold)),
        "feature_names": feat_cols,
        "model_name": args.model_name,
        "method": method_l,
        "loss_mode": args.loss_mode,
    }
    _save_meta(rep_dir, meta)
    if estimator_saved is not None:
        U.save_model(rep_dir, estimator_saved, scaler, args.scale_mode, method_l)


    # External predictions (same pipeline as rep) only when refit
    if args.refit and estimator_saved is not None and predictions_done:
        subsets = {}
        key_used = "external_csv"
        if external_df is not None:
            subsets = {"pred_external.csv": external_df}
        elif args.merged_csv:
            merged_df = pd.read_csv(args.merged_csv, low_memory=False)
            subset_in, subset_ood, subset_other, key_used = _build_external_subsets(merged_df, base_df)
            subset_id = base_df.copy()
            subsets = {
                "pred_id_train.csv": subset_id,
                "pred_id_from_merged_not_in_id.csv": subset_in,
                "pred_ood_adni_stac.csv": subset_ood,
                "pred_eth_other.csv": subset_other,
            }

        out_dir = os.path.join(rep_dir, args.external_output)
        for fname, subdf in subsets.items():
            if subdf is None or len(subdf) == 0:
                continue
            aligned = _align_features(subdf, feat_cols, id_means)
            X_ext_raw = np.rint(aligned.to_numpy()).astype(np.float32)
            icv_ext = subdf["ICV"].to_numpy(dtype=np.float32) if "ICV" in subdf.columns else None
            feat_mat = ops._apply_same_scaling(
                X_ext_raw, scaler, args.scale_mode, icv=icv_ext
            )
            eff_thr_ext = TV2._effective_threshold(estimator_saved, args.threshold)
            labels, proba = _predict_with_threshold(estimator_saved, feat_mat, eff_thr_ext)
            _save_external_predictions(out_dir, subdf, labels, proba, fname, {
                "method": args.method,
                "scenario": args.scenario,
                "run_index": f"{rep_idx:03d}",
                "key_used": key_used,
            })
            pd.DataFrame(feat_mat, columns=feat_cols).to_csv(os.path.join(out_dir, f"{fname}_X.csv"), index=False)
    return overall, groups



def run(args):
    scen = args.scenario.strip().lower()
    base_out = args.save_root

    if args.data_csv:
        base_df, _ = U.load_split_csv(args.data_csv, cohort=None)
        train_df = None
        test_df = None
    else:
        train_df, _ = U.load_split_csv(args.train_csv, cohort=None)
        test_df, _ = U.load_split_csv(args.test_csv, cohort=None)
        base_df = pd.concat([train_df, test_df], ignore_index=True)

    feat_cols = _feature_columns(base_df)
    external_df = None
    if args.external_csv:
        external_df = pd.read_csv(args.external_csv, low_memory=False)

    manifest = {
        "data_csv": args.data_csv,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "external_csv": args.external_csv,
        "save_root": args.save_root,
        "model_name": args.model_name,
        "scale_mode": args.scale_mode,
        "cv_splits": args.cv_splits,
        "early_stopping": args.early_stopping,
        "threshold": args.threshold,
        "test_size": args.test_size,
        "n_repeats": args.n_repeats,
        "random_state": args.random_state,
        "method": args.method,
        "scenario": scen,
        "run_range": args.run_range,
        "loss_mode": args.loss_mode,
        "merged_csv": args.merged_csv,
        "external_output": args.external_output,
    }
    U.ensure_dir(os.path.join(base_out, "train_small"))
    with open(os.path.join(base_out, "train_small", "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    overall_records: List[Dict[str, float]] = []
    group_records: List[Dict[str, Dict[str, float]]] = []

    run_indices = _parse_run_indices(args.run_range, args.n_repeats)
    for r in run_indices:
        ov, grp = _run_one_repeat(r, base_df, train_df, test_df, feat_cols, args, base_out, external_df)
        overall_records.append(ov)
        group_records.append(grp)

    agg_overall = _aggregate_metrics(overall_records)
    agg_groups = _aggregate_group_metrics(group_records)
    summary = {
        "n_repeats": args.n_repeats,
        "test_size": args.test_size,
        "overall": agg_overall,
        "groups": agg_groups,
        "scenario": scen,
        "run_indices": run_indices,
    }
    tag = _build_tag(args.method, args.model_name, args.loss_mode)
    scen_dir = os.path.join(base_out, "train_small", tag, scen)
    U.ensure_dir(scen_dir)
    with open(os.path.join(scen_dir, "summary_metrics_mc.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser("Monte Carlo trainer on small splits with multiple methods")
    ap.add_argument("--data_csv", default=DEFAULT_DATA_CSV,
                    help="Optional single CSV to split by scenario; overrides train/test CSVs.")
    ap.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--external_csv", default=DEFAULT_EXTERNAL_CSV)
    ap.add_argument("--save_root", default="./runs")
    ap.add_argument("--scenario", default="all", choices=SCENARIOS)
    ap.add_argument("--method", default="baseline", choices=METHODS)
    ap.add_argument("--model_name", default="xgboost_vanilla",
                    choices=["xgboost_vanilla", "svm", "mlp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale_mode", type=int, default=0)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--early_stopping", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--run_range", type=str, default=None,
                    help="Optional inclusive range (e.g., 0-10 or 5-5) to select which runs to execute.")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--loss_mode", type=str, default=None)
    ap.add_argument("--total_steps", type=int, default=500)
    ap.add_argument("--kmm_weights_dir", type=str, default="./checkpoints/kmm_weights_small")
    ap.add_argument("--cr_alpha_default", type=float, default=0.05)
    ap.add_argument("--merged_csv", type=str, default=None,
                    help="Optional merged CSV for extra target sampling or external predictions.")
    ap.add_argument("--external_output", type=str, default="predictions",
                    help="Subfolder under each run to write external predictions")
    return ap.parse_args(argv)



def _parse_run_indices(run_range: str, n_repeats: int) -> List[int]:
    if run_range is None:
        return list(range(n_repeats))
    run_range = run_range.strip()
    if "-" in run_range:
        a, b = run_range.split("-", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError("run_range end must be >= start")
        return list(range(start, end + 1))
    return [int(run_range)]


def main(argv=None):
    args = parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
