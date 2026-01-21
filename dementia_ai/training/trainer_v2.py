from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid

from dementia_ai.core import ops
from dementia_ai.core import train_utils as U
from dementia_ai.models import models as ML


def _effective_threshold(est_or_params, default_thr: float):
    try:
        if hasattr(est_or_params, "best_threshold_"):
            return float(getattr(est_or_params, "best_threshold_", default_thr))
    except Exception:
        pass
    t_cost = U.cost_threshold_from_params(est_or_params)
    return float(t_cost) if t_cost is not None else float(default_thr)


def _best_threshold(y_true, p):
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, -np.inf
    for t in grid:
        yhat = (p >= t).astype(int)
        ba = balanced_accuracy_score(y_true, yhat)
        if ba > best_score:
            best_t, best_score = float(t), float(ba)
    return best_t


def _evaluate(est, X, y, thr, eth=None):
    p = est.predict_proba(X)[:, 1]
    yhat = (p >= thr).astype(int)
    overall = {
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "f1": float(f1_score(y, yhat)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "acc": float(accuracy_score(y, yhat)),
        "thr": float(thr),
    }
    groups = U.group_metrics(y, p, eth, thr) if eth is not None else {}
    return overall, groups, p


def _fit_with_threshold(estimator, X, y, feat_cols, early_stopping: int = 50, sample_weight=None):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    (tr_idx, va_idx), = sss.split(X, y)
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

    if hasattr(estimator, "_fit_with_es"):
        estimator._fit_with_es(
            X_tr, y_tr, X_val=X_va, y_val=y_va,
            early_stopping_rounds=early_stopping,
            sample_weight=sw_tr,
        )
        return estimator

    # fit without early stopping; pass sample_weight only if supported
    try:
        estimator.fit(X_tr, y_tr, sample_weight=sw_tr)
    except TypeError:
        estimator.fit(X_tr, y_tr)

    try:
        p_val = estimator.predict_proba(X_va)[:, 1]
        thr = _best_threshold(y_va, p_val)
        setattr(estimator, "best_threshold_", float(thr))
    except Exception:
        pass
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        estimator.fit(X, y)
    return estimator


def _load_fold_data(scenario: str, fold: int, splits_dir: str, scale_mode: int):
    scenario_l = scenario.lower()
    train_csv = os.path.join(splits_dir, f"train_{scenario_l}_fold_{fold}.csv")
    test_csv = os.path.join(splits_dir, f"test_{scenario_l}_fold_{fold}.csv")

    train_df, y_train = U.load_split_csv(train_csv, cohort=None)
    test_df, y_test = U.load_split_csv(test_csv, cohort=None)
    #train_df, test_df = U.maybe_harmonize(train_df, test_df)

    feat_cols = U.select_feature_columns(train_df)
    Xtr_raw = np.rint(train_df[feat_cols].to_numpy()).astype(np.float32)
    Xte_raw = np.rint(test_df[feat_cols].to_numpy()).astype(np.float32)
    icv_tr = train_df['ICV'].to_numpy(dtype=np.float32) if 'ICV' in train_df.columns else None
    icv_te = test_df['ICV'].to_numpy(dtype=np.float32) if 'ICV' in test_df.columns else None
    Xtr, Xte, scaler = ops.scaling_data(
        Xtr_raw, Xte_raw, scale_mode, return_scaler=True,
        icv_train=icv_tr, icv_test=icv_te,
    )
    return train_df, test_df, y_train, y_test, feat_cols, Xtr, Xte, scaler, Xtr_raw, icv_tr


def _load_external_blocks(external_csvs: Optional[List[str]], external_cohorts: Optional[List[str]],
                          feat_cols: List[str], scaler, scale_mode: int,
                          Xtr_raw: np.ndarray, icv_train: Optional[np.ndarray]):
    blocks = []
    if not external_csvs:
        return blocks
    for p in external_csvs:
        if not os.path.exists(p):
            continue
        df_e, y_e = U.load_split_csv(p, cohort="all")
        base_name = os.path.basename(p)
        name_no_ext = os.path.splitext(base_name)[0]
        is_other = "other" in name_no_ext.lower()
        if external_cohorts and (not is_other) and "COHORT" in df_e.columns:
            df_e = df_e[df_e["COHORT"].astype(str).isin(external_cohorts)].copy()
            y_e = df_e["DX"].values
        if not set(feat_cols).issubset(df_e.columns):
            continue
        Xe_raw = np.rint(df_e[feat_cols].to_numpy()).astype(np.float32)
        icv_ext = df_e['ICV'].to_numpy(dtype=np.float32) if (scale_mode == 5 and 'ICV' in df_e.columns) else None
        Xe = ops._apply_same_scaling(Xe_raw, scaler, scale_mode, icv=icv_ext)
        blocks.append((name_no_ext, df_e.reset_index(drop=True), y_e, Xe, is_other))
    return blocks


def _load_kmm_weights(weights_root: str, scenario: str, param_id: str) -> Optional[pd.DataFrame]:
    path = os.path.join(weights_root, scenario.lower(), f"{param_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _eth_masks(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, np.ndarray]]:
    if "ETHNICITY" in df.columns:
        eth_raw = pd.Series(df["ETHNICITY"])
    else:
        eth_raw = pd.Series(["other"] * len(df))
    mapped = eth_raw.astype(str).str.lower().map({
        "white": "white", "black": "black", "hispanic": "hispanic",
        "nhw": "white", "nha": "black", "hwa": "hispanic",
    }).fillna("other")
    masks = {
        "white": (mapped == "white").values,
        "black": (mapped == "black").values,
        "hispanic": (mapped == "hispanic").values,
        "other": (mapped == "other").values,
    }
    return mapped, masks


def _kmm_sample_weights_for_df(df: pd.DataFrame, scenario: str, weight_df: Optional[pd.DataFrame],
                               target_other: bool = False) -> np.ndarray:
    n = len(df)
    weights = np.ones(n, dtype=float)
    if weight_df is None or weight_df.empty:
        return weights
    id_col = U.id_column(df)
    if id_col is None or id_col not in weight_df.columns:
        return weights
    # Ensure consistent string join on ID to avoid all-NaN merges
    df_ids = df[[id_col]].copy()
    df_ids[id_col] = df_ids[id_col].astype(str)
    wdf = weight_df.copy()
    wdf[id_col] = wdf[id_col].astype(str)
    merged = df_ids.merge(wdf, how="left", on=id_col)

    def _nanmax_or_default(arrs, default=1.0):
        stack = np.vstack(arrs)
        # default any-all-nan columns
        col_all_nan = np.all(np.isnan(stack), axis=0)
        with np.errstate(all="ignore", invalid="ignore"):
            out = np.nanmax(stack, axis=0)
        if col_all_nan.any():
            out[col_all_nan] = default
        return out

    def _col(name: str):
        return merged[name].to_numpy(dtype=float) if name in merged.columns else np.full(n, np.nan, dtype=float)

    eth_labels, masks = _eth_masks(df)

    if target_other:
        mapping = {
            "white": "Weight_sWhite_tOther",
            "black": "Weight_sBlack_tOther",
            "hispanic": "Weight_sHispanic_tOther",
        }
        for grp, col in mapping.items():
            vals = _col(col)
            if masks[grp].any():
                weights[masks[grp]] = np.where(np.isfinite(vals[masks[grp]]), vals[masks[grp]], weights[masks[grp]])
        return weights

    scn = scenario.lower()
    if scn == "all":
        w_wb = _col("Weight_sWhite_tBlack")
        w_wh = _col("Weight_sWhite_tHispanic")
        w_white = _nanmax_or_default([w_wb, w_wh], default=1.0)
        white_vals = w_white[masks["white"]]
        median_white, n_white = U.clipped_median_stats(white_vals)
        num_white = max(1, n_white if n_white > 0 else int(masks["white"].sum()))
        num_black = max(1, int(masks["black"].sum()))
        num_hisp = max(1, int(masks["hispanic"].sum()))
        weights[masks["white"]] = np.where(np.isfinite(w_white[masks["white"]]), w_white[masks["white"]], median_white)
        weights[masks["black"]] = (num_white / num_black) * median_white
        weights[masks["hispanic"]] = (num_white / num_hisp) * median_white
        return weights

    if scn == "white+black":
        w_w = _col("Weight_sWhite_tHispanic")
        w_b = _col("Weight_sBlack_tHispanic")
        weights[masks["white"]] = np.where(np.isfinite(w_w[masks["white"]]), w_w[masks["white"]], 1.0)
        weights[masks["black"]] = np.where(np.isfinite(w_b[masks["black"]]), w_b[masks["black"]], 1.0)
        if masks["hispanic"].any():
            w_h = _col("Weight_sHispanic_tOther")
            weights[masks["hispanic"]] = np.where(np.isfinite(w_h[masks["hispanic"]]), w_h[masks["hispanic"]], 1.0)
        return weights

    if scn == "white+hispanic":
        w_w = _col("Weight_sWhite_tBlack")
        w_h = _col("Weight_sHispanic_tBlack")
        weights[masks["white"]] = np.where(np.isfinite(w_w[masks["white"]]), w_w[masks["white"]], 1.0)
        weights[masks["hispanic"]] = np.where(np.isfinite(w_h[masks["hispanic"]]), w_h[masks["hispanic"]], 1.0)
        if masks["black"].any():
            w_b = _col("Weight_sBlack_tOther")
            weights[masks["black"]] = np.where(np.isfinite(w_b[masks["black"]]), w_b[masks["black"]], 1.0)
        return weights

    if scn == "black+hispanic":
        w_b = _col("Weight_sBlack_tWhite")
        w_h = _col("Weight_sHispanic_tWhite")
        weights[masks["black"]] = np.where(np.isfinite(w_b[masks["black"]]), w_b[masks["black"]], 1.0)
        weights[masks["hispanic"]] = np.where(np.isfinite(w_h[masks["hispanic"]]), w_h[masks["hispanic"]], 1.0)
        if masks["white"].any():
            w_w = _col("Weight_sWhite_tOther")
            weights[masks["white"]] = np.where(np.isfinite(w_w[masks["white"]]), w_w[masks["white"]], 1.0)
        return weights

    if scn == "white":
        w1 = _col("Weight_sWhite_tBlack")
        w2 = _col("Weight_sWhite_tHispanic")
        w_white = _nanmax_or_default([w1, w2], default=1.0)
        weights[masks["white"]] = np.where(np.isfinite(w_white[masks["white"]]), w_white[masks["white"]], 1.0)
        return weights

    if scn == "black":
        w1 = _col("Weight_sBlack_tWhite")
        w2 = _col("Weight_sBlack_tHispanic")
        w_blk = _nanmax_or_default([w1, w2], default=1.0)
        weights[masks["black"]] = np.where(np.isfinite(w_blk[masks["black"]]), w_blk[masks["black"]], 1.0)
        return weights

    if scn == "hispanic":
        w1 = _col("Weight_sHispanic_tWhite")
        w2 = _col("Weight_sHispanic_tBlack")
        w_h = _nanmax_or_default([w1, w2], default=1.0)
        weights[masks["hispanic"]] = np.where(np.isfinite(w_h[masks["hispanic"]]), w_h[masks["hispanic"]], 1.0)
        return weights

    return weights


def _external_tag(name_no_ext: str, cohorts: Optional[List[str]], is_other: bool) -> str:
    clean = name_no_ext or ""
    if clean.lower().startswith("external"):
        remainder = clean[len("external"):].lstrip("_-")
        clean = "external" + (f"_{remainder}" if remainder else "")
    elif not clean:
        clean = "external"
    cohort_tokens = []
    for c in (cohorts or []):
        if not c:
            continue
        token = os.path.basename(str(c))
        token = token.replace(".csv", "")
        if "/" in token or "\\" in token:
            continue
        cohort_tokens.append(token)
    suffix = "_all" if (is_other or not cohort_tokens) else "_" + "_".join(cohort_tokens)
    tag = f"{clean}{suffix}"
    # collapse any accidental double 'external' segments
    while "external_external" in tag.lower():
        tag = tag.replace("external_external", "external_", 1)
    return tag


def _tune_baseline(Xtr, ytr, model_name: str, cv_splits: int, device: str):
    init_pars = U.default_params(model_name, device=device)
    param_grid = U.grid_params(model_name)
    if model_name == "xgboost_vanilla":
        n_pos = max(1, int((ytr == 1).sum()))
        n_neg = int((ytr == 0).sum())
        init_pars = dict(init_pars)
        init_pars["scale_pos_weight"] = max(1.0, n_neg / n_pos)
        if "scale_pos_weight" in param_grid:
            param_grid.pop("scale_pos_weight", None)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    base = ML.CustomML(model_name, paras=init_pars, sample_weight=None)
    search = ML.MyGridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=ML.ba_cost_scorer,
        n_jobs=max(1, os.cpu_count() // 2),
        cv=cv,
        verbose=0,
        refit=True,
    ).fit(Xtr, ytr)
    best_params = getattr(search, "best_params_", init_pars)
    records = []
    if hasattr(search, "cv_results_") and search.cv_results_:
        cvres = search.cv_results_
        for i in range(len(cvres.get("mean_test_score", []))):
            rec = {"mean_test_score": float(cvres["mean_test_score"][i])}
            if "std_test_score" in cvres:
                rec["std_test_score"] = float(cvres["std_test_score"][i])
            for k, v in cvres.items():
                if k.startswith("param_"):
                    rec[k.replace("param_", "")] = v[i]
            records.append(rec)
    return best_params, records


def _prefit_cr_models(Xtr, Xte, train_df, test_df, alphas: List[float],
                      external_blocks: Optional[List[Tuple[str, pd.DataFrame, np.ndarray, np.ndarray, bool]]]):
    def _sens_vec(df):
        if df is None or "ETHNICITY" not in df.columns:
            return np.zeros(len(df) if df is not None else 0, dtype=int)
        mapped = df["ETHNICITY"].map(U.ETH_MAP).fillna(df["ETHNICITY"]).str.lower()
        return mapped.map({"nhw": 1, "nha": 2, "hwa": 3}).fillna(4).astype(int).values

    s_tr = _sens_vec(train_df)
    s_te = _sens_vec(test_df)

    X_blocks = []
    s_blocks = []
    slices = {}
    start = 0
    external_map: Dict[int, str] = {}

    def _add_block(key: str, X, s):
        nonlocal start
        if X is None or len(X) == 0:
            return
        X_blocks.append(X)
        s_blocks.append(s.astype(int))
        slices[key] = (start, start + len(X))
        start += len(X)

    _add_block("train", Xtr, s_tr)
    _add_block("test", Xte, s_te)

    if external_blocks:
        for idx, (name, df_e, _, Xe, _) in enumerate(external_blocks):
            s_ext = _sens_vec(df_e)
            key = f"ext_{idx}"
            _add_block(key, Xe, s_ext)
            external_map[idx] = key

    if not X_blocks:
        cache = {a: {"train": Xtr, "test": Xte, "externals": {}} for a in alphas}
        pool = {"X": None, "s": None, "has_groups": False}
        return cache, pool

    X_pool = np.vstack(X_blocks)
    s_pool = np.concatenate(s_blocks)
    has_groups = np.unique(s_pool).size >= 2
    if not has_groups:
        print("[WARN] CR: pooled data contains a single group; correlation removal skipped.")
    else:
        cnts = {
            "NHW": int((s_pool == 1).sum()),
            "NHA": int((s_pool == 2).sum()),
            "HWA": int((s_pool == 3).sum()),
            "Other": int((s_pool == 4).sum()),
        }
        print("[CR] pooled sensitive counts:", ", ".join(f"{k}={v}" for k, v in cnts.items()))

    cache = {}
    rng = np.random.default_rng(42)
    for i, a in enumerate(alphas):
        if has_groups:
            random_number = np.random.randint(1, 11)
            rng = np.random.default_rng(random_number**i+42)
            n_pool = len(X_pool)
            sel_size = max(int(0.9 * n_pool), min(n_pool, 100))
            idx = rng.choice(n_pool, size=sel_size, replace=False) if sel_size < n_pool else np.arange(n_pool)
            _, X_pool_cr = U.apply_correlation_removal(X_pool[idx], X_pool, s_pool[idx], s_pool, a)
        else:
            X_pool_cr = X_pool.copy()
        block = {}
        if "train" in slices:
            sl = slices["train"]
            block["train"] = X_pool_cr[sl[0]:sl[1]]
        if "test" in slices:
            sl = slices["test"]
            block["test"] = X_pool_cr[sl[0]:sl[1]]
        external_vals = {}
        for idx, key in external_map.items():
            sl = slices.get(key)
            if sl is not None:
                external_vals[idx] = X_pool_cr[sl[0]:sl[1]]
        block["externals"] = external_vals
        cache[a] = block

    pool = {"X": X_pool, "s": s_pool, "has_groups": has_groups}
    return cache, pool

def _cr_transform_with_pool(pool_info, Xe, s_e, alpha):
    if not pool_info or not pool_info.get("has_groups"):
        return Xe
    X_pool = pool_info["X"]
    s_pool = pool_info["s"]
    _, Xe_cr = U.apply_correlation_removal(X_pool, Xe, s_pool, s_e, alpha)
    return Xe_cr


def _select_cr_alpha(cr_cache: Dict[float, Dict[str, Any]], y_train, baseline_params, model_name,
                     cv_splits: int, default_Xtr: np.ndarray):
    best_alpha, best_score = None, -np.inf
    records = []
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    for alpha, block in cr_cache.items():
        Xcr = block.get("train", default_Xtr)
        scores = []
        for tr_idx, va_idx in cv.split(Xcr, y_train):
            est = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
            est.fit(Xcr[tr_idx], y_train[tr_idx])
            scores.append(ML.ba_cost_scorer(est, Xcr[va_idx], y_train[va_idx]))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        std_score = float(np.std(scores)) if scores else float("nan")
        records.append({"alpha": float(alpha), "mean_test_score": mean_score, "std_test_score": std_score})
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha
    return best_alpha, records


def _select_kmm_params(train_df, scenario: str, weight_root: str, baseline_params: Dict,
                       model_name: str, Xtr: np.ndarray, y_train: np.ndarray, cv_splits: int):
    best_score, best_param_id = -np.inf, None
    best_weights = None
    best_weights_other = None
    kmm_grid = U.kmm_param_grid_list()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    records = []

    for params in kmm_grid:
        pid = U.kmm_param_id(params)
        w_df = _load_kmm_weights(weight_root, scenario, pid)
        if w_df is None:
            continue
        sw_full = _kmm_sample_weights_for_df(train_df, scenario, w_df, target_other=False)
        scores = []
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            est = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
            est.fit(Xtr[tr_idx], y_train[tr_idx], sample_weight=sw_full[tr_idx])
            scores.append(ML.ba_cost_scorer(est, Xtr[va_idx], y_train[va_idx]))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        std_score = float(np.std(scores)) if scores else float("nan")
        records.append({"param_id": pid, "mean_test_score": mean_score, "std_test_score": std_score})
        if mean_score > best_score:
            best_score = mean_score
            best_param_id = pid
            best_weights = sw_full
            best_weights_other = _kmm_sample_weights_for_df(train_df, scenario, w_df, target_other=True)

    return best_param_id, best_weights, best_weights_other, records


def _build_custom_targets(method: str, scenario: str, Xte: np.ndarray, y_test: np.ndarray,
                          test_df: pd.DataFrame, external_blocks: List[Tuple[str, pd.DataFrame, np.ndarray, np.ndarray, bool]]):
    def _eth_to_gid(series: pd.Series) -> np.ndarray:
        mapped, _ = _eth_masks(pd.DataFrame({"ETHNICITY": series})) if isinstance(series, pd.Series) else (pd.Series(series), None)
        ser = mapped.astype(str).str.lower()
        code_map = {"white": 0, "black": 1, "hispanic": 2, "other": 3}
        return ser.map(code_map).fillna(3).astype(int).values

    scn = scenario.lower()
    if method.lower() == "ssda":
        parts = [Xte]
        gids = [_eth_to_gid(test_df["ETHNICITY"])] if "ETHNICITY" in test_df.columns else []
        for _, df_e, _, Xe, _ in external_blocks:
            parts.append(Xe)
            if "ETHNICITY" in df_e.columns:
                gids.append(_eth_to_gid(df_e["ETHNICITY"]))
        if parts:
            target_X = np.vstack(parts).astype(np.float32)
        else:
            target_X = np.empty((0, Xte.shape[1]), dtype=np.float32)
        target_gid = np.concatenate(gids).astype(int) if gids else None
        return target_X, None, target_gid

    groups_map = {
        "all": ["white", "black", "hispanic"],
        "white+black": ["white", "black"],
        "white+hispanic": ["white", "hispanic"],
        "black+hispanic": ["black", "hispanic"],
        "white": ["white"],
        "black": ["black"],
        "hispanic": ["hispanic"],
    }
    groups = groups_map.get(scn, ["white", "black", "hispanic"])

    def _sample_block(df, Xe, y_arr, groups_sel, seed_base=0, per_class: int = 10):
        X_parts, y_parts = [], []
        eth_lab, _ = _eth_masks(df)
        for g in groups_sel:
            gmask = (eth_lab == g).values
            for lbl in [0, 1]:
                idx = np.where(gmask & (y_arr == lbl))[0]
                if len(idx) == 0:
                    continue
                k = min(per_class, len(idx))
                rng = np.random.default_rng(seed_base + len(X_parts) + lbl)
                sel = rng.choice(idx, size=k, replace=False)
                X_parts.append(Xe[sel])
                y_parts.append(y_arr[sel])
        return X_parts, y_parts

    X_parts, y_parts = _sample_block(test_df, Xte, y_test, groups, seed_base=42)
    gid_parts = []
    if "ETHNICITY" in test_df.columns:
        gid_parts.append(_eth_to_gid(test_df["ETHNICITY"]))
    for _, df_e, y_e, Xe, _ in external_blocks:
        eth_block, _ = _eth_masks(df_e)
        block_groups = list(groups)
        if "other" in eth_block.unique():
            block_groups = block_groups + (["other"] if "other" not in block_groups else [])
        xp, yp = _sample_block(df_e, Xe, y_e, block_groups, seed_base=21)
        X_parts.extend(xp)
        y_parts.extend(yp)
        if "ETHNICITY" in df_e.columns and len(xp) > 0:
            # expand gid to match sampled rows
            gid_full = _eth_to_gid(df_e["ETHNICITY"])
            # collect gids corresponding to selected indices
            idxs = []
            for g in block_groups:
                mask = (eth_block == g)
                for lbl in [0, 1]:
                    sel = np.where(mask & (df_e["DX"].values == lbl))[0]
                    k = min(10, len(sel))
                    if k > 0:
                        rng = np.random.default_rng(21 + len(idxs) + lbl)
                        idxs.extend(rng.choice(sel, size=k, replace=False).tolist())
            if idxs:
                gid_parts.append(gid_full[idxs])

    if X_parts:
        target_X = np.vstack(X_parts).astype(np.float32)
        target_y = np.concatenate(y_parts).astype(np.float32)
    else:
        target_X = np.empty((0, Xte.shape[1]), dtype=np.float32)
        target_y = np.empty((0,), dtype=np.float32)
    target_gid = np.concatenate(gid_parts).astype(int) if gid_parts else None
    return target_X, target_y, target_gid


def run_one_fold(
    scenario: str,
    fold: int,
    method: str,
    model_name: str,
    splits_dir: str,
    save_root: str,
    device: str = "cuda",
    scale_mode: int = 0,
    cv_splits: int = 5,
    early_stopping: int = 50,
    threshold: float = 0.5,
    external_csvs: Optional[List[str]] = None,
    external_cohorts: Optional[List[str]] = None,
    refit: bool = True,
    kmm_weights_dir: str = "./checkpoints/kmm_weights",
    loss_mode: Optional[str] = None,
    total_steps: int = 500,
    cr_alpha_default: float = 0.05,
):
    method_l = method.lower()
    scenario_l = scenario.lower()

    needs_loss_suffix = method_l in {"ssda", "regalign"}
    effective_loss_mode = loss_mode
    if needs_loss_suffix and not effective_loss_mode:
        effective_loss_mode = "ce+target+align+gap" if method_l == "ssda" else "ce+align+gap"

    tag_parts = [method_l, model_name]
    if needs_loss_suffix and effective_loss_mode:
        tag_parts.append(str(effective_loss_mode))
    tag = "_".join(tag_parts)
    out_dir = os.path.join(save_root, tag, scenario_l, f"fold_{fold:02d}")
    U.ensure_dir(out_dir)

    train_df, test_df, y_train, y_test, feat_cols, Xtr, Xte, scaler, Xtr_raw, icv_tr = _load_fold_data(
        scenario, fold, splits_dir, scale_mode
    )
    external_blocks = _load_external_blocks(external_csvs, external_cohorts, feat_cols, scaler, scale_mode, Xtr_raw, icv_tr)

    baseline_params = U.try_load_baseline_best_params(save_root, model_name, scenario, fold)
    baseline_sens = U.try_load_baseline_sensitivity(save_root, model_name, scenario, fold)
    if (baseline_params is None) or (baseline_sens is None):
        baseline_params, baseline_sens = _tune_baseline(Xtr, y_train, model_name, cv_splits, device)
        U.save_baseline_best_params(save_root, model_name, scenario, fold, baseline_params)
        U.save_baseline_sensitivity(save_root, model_name, scenario, fold, baseline_sens)

    # Try to load existing method params/sensitivity from prior run (for prediction-only reuse)
    best_method_params: Dict = {}
    sensitivity_records: List[Dict] = list(baseline_sens or [])
    best_params_path = os.path.join(out_dir, "best_params.json")
    sens_stage_path = os.path.join(out_dir, "sensitivity_stageB.json")
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, "r") as f:
                cached = json.load(f)
            best_method_params = cached.get("best_method_params") or {}
            # allow baseline params override if found
            baseline_params = cached.get("best_model_params") or baseline_params
        except Exception:
            pass
    if os.path.exists(sens_stage_path) and method_l != "baseline":
        try:
            with open(sens_stage_path, "r") as f:
                sensitivity_records = json.load(f)
        except Exception:
            pass

    # -------------- Method branches --------------
    if method_l == "baseline":
        estimator = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
        estimator = _fit_with_threshold(estimator, Xtr, y_train, feat_cols, early_stopping) if refit else estimator.fit(Xtr, y_train)
        eff_thr = _effective_threshold(estimator, threshold)
        overall, groups, yproba = _evaluate(estimator, Xte, y_test, eff_thr, test_df.get("ETHNICITY"))
        U.save_predictions(test_df, yproba, out_dir, tag="fold")
        U.save_summaries(out_dir, tag, fold, overall, groups, tag="fold")
        # External evals (no fairness transforms for baseline)
        if external_blocks:
            for name, df_e, y_e, Xe, is_other in external_blocks:
                base_tag = _external_tag(name, external_cohorts, is_other)
                o_ext, g_ext, p_ext = _evaluate(estimator, Xe, y_e, eff_thr, df_e.get("ETHNICITY"))
                U.save_predictions(df_e, p_ext, out_dir, tag=base_tag)
                if is_other:
                    with open(os.path.join(out_dir, f"summary_metrics_{base_tag}.json"), "w") as f:
                        json.dump({**o_ext, "suffix": base_tag, "fold": fold}, f, indent=2)
                else:
                    U.save_summaries(out_dir, base_tag, fold, o_ext, g_ext, tag=base_tag)
    elif method_l == "cr":
        if best_method_params:
            best_alpha = best_method_params.get("cr_alpha", cr_alpha_default)
            cr_cache, cr_pool = _prefit_cr_models(Xtr, Xte, train_df, test_df, [best_alpha], external_blocks)
        else:
            alphas = U.method_param_grid("cr").get("cr_alpha", [cr_alpha_default])
            cr_cache, cr_pool = _prefit_cr_models(Xtr, Xte, train_df, test_df, alphas, external_blocks)
            best_alpha, sens = _select_cr_alpha(cr_cache, y_train, baseline_params, model_name, cv_splits, Xtr)
            sensitivity_records = sens
            best_method_params = {"cr_alpha": best_alpha}
            # keep only best-alpha matrices for downstream use
            cr_cache, cr_pool = _prefit_cr_models(Xtr, Xte, train_df, test_df, [best_alpha], external_blocks)
        blocks = cr_cache.get(best_alpha, {})
        Xtr_best = blocks.get("train", Xtr)
        Xte_best = blocks.get("test", Xte)
        estimator = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
        estimator = _fit_with_threshold(estimator, Xtr_best, y_train, feat_cols, early_stopping) if refit else estimator.fit(Xtr_best, y_train)
        eff_thr = _effective_threshold(estimator, threshold)
        overall, groups, yproba = _evaluate(estimator, Xte_best, y_test, eff_thr, test_df.get("ETHNICITY"))
        U.save_predictions(test_df, yproba, out_dir, tag="fold")
        U.save_summaries(out_dir, tag, fold, overall, groups, tag="fold")

        # external with CR transform
        if external_blocks:
            ext_cache = blocks.get("externals", {}) if blocks else {}
            for idx, (name, df_e, y_e, Xe, is_other) in enumerate(external_blocks):
                if "ETHNICITY" in df_e.columns:
                    s_e = df_e["ETHNICITY"].map(U.ETH_MAP).fillna(df_e["ETHNICITY"]).str.lower()
                    s_e = s_e.map({"nhw": 1, "nha": 2, "hwa": 3}).fillna(4).astype(int).values
                else:
                    s_e = np.zeros(len(df_e), dtype=int)
                if idx in ext_cache:
                    Xe_cr = ext_cache[idx]
                else:
                    Xe_cr = _cr_transform_with_pool(cr_pool, Xe, s_e, best_alpha)
                base_tag = _external_tag(name, external_cohorts, is_other)
                o_ext, g_ext, p_ext = _evaluate(estimator, Xe_cr, y_e, eff_thr, df_e.get("ETHNICITY"))
                U.save_predictions(df_e, p_ext, out_dir, tag=base_tag)
                if is_other:
                    with open(os.path.join(out_dir, f"summary_metrics_{base_tag}.json"), "w") as f:
                        json.dump({**o_ext, "suffix": base_tag, "fold": fold}, f, indent=2)
                else:
                    U.save_summaries(out_dir, base_tag, fold, o_ext, g_ext, tag=base_tag)
    elif method_l == "kmm":
        if best_method_params:
            pid = best_method_params.get("kmm_param_id")
            sw_main = _kmm_sample_weights_for_df(train_df, scenario, _load_kmm_weights(kmm_weights_dir, scenario, pid), target_other=False) if pid else None
            sw_other = _kmm_sample_weights_for_df(train_df, scenario, _load_kmm_weights(kmm_weights_dir, scenario, pid), target_other=True) if pid else None
        else:
            pid, sw_main, sw_other, sens = _select_kmm_params(train_df, scenario, kmm_weights_dir, baseline_params,
                                                        model_name, Xtr, y_train, cv_splits)
            sensitivity_records = sens
            best_method_params = {"kmm_param_id": pid}
        if sw_main is None:
            sw_main = np.ones(len(train_df), dtype=float)
        estimator_main = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
        estimator_main = _fit_with_threshold(estimator_main, Xtr, y_train, feat_cols, early_stopping, sample_weight=sw_main) if refit else estimator_main.fit(Xtr, y_train, sample_weight=sw_main)
        eff_thr_main = _effective_threshold(estimator_main, threshold)
        overall, groups, yproba = _evaluate(estimator_main, Xte, y_test, eff_thr_main, test_df.get("ETHNICITY"))
        U.save_predictions(test_df, yproba, out_dir, tag="fold")
        U.save_summaries(out_dir, tag, fold, overall, groups, tag="fold")

        estimator_other = None
        eff_thr_other = None
        if sw_other is not None and sw_other.size == len(train_df):
            estimator_other = ML.CustomML(model_name, paras=baseline_params, sample_weight=None)
            estimator_other = _fit_with_threshold(estimator_other, Xtr, y_train, feat_cols, early_stopping, sample_weight=sw_other) if refit else estimator_other.fit(Xtr, y_train, sample_weight=sw_other)
            eff_thr_other = _effective_threshold(estimator_other, threshold)

        if external_blocks:
            for name, df_e, y_e, Xe, is_other in external_blocks:
                eth_block, _ = _eth_masks(df_e)
                is_other_block = (set(eth_block.unique()) == {"other"}) or is_other
                est_use = estimator_other if (estimator_other is not None and is_other_block) else estimator_main
                thr_use = eff_thr_other if (is_other_block and eff_thr_other is not None) else eff_thr_main
                base_tag = _external_tag(name, external_cohorts, is_other_block)
                o_ext, g_ext, p_ext = _evaluate(est_use, Xe, y_e, thr_use, df_e.get("ETHNICITY"))
                U.save_predictions(df_e, p_ext, out_dir, tag=base_tag)
                if is_other_block:
                    with open(os.path.join(out_dir, f"summary_metrics_{base_tag}.json"), "w") as f:
                        json.dump({**o_ext, "suffix": base_tag, "fold": fold}, f, indent=2)
                else:
                    U.save_summaries(out_dir, base_tag, fold, o_ext, g_ext, tag=base_tag)
    elif method_l in {"ssda", "regalign"}:
        target_X, target_y, target_gid = _build_custom_targets(method_l, scenario, Xte, y_test, test_df, external_blocks)
        base_params = dict(baseline_params)
        base_params.pop("eval_metric", None)
        gid_source = _eth_masks(train_df)[0].map({"white":0,"black":1,"hispanic":2,"other":3}).fillna(3).astype(int).values \
                     if "ETHNICITY" in train_df.columns else None
        est_base = ML.XGBCustomObjectiveEstimator(
            objective_version="v3",
            objective_subtype="1C" if method_l == "regalign" else "2B",
            total_steps=total_steps,
            eval_metric="auc",
            loss_mode=effective_loss_mode,
            target_X=target_X,
            target_y=target_y if method_l == "regalign" else None,
            group_ids_source=gid_source,
            group_ids_target=target_gid,
            **base_params,
        )
        if best_method_params:
            # reuse existing best params (skip grid search)
            if refit:
                refit_params = dict(baseline_params)
                refit_params.pop("eval_metric", None)
                refit_params.update(best_method_params)
                estimator = ML.XGBCustomObjectiveEstimator(
                    objective_version="v3",
                    objective_subtype="1C" if method_l == "regalign" else "2B",
                    total_steps=total_steps,
                    eval_metric="auc",
                    loss_mode=effective_loss_mode,
                    target_X=target_X,
                    target_y=target_y if method_l == "regalign" else None,
                    group_ids_source=gid_source,
                    group_ids_target=target_gid,
                    **refit_params,
                )
                estimator = _fit_with_threshold(estimator, Xtr, y_train, feat_cols, early_stopping)
            else:
                estimator = est_base.set_params(**best_method_params)
                estimator.fit(Xtr, y_train)
        else:
            raw_grid = U.method_param_grid(method_l)
            # Ensure all entries are iterable for ParameterGrid
            norm_grid = {}
            for k, v in raw_grid.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    norm_grid[k] = list(v)
                else:
                    norm_grid[k] = [v]
            param_grid = norm_grid
            grid_list = list(ParameterGrid(param_grid))
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            gs = ML.MyGridSearchCV(
                estimator=est_base,
                param_grid=param_grid,
                scoring=ML.ba_cost_scorer,
                n_jobs=max(1, os.cpu_count() // 2),
                cv=cv,
                verbose=0,
                refit=True,
            ).fit(Xtr, y_train)
            best_method_params = getattr(gs, "best_params_", {}) or {}
            if hasattr(gs, "cv_results_") and gs.cv_results_:
                cvres = gs.cv_results_
                sens = []
                for i in range(len(cvres.get("mean_test_score", []))):
                    rec = {"mean_test_score": float(cvres["mean_test_score"][i])}
                    if "std_test_score" in cvres:
                        rec["std_test_score"] = float(cvres["std_test_score"][i])
                    for k, v in cvres.items():
                        if k.startswith("param_"):
                            rec[k.replace("param_", "")] = v[i]
                    sens.append(rec)
                sensitivity_records = sens
            if refit:
                refit_params = dict(baseline_params)
                refit_params.pop("eval_metric", None)
                refit_params.update(best_method_params)
                estimator = ML.XGBCustomObjectiveEstimator(
                    objective_version="v3",
                    objective_subtype="1C" if method_l == "regalign" else "2B",
                    total_steps=total_steps,
                    eval_metric="auc",
                    loss_mode=effective_loss_mode,
                    target_X=target_X,
                    target_y=target_y if method_l == "regalign" else None,
                    group_ids_source=gid_source,
                    group_ids_target=target_gid,
                    **refit_params,
                )
                estimator = _fit_with_threshold(estimator, Xtr, y_train, feat_cols, early_stopping)
            else:
                estimator = gs.best_estimator_
        eff_thr = _effective_threshold(estimator, threshold)
        overall, groups, yproba = _evaluate(estimator, Xte, y_test, eff_thr, test_df.get("ETHNICITY"))
        U.save_predictions(test_df, yproba, out_dir, tag="fold")
        U.save_summaries(out_dir, tag, fold, overall, groups, tag="fold")

        if external_blocks:
            for name, df_e, y_e, Xe, is_other in external_blocks:
                base_tag = _external_tag(name, external_cohorts, is_other)
                o_ext, g_ext, p_ext = _evaluate(estimator, Xe, y_e, eff_thr, df_e.get("ETHNICITY"))
                U.save_predictions(df_e, p_ext, out_dir, tag=base_tag)
                if is_other:
                    with open(os.path.join(out_dir, f"summary_metrics_{base_tag}.json"), "w") as f:
                        json.dump({**o_ext, "suffix": base_tag, "fold": fold}, f, indent=2)
                else:
                    U.save_summaries(out_dir, base_tag, fold, o_ext, g_ext, tag=base_tag)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Sensitivity dumps
    # - baseline: keep sensitivity.json (from cache) in fold directory for convenience
    # - methods: write sensitivity_stageB.json
    if method_l == "baseline":
        with open(os.path.join(out_dir, "sensitivity.json"), "w") as f:
            json.dump(baseline_sens or sensitivity_records, f, indent=2)
    else:
        with open(os.path.join(out_dir, "sensitivity_stageB.json"), "w") as f:
            json.dump(sensitivity_records, f, indent=2)

    meta = {
        "feature_names": feat_cols,
        "model_name": model_name,
        "effective_threshold": float(_effective_threshold(locals().get("estimator", None) or locals().get("estimator_main", None), threshold)),
        "baseline_params": baseline_params,
        "best_method_params": best_method_params,
    }
    if needs_loss_suffix:
        meta["loss_mode"] = effective_loss_mode
    with open(os.path.join(out_dir, f"meta_fold{fold}_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"best_model_params": baseline_params, "best_method_params": best_method_params}, f, indent=2)


def main():
    ap = argparse.ArgumentParser("Training / Evaluation v2 refreshed pipeline")
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--save_root", type=str, default="./runs")
    ap.add_argument("--scenarios", type=str, default="all")
    ap.add_argument("--folds", type=str, default="1-10")
    ap.add_argument("--method", type=str, default="baseline")
    ap.add_argument("--model_name", type=str, default="xgboost_vanilla")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scale_mode", type=int, default=0)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--early_stopping", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--external_csvs", nargs="*", default=None)
    ap.add_argument("--external_cohorts", nargs="*", default=None)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--kmm_weights_dir", type=str, default="./checkpoints/kmm_weights")
    ap.add_argument("--loss_mode", type=str, default=None)
    ap.add_argument("--total_steps", type=int, default=500)
    ap.add_argument("--cr_alpha_default", type=float, default=0.05)
    args = ap.parse_args()

    def _parse_range(txt):
        if "-" in txt:
            a, b = txt.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in txt.split(",") if x]

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    folds = _parse_range(args.folds)

    for sc in scenarios:
        for k in folds:
            run_one_fold(
                scenario=sc, fold=k, method=args.method, model_name=args.model_name,
                splits_dir=args.splits_dir, save_root=args.save_root, device=args.device,
                scale_mode=args.scale_mode, cv_splits=args.cv_splits, early_stopping=args.early_stopping,
                threshold=args.threshold, external_csvs=args.external_csvs, external_cohorts=args.external_cohorts,
                refit=args.refit, kmm_weights_dir=args.kmm_weights_dir, loss_mode=args.loss_mode,
                total_steps=args.total_steps, cr_alpha_default=args.cr_alpha_default,
            )


if __name__ == "__main__":
    main()
