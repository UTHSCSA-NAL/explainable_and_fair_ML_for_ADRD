#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
)

from dementia_ai.core import ops
from dementia_ai.models import models as ML

# -------------------- constants --------------------
_CONFIG_CACHE = {}

def _config_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _load_yaml(path: Path) -> dict:
    try:
        with path.open('r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def _load_configs() -> dict:
    if _CONFIG_CACHE:
        return _CONFIG_CACHE
    root = _config_root()
    defaults = _load_yaml(root / 'configs' / 'default.yaml')
    grids = _load_yaml(root / 'configs' / 'grids.yaml')
    if not defaults or not grids:
        pkg_root = Path(__file__).resolve().parents[1]
        if not defaults:
            defaults = _load_yaml(pkg_root / 'configs' / 'default.yaml')
        if not grids:
            grids = _load_yaml(pkg_root / 'configs' / 'grids.yaml')
    _CONFIG_CACHE['defaults'] = defaults
    _CONFIG_CACHE['grids'] = grids
    return _CONFIG_CACHE


NON_FEATURE_COLS = [
    "COHORT", "PHASE", "PTID", "RID", "MRI_ID", "SITEID",
    "EDUCATION", "MARRY", "SEX", "AGE", "ADAS13", "CDRSB", "MMSE",
    "ICV",
    "ETHNICITY", "DX"]
ETH_MAP = {"White": "NHW", "Black": "NHA", "Hispanic": "HWA",
           "white": "NHW", "black": "NHA", "hispanic": "HWA",
           "NHW": "NHW", "NHA": "NHA", "HWA": "HWA"}
DX_MAP = {"CN": 0, "AD": 1, 0: 0, 1: 1}

METHOD_KEYS_REGALIGN = {
    'eq_lambda', 'cal_lambda', 'fair_grad_alpha', 'fair_grad_ema',
    'tau', 'eps', 'target_weight',
    'source_focal_gamma', 'target_focal_gamma', 'target_pos_weight', 'class1_boost',
    'gap_lambda', 'ba_tau', 'update_every'

}
METHOD_KEYS_CR = {'cr_alpha'}
METHOD_KEYS_SSDA = {
    'eq_lambda', 'cal_lambda', 'fair_grad_alpha', 'fair_grad_ema',
    'tau', 'eps', 'target_weight',
    'source_focal_gamma', 'target_pos_weight', 'class1_boost',
    'gap_lambda', 'ba_tau', 'update_every'
}

# -------------------- trivial helpers --------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

def load_split_csv(path: str, cohort: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    if df["DX"].dtype == object:
        df["DX"] = df["DX"].map(DX_MAP).astype(int)
    df = df[df["DX"].isin([0, 1])].reset_index(drop=True)
    if cohort is not None and cohort.strip().lower() != "all" and "COHORT" in df.columns:
        df = df[df["COHORT"].str.lower() == cohort.lower()].reset_index(drop=True)
    return df, df["DX"].values.astype(int)

def maybe_harmonize(train_df: pd.DataFrame, val_df: pd.DataFrame, site_col: str = "SITEID"):
    # Placeholder for ComBat/etc if needed
    return train_df, val_df

def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["RID", "PTID", "MRI_ID", "Subject_ID", "Med_ID"]:
        if c in df.columns:
            return c
    return None

# -------------------- CR / KMM wrappers --------------------

def compute_kmm_weights(Xs: np.ndarray, Xt: np.ndarray) -> Optional[np.ndarray]:
    try:
        w = ops.compute_sample_weight_kmm(Xs, Xt)
        if hasattr(ops, "normalize_weight"):
            w = ops.normalize_weight(w)
        return w
    except Exception as e:
        print(f"[WARN] KMM failed: {e}")
        return None

def apply_correlation_removal(Xtr, Xte, s_tr, s_te, alpha: float):
    try:
        return ops.correlation_remover_v2(Xtr, Xte, s_tr, s_te, alpha=alpha)
    except Exception as e:
        print(f"[WARN] CR skipped: {e}")
        return Xtr, Xte

class PrecomputedCV:
    """Wrap a list of (train_idx, valid_idx) into a splitter with .split()."""
    def __init__(self, splits):
        self._splits = list(splits)

    def split(self, X, y=None, groups=None):
        # yield like an sklearn splitter
        for tr, va in self._splits:
            yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._splits)

def _median_gamma(X: np.ndarray) -> float:
    """Median heuristic gamma for RBF; gamma = 1/(2*sigma^2)."""
    if X.shape[0] > 2000:
        Xs = X[np.random.choice(X.shape[0], 2000, replace=False)]
    else:
        Xs = X
    m = np.mean(Xs, axis=0, keepdims=True)
    Xc = Xs - m
    idx = np.random.choice(Xc.shape[0], min(2000, Xc.shape[0]), replace=False)
    D = np.sum((Xc[idx, None, :] - Xc[None, idx, :])**2, axis=2)
    dmed = np.median(D[D > 0])
    if not np.isfinite(dmed) or dmed <= 0:
        return 1.0
    sigma2 = dmed / 2.0
    return 1.0 / max(1e-12, sigma2)

def build_target_mixture(
    target_blocks: List[np.ndarray],
    mix_weights: Optional[List[float]],
    max_size: int
) -> np.ndarray:
    """Create Xt_mix by sampling each block to keep total <= max_size."""
    sizes = np.array([tb.shape[0] for tb in target_blocks], dtype=float)
    if mix_weights is None:
        w = sizes / max(1.0, sizes.sum())
    else:
        w = np.asarray(mix_weights, dtype=float)
        w = w / max(1e-12, w.sum())
    counts = np.maximum(1, np.floor(w * max_size).astype(int))
    counts[-1] += max_size - int(counts.sum())  # fix rounding
    parts = []
    for tb, c in zip(target_blocks, counts):
        if tb.shape[0] <= c:
            parts.append(tb)
        else:
            idx = np.random.choice(tb.shape[0], c, replace=False)
            parts.append(tb[idx])
    return np.vstack(parts)

def compute_kmm_weights_custom(
    Xs: np.ndarray,
    target_blocks: List[np.ndarray],
    mix_weights: Optional[List[float]],
    clip: Tuple[float, float],
    gamma: Optional[float],
    kernel: str,
    B: float,
    eps: Optional[float],
) -> Optional[np.ndarray]:
    """
    Mixture-of-targets KMM using your ops.compute_sample_weight_kmm,
    with mean-1 renorm and clipping.
    """
    from dementia_ai.core import ops as mops
    if eps is None:
        eps = 1.0 / np.sqrt(max(1, Xs.shape[0]))

    sizes = np.array([tb.shape[0] for tb in target_blocks], dtype=float)
    if mix_weights is None:
        mix = sizes / max(1.0, sizes.sum())
    else:
        mix = np.asarray(mix_weights, dtype=float)
        mix = mix / max(1e-12, mix.sum())

    acc = None
    for tb, a in zip(target_blocks, mix):
        w = mops.compute_sample_weight_kmm(
            Xs, tb, kernel=kernel, B=B, eps=eps, gamma=gamma
        )
        if w is None:
            continue
        w = np.asarray(w, float)
        acc = w * a if acc is None else (acc + a * w)

    if acc is None:
        return None
    # mean-1 renorm + clip
    m = float(np.mean(acc)) if acc.size else 1.0
    if m > 0:
        acc = acc / m
    wmin, wmax = clip
    acc = np.clip(acc, wmin, wmax)
    return acc

def make_adapt_kmm_estimator(
    base_estimator,
    Xt_mix: np.ndarray,
    kernel: str,
    B: float,
    eps: Optional[float],
    max_size: int,
    gamma: Optional[float],
    verbose: int = 0,
):
    """Wrap base_estimator with ADAPT's KMM using Xt_mix."""
    try:
        from adapt.instance_based import KMM
    except Exception as e:
        raise ImportError("ADAPT not installed. pip install adapt") from e

    if eps is None:
        eps = 1.0 / np.sqrt(max(1, Xt_mix.shape[0]))
    params = {}
    if gamma is not None:
        params["gamma"] = gamma

    return KMM(
        estimator=base_estimator,
        Xt=Xt_mix,
        kernel=kernel,
        B=B,
        eps=eps,
        max_size=max_size,
        verbose=verbose,
        **params
    )

# -------------------- params & grids --------------------

def default_params(model_name: str, device: str = "cuda") -> Dict:
    cfg = _load_configs().get('defaults', {})
    model_defaults = cfg.get('model_defaults', {})
    params = dict(model_defaults.get(model_name, {}) or {})
    if not params:
        raise ValueError(model_name)
    if model_name == 'xgboost_vanilla' and device:
        params['device'] = device
    return params


def grid_params(model_name: str) -> Dict[str, List]:
    grids = _load_configs().get('grids', {}).get('model_grids', {})
    return grids.get(model_name, {})


def method_param_grid(method: str) -> Dict[str, Any]:
    grids = _load_configs().get('grids', {}).get('method_grids', {})
    return grids.get(method.lower(), {})


def split_param_grid(full_grid: Dict[str, List], method: str):
    mkeys = METHOD_KEYS_REGALIGN if method.lower() == 'regalign' else \
            METHOD_KEYS_CR if method.lower() == 'cr' else \
            METHOD_KEYS_SSDA if method.lower() == 'ssda' else set()
    xgb_grid, method_grid = {}, {}
    for k, v in full_grid.items():
        (method_grid if k in mkeys else xgb_grid)[k] = v
    return xgb_grid, method_grid

# -------------------- KMM helpers --------------------

def kmm_param_grid_list() -> List[Dict[str, Any]]:
    """Materialize the KMM parameter grid for iteration."""
    from sklearn.model_selection import ParameterGrid
    return list(ParameterGrid(method_param_grid("kmm")))

KMM_GROUP_PAIRS = [
    ("White", "Black"), ("White", "Hispanic"), ("White", "Other"),
    ("Black", "White"), ("Black", "Hispanic"), ("Black", "Other"),
    ("Hispanic", "White"), ("Hispanic", "Black"), ("Hispanic", "Other"),
]

def kmm_group_key(src: str, tgt: str) -> str:
    return f"s{src}_t{tgt}"

def kmm_column_name(src: str, tgt: str) -> str:
    return f"Weight_{kmm_group_key(src, tgt)}"

KMM_SCENARIO_COLUMN_MAP = {
    "all": {
        kmm_column_name("White", "Black"): kmm_group_key("White", "Black"),
        kmm_column_name("White", "Hispanic"): kmm_group_key("White", "Hispanic"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "white+black": {
        kmm_column_name("White", "Hispanic"): kmm_group_key("White", "Hispanic"),
        kmm_column_name("Black", "Hispanic"): kmm_group_key("Black", "Hispanic"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "white+hispanic": {
        kmm_column_name("White", "Black"): kmm_group_key("White", "Black"),
        kmm_column_name("Hispanic", "Black"): kmm_group_key("Hispanic", "Black"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "black+hispanic": {
        kmm_column_name("Black", "White"): kmm_group_key("Black", "White"),
        kmm_column_name("Hispanic", "White"): kmm_group_key("Hispanic", "White"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "white": {
        kmm_column_name("White", "Black"): kmm_group_key("White", "Black"),
        kmm_column_name("White", "Hispanic"): kmm_group_key("White", "Hispanic"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "black": {
        kmm_column_name("Black", "White"): kmm_group_key("Black", "White"),
        kmm_column_name("Black", "Hispanic"): kmm_group_key("Black", "Hispanic"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
    "hispanic": {
        kmm_column_name("Hispanic", "White"): kmm_group_key("Hispanic", "White"),
        kmm_column_name("Hispanic", "Black"): kmm_group_key("Hispanic", "Black"),
        kmm_column_name("White", "Other"): kmm_group_key("White", "Other"),
        kmm_column_name("Black", "Other"): kmm_group_key("Black", "Other"),
        kmm_column_name("Hispanic", "Other"): kmm_group_key("Hispanic", "Other"),
    },
}

def kmm_columns_for_scenario(scenario: str) -> Dict[str, str]:
    return KMM_SCENARIO_COLUMN_MAP.get(scenario.lower(), {})


def kmm_param_id(params: Dict[str, Any]) -> str:
    """
    Stable identifier for a KMM param combo.
    Format (shortened): kernel<val>B<val>eps<val>max_size<val>gamma<val>scale<val>
    Example: kernelrbfB512eps0.1max_size500gamma0.5scale0.5
    """
    order = [
        ("kmm_kernel", "kernel"),
        ("kmm_B", "B"),
        ("kmm_eps", "eps"),
        ("kmm_max_size", "max_size"),
        ("kmm_gamma", "gamma"),
        ("kmm_scale", "scale"),
    ]
    parts = []
    for key, short in order:
        if key not in params:
            continue
        val = params[key]
        if isinstance(val, float) and val.is_integer():
            val_txt = str(int(val))
        else:
            val_txt = str(val).replace(" ", "")
        if val_txt.lower() == "none":
            val_txt = "none"
        parts.append(f"{short}{val_txt}")
    return "kmm_default" if not parts else "".join(parts)


def canonical_ethnicity(series: pd.Series) -> pd.Series:
    ser = pd.Series(series)
    mapped = ser.map(ETH_MAP)
    mapped = mapped.where(~mapped.isna(), ser)
    # Robust remapping for free-text/numeric variants
    def _norm(x):
        s = str(x).strip().lower()
        if s in {"1", "01"}:  # common numeric coding: 1=white
            s = "white"
        elif s in {"2", "02"}:
            s = "black"
        elif s in {"3", "03"}:
            s = "hispanic"
        elif any(k in s for k in ["white", "cauc", "nhw"]):
            s = "white"
        elif any(k in s for k in ["black", "african", "nha"]):
            s = "black"
        elif any(k in s for k in ["hisp", "latino", "latina", "hwa"]):
            s = "hispanic"
        else:
            s = "other"
        return s
    return mapped.apply(_norm)


def id_column(df: pd.DataFrame, fallback: str = "MRI_ID") -> Optional[str]:
    if fallback in df.columns:
        return fallback
    return pick_id_col(df)


def clipped_median_stats(weights: np.ndarray) -> tuple[float, int]:
    """Return median within mean±2SD and count of inlier weights."""
    arr = np.asarray(weights, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0, 0
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    lower, upper = mu - 2.0 * sd, mu + 2.0 * sd
    mask = (arr >= lower) & (arr <= upper)
    subset = arr[mask]
    if subset.size == 0:
        subset = arr
    return float(np.median(subset)), int(len(subset))

# -------------------- CV helpers --------------------

class StratifyOnArray:
    """Wrap StratifiedKFold to stratify by a prebuilt array."""
    def __init__(self, y_strat, n_splits=3, shuffle=True, random_state=42):
        self.y_strat = np.asarray(y_strat)
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        for tr, te in self.cv.split(X, self.y_strat):
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, self.y_strat, groups)

def prepare_cv(scenario: str, y_train: np.ndarray, train_eth: np.ndarray, cv_splits: int):
    if scenario.lower() == "all":
        mapped = pd.Series(train_eth)
        m2 = mapped.map(ETH_MAP)
        m2 = m2.where(~m2.isna(), mapped)
        e = m2.values
        y_strat = np.array([f"{int(y)}_{ee}" for y, ee in zip(y_train, e)], dtype=object)
        return StratifyOnArray(y_strat, n_splits=cv_splits, shuffle=True, random_state=42)
    else:
        return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

def filter_cv_with_min_classes(cv, y, min_classes: int = 2):
    """
    Return a list of (train_idx, valid_idx) where TRAIN has at least `min_classes` labels.
    Accepts an sklearn splitter (with .split) or an iterable of (tr, va).
    """
    import yaml

import numpy as np
    if hasattr(cv, "split"):
        splits = [(tr, va) for tr, va in cv.split(np.zeros(len(y)), y)
                  if np.unique(y[tr]).size >= min_classes]
    else:
        splits = [(tr, va) for tr, va in cv if np.unique(y[tr]).size >= min_classes]
    return splits


def label_only_cv(y, n_splits=5, random_state=42, min_classes: int = 2):
    """
    Build a StratifiedKFold on labels only, filter out any 1-class train folds,
    and wrap into a splitter with .split().
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = filter_cv_with_min_classes(skf, y, min_classes=min_classes)
    if not splits:
        raise RuntimeError("[CV] No valid folds with ≥2 classes in training splits. "
                           "Reduce n_splits or adjust data.")
    return PrecomputedCV(splits)


# -------------------- KMM external blocks --------------------

def other_eth_groups(scn: str) -> List[str]:
    scn = scn.lower()
    if scn == "white":    return ["black", "hispanic"]
    if scn == "black":    return ["white", "hispanic"]
    if scn == "hispanic": return ["white", "black"]
    return []

def load_external_groups_for_fold(
    fold: int, scenario: str, splits_dir: str, feat_cols: List[str], cohort: Optional[str],
    X_train_full_raw: np.ndarray, scale_mode: int, icv_train: Optional[np.ndarray] = None
):
    """Return list of (name, df, y, X_scaled) for two out-groups, if available."""
    ext_blocks = []
    for oth in other_eth_groups(scenario):
        p = os.path.join(splits_dir, f"test_{oth}_fold_{fold}.csv")
        if os.path.exists(p):
            dfe, ye = load_split_csv(p, cohort)
            Xe_raw = np.rint(dfe[feat_cols].to_numpy()).astype(np.float32)
            icv_ext = dfe['ICV'].to_numpy(dtype=np.float32) if ('ICV' in dfe.columns and scale_mode == 5) else None
            _, Xe_scaled = ops.scaling_data(X_train_full_raw, Xe_raw, scale_mode,
                                           icv_train=icv_train, icv_test=icv_ext)
            ext_blocks.append((oth, dfe, ye, Xe_scaled))
    return ext_blocks

# -------------------- dual KMM routing --------------------

@dataclass
class DualKMM:
    est1: any
    est2: any
    tgt1: str
    tgt2: str

def route_dual_kmm_predictions(est1, est2, X_fold_scaled, test_df_fold_eth: np.ndarray,
                               in_group_code: str) -> np.ndarray:
    """For the *fold* test, average both estimators on the in-group; others unused here."""
    p1 = est1.predict_proba(X_fold_scaled)[:, 1]
    p2 = est2.predict_proba(X_fold_scaled)[:, 1]
    grp = pd.Series(test_df_fold_eth).map(ETH_MAP).fillna(test_df_fold_eth).str.upper().values
    mask_in = (grp == in_group_code.upper())
    yproba = 0.5 * (p1 + p2)  # average for in-group
    # If there are accidental other-group rows in fold set, keep avg (safe default)
    return yproba

# -------------------- metrics & persistence --------------------

def group_metrics(y_true: np.ndarray, y_prob: np.ndarray, group_labels: np.ndarray, thr: float) -> Dict[str, Dict[str, float]]:
    # Map ETHNICITY to canonical NHW/NHA/HWA, safely handling ndarray inputs.
    ser = pd.Series(group_labels)
    mapped = ser.map(ETH_MAP)
    mapped = mapped.where(~mapped.isna(), ser)   # <- FIX: fill with Series, not ndarray
    grp = mapped.astype(str).str.upper().values

    out = {}
    for key, mask in {"NHW": (grp == "NHW"), "NHA": (grp == "NHA"), "HWA": (grp == "HWA")}.items():
        if mask.sum() == 0:
            out[key] = {k: float('nan') for k in ["ba", "f1", "auc", "acc", "fpr", "fnr", "n"]}
            out[key]["n"] = 0
            continue
        yt = y_true[mask]; yp = y_prob[mask]; yhat = (yp >= thr).astype(int)

        ba = balanced_accuracy_score(yt, yhat)
        f1 = f1_score(yt, yhat)
        try:
            auc = roc_auc_score(yt, yp)
        except Exception:
            auc = float('nan')
        acc = accuracy_score(yt, yhat)
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')

        out[key] = {"ba": float(ba), "f1": float(f1), "auc": float(auc), "acc": float(acc),
                    "fpr": float(fpr), "fnr": float(fnr), "n": int(mask.sum())}
    return out

# def save_predictions(test_df: pd.DataFrame, yproba: np.ndarray, out_dir: str, tag: str):
#     id_col = pick_id_col(test_df)
#     out = pd.DataFrame({
#         "ID": test_df[id_col] if id_col else np.arange(len(test_df)),
#         "ETHNICITY": test_df["ETHNICITY"] if "ETHNICITY" in test_df.columns else "unknown",
#         "DX": test_df["DX"].values,
#         "PRED_PROB": yproba
#     })
#     out.to_csv(os.path.join(out_dir, f"predictions_{tag}.csv"), index=False)

def save_predictions(
        df: pd.DataFrame,
        y_proba: np.ndarray,
        out_dir: str,
        tag: str = "fold",

        id_priority: tuple = ("RID", "ID"),
        eth_col: str = "ETHNICITY",
        label_col: str = "DX",
        append: bool = False,
        filename: Optional[str] = None,
    ) -> str:
    ensure_dir(out_dir)
    n = len(df)
    if len(y_proba) != n:
        raise ValueError(f"save_predictions: y_proba length {len(y_proba)} != df length {n}")

    # COHORT
    if "COHORT" in df.columns:
        cohort_series = df["COHORT"].astype(str).values
    else:
        cohort_series = np.array(["UNKNOWN"] * n)

    # ID
    if id_priority[0] in df.columns:
        id_series = df[id_priority[0]].astype(str).values
    elif id_priority[1] in df.columns:
        id_series = df[id_priority[1]].astype(str).values
    else:
        # fall back to a stable sequential ID
        id_series = np.arange(n).astype(str)

    # ETHNICITY
    eth_series = df[eth_col].astype(str).values if eth_col in df.columns else np.array(["UNKNOWN"] * n)

    # Y_TRUE
    y_true = df[label_col].values if label_col in df.columns else np.full(n, np.nan)

    out_df = pd.DataFrame({
        "COHORT": cohort_series,
        "ID": id_series,
        "ETHNICITY": eth_series,
        "Y_TRUE": y_true,
        "Y_PRED": y_proba.astype(float),
    })

    out_path = os.path.join(out_dir, filename or f"predictions_{tag}.csv")
    write_header = not (append and os.path.exists(out_path))
    out_df.to_csv(out_path, index=False, mode=("a" if append else "w"), header=write_header)
    return out_path

def save_summaries(out_dir: str, suffix: str, fold: int,
                   overall: Dict[str, float], groups: Dict[str, Dict[str, float]], tag: str):
    with open(os.path.join(out_dir, f"summary_metrics_{tag}.json"), "w") as f:
        json.dump({**overall, "suffix": suffix, "fold": fold}, f, indent=2)
    with open(os.path.join(out_dir, f"summary_metrics_groups_{tag}.json"), "w") as f:
        json.dump(groups, f, indent=2)

def save_model(out_dir: str, best_estimator, scaler, scale_mode: int, method: str,
               dual: Optional[DualKMM] = None):
    payload = {"estimator": best_estimator, "scaler": scaler,
               "scale_mode": scale_mode, "method": method}
    if dual is not None:
        payload.update({"estimator_1": dual.est1, "estimator_2": dual.est2,
                        "kmm_targets": [dual.tgt1, dual.tgt2]})
    try:
        import joblib
        joblib.dump(payload, os.path.join(out_dir, "model.pkl"))
    except Exception as e:
        print(f"[WARN] model.pkl save failed: {e}")

# -------------------- Stage-A baseline cache --------------------

def baseline_cache_paths(save_root: str, model_name: str, scenario: str, fold: int) -> Dict[str, str]:
    base_dir = os.path.join(save_root, f"baseline_{model_name}", f"{scenario}", f"fold_{fold:02d}")
    ensure_dir(base_dir)
    return {
        "dir": base_dir,
        "best_params_json": os.path.join(base_dir, "best_params.json"),
        "sensitivity_json": os.path.join(base_dir, "sensitivity.json"),
    }

def try_load_baseline_best_params(save_root: str, model_name: str, scenario: str, fold: int) -> Optional[Dict]:
    paths = baseline_cache_paths(save_root, model_name, scenario, fold)
    if os.path.exists(paths["best_params_json"]):
        with open(paths["best_params_json"], "r") as f:
            data = json.load(f)
        return data.get("best_model_params") or data.get("best_params")
    return None

def save_baseline_best_params(save_root: str, model_name: str, scenario: str, fold: int, best_model_params: Dict):
    paths = baseline_cache_paths(save_root, model_name, scenario, fold)
    payload = {"best_model_params": best_model_params, "fold": fold,
               "scenario": scenario, "model_name": model_name}
    with open(paths["best_params_json"], "w") as f:
        json.dump(payload, f, indent=2)

def try_load_baseline_sensitivity(save_root: str, model_name: str, scenario: str, fold: int) -> Optional[list]:
    paths = baseline_cache_paths(save_root, model_name, scenario, fold)
    if os.path.exists(paths["sensitivity_json"]):
        try:
            with open(paths["sensitivity_json"], "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_baseline_sensitivity(save_root: str, model_name: str, scenario: str, fold: int, records: list):
    paths = baseline_cache_paths(save_root, model_name, scenario, fold)
    with open(paths["sensitivity_json"], "w") as f:
        json.dump(records, f, indent=2)
        
def cost_threshold_from_params(est_or_params):
    """Return t_cost from estimator or params dict. None if not available."""
    cf = cp = None
    if hasattr(est_or_params, "get_params"):
        p = est_or_params.get_params()
        cf = p.get("cs_cost_fn", None)
        cp = p.get("cs_cost_fp", None)
        # CustomML keeps flat params in `paras` as well
        if (cf is None or cp is None) and hasattr(est_or_params, "paras"):
            cf = est_or_params.paras.get("cs_cost_fn", cf)
            cp = est_or_params.paras.get("cs_cost_fp", cp)
    elif isinstance(est_or_params, dict):
        cf = est_or_params.get("cs_cost_fn", None) or est_or_params.get("best_model_params", {}).get("cs_cost_fn")
        cp = est_or_params.get("cs_cost_fp", None) or est_or_params.get("best_model_params", {}).get("cs_cost_fp")
    if cf is None or cp is None or (cf + cp) == 0:
        return None
    return float(cp) / (float(cp) + float(cf))

def choose_threshold_fairness(y_true, y_prob, group_labels, w_tpr=0.5, w_fpr=0.5, lam_ba=0.1):
    """Scan thresholds and return (thr, diagnostics) minimizing
    max_{g,g'} ( w_tpr*|TPR_g-TPR_g'| + w_fpr*|FPR_g-FPR_g'| ) - lam_ba * BA_overall.
    """
    import numpy as _np
    y_true = _np.asarray(y_true).astype(int)
    y_prob = _np.asarray(y_prob).astype(float)
    ser = pd.Series(group_labels); mapped = ser.map(ETH_MAP); mapped = mapped.where(~mapped.isna(), ser)
    groups = mapped.astype(str).str.upper().values

    best = (0.5, float('inf'))
    diag = {}
    for t in _np.linspace(0.05, 0.95, 91):
        yhat = (y_prob >= t).astype(int)
        ba   = balanced_accuracy_score(y_true, yhat)
        # per-group metrics
        by = group_metrics(y_true, y_prob, groups, t)
        # compute disparity term
        tprs, fprs = [], []
        for k, vals in by.items():
            # reconstruct TPR from fnr and pos rate if needed (approx from confusion)
            # we can recompute precisely here:
            mask = (groups == k)
            if mask.sum() == 0:
                continue
            yt = y_true[mask]; yp = (y_prob[mask] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tprs.append(tpr); fprs.append(fpr)
        if len(tprs) >= 2:
            gap_tpr = max(tprs) - min(tprs)
            gap_fpr = max(fprs) - min(fprs)
            obj = (w_tpr * gap_tpr + w_fpr * gap_fpr) - lam_ba * ba
        else:
            obj = 1.0 - lam_ba * ba
        if obj < best[1]:
            best = (float(t), float(obj))
            diag = {"ba": float(ba), "gap_tpr": float(gap_tpr if len(tprs)>=2 else 0.0), "gap_fpr": float(gap_fpr if len(fprs)>=2 else 0.0)}
    return best[0], diag
