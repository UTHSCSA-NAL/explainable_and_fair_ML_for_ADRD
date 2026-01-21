#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = str(ROOT / 'data_splits' / 'train.csv')
DEFAULT_TEST_CSV = str(ROOT / 'data_splits' / 'test.csv')
DEFAULT_DATA_CSV = None
DEFAULT_RESULTS_ROOT = str(ROOT / 'runs')
DEFAULT_OUTPUT_ROOT = str(ROOT / 'runs' / 'shap_small')

from dementia_ai.core import ops
from dementia_ai.core import train_utils as U
from dementia_ai.explainer import shap_utils as CSU


def _parse_run_indices(run_range: Optional[str], max_runs: int) -> List[int]:
    if not run_range:
        return list(range(max_runs))
    run_range = run_range.strip()
    if "-" in run_range:
        a, b = run_range.split("-", 1)
        start, end = int(a), int(b)
        if end < start:
            raise ValueError("run_range end must be >= start")
        return list(range(start, end + 1))
    return [int(run_range)]


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = U.select_feature_columns(df)
    return [c for c in cols if not c.lower().startswith("dx")]


def _match_id_column(df: pd.DataFrame, ids: pd.Series) -> str:
    cand_cols = [c for c in ("RID", "PTID", "MRI_ID", "ID") if c in df.columns]
    ids_str = ids.astype(str)
    best_col, best_n = None, -1
    for col in cand_cols:
        n = int(df[col].astype(str).isin(ids_str).sum())
        if n > best_n:
            best_col, best_n = col, n
    if best_col is None or best_n == 0:
        raise ValueError("Could not match prediction IDs to any of RID/PTID/MRI_ID/ID")
    return best_col


def _unwrap_estimator(pkg):
    est = pkg.get("estimator", pkg) if isinstance(pkg, dict) else pkg
    base = est
    if hasattr(est, "predictor"):
        base = est.predictor
    elif hasattr(est, "booster_"):
        base = est.booster_
    return est, base


def _scale_with_saved(raw_bg: np.ndarray, raw_te: np.ndarray, scaler, scale_mode: int,
                      icv_bg: Optional[np.ndarray], icv_te: Optional[np.ndarray]):
    if scaler is not None:
        bg_scaled = ops._apply_same_scaling(raw_bg, scaler, scale_mode, icv=icv_bg)
        te_scaled = ops._apply_same_scaling(raw_te, scaler, scale_mode, icv=icv_te)
        return bg_scaled, te_scaled
    # fallback: fit on background, transform test
    bg_scaled, te_scaled = ops.scaling_data(
        raw_bg, raw_te, scaling_mode=scale_mode, return_scaler=False,
        icv_train=icv_bg, icv_test=icv_te
    )
    return bg_scaled, te_scaled


def _make_explainer(base_model, X_bg, model_output: str):
    """
    Prefer GPU SHAP if available; otherwise fall back to CPU TreeExplainer.
    """
    try:
        return shap.GPUTreeExplainer(base_model, data=X_bg, model_output=model_output)
    except Exception as e:
        print(f"[WARN] GPU SHAP unavailable ({e}); falling back to TreeExplainer on CPU.")
        return shap.TreeExplainer(
            base_model,
            data=X_bg,
            model_output=model_output,
            feature_perturbation="tree_path_dependent",
        )


def _compute_custom_shap(estimator, X_bg, X_te, device: str, filter_idx: Optional[List[int]] = None,
                         internal_batch_size: int = 100):
    """
    Use custom Torch Tree SHAP implementation (batch) from dementia_ai.explainer.shap_utils.
    """
    model = estimator if hasattr(estimator, "predictor") else estimator
    if hasattr(model, "booster_"):
        model = model.booster_
    predict_fn = CSU.get_xgboost_predictor(model, internal_batch_size=internal_batch_size)
    cols = list(range(X_te.shape[1])) if filter_idx is None else filter_idx
    return CSU.custom_tree_shap_batch_torch(predict_fn, X_te, background_data=X_bg, device=device, filter_column_idx=cols)


def compute_one_run(run_idx: int, args, df_full: pd.DataFrame):
    tag = args.method_tag
    run_dir = os.path.join(args.results_root, "train_small", tag, args.scenario, f"run_{run_idx:03d}")
    model_path = os.path.join(run_dir, "model.pkl")
    pred_path = os.path.join(run_dir, "predictions_rep.csv")
    if not os.path.exists(model_path) or not os.path.exists(pred_path):
        print(f"[WARN] missing model or predictions for run {run_idx}, skipping.")
        return None

    payload = joblib.load(model_path)
    est_saved, base_model = _unwrap_estimator(payload)
    scale_mode = payload.get("scale_mode", args.scale_mode) if isinstance(payload, dict) else args.scale_mode
    scaler = payload.get("scaler", None) if isinstance(payload, dict) else None

    preds = pd.read_csv(pred_path)
    id_col_pred = "MRI_ID" if "MRI_ID" in preds.columns else "ID"
    test_ids = preds[id_col_pred].astype(str)

    feat_cols = _feature_columns(df_full)
    id_col = _match_id_column(df_full, test_ids)

    df_test = df_full[df_full[id_col].astype(str).isin(test_ids)].reset_index(drop=True)
    df_train_bg = df_full[~df_full[id_col].astype(str).isin(test_ids)].reset_index(drop=True)

    if df_test.empty:
        print(f"[WARN] no test rows matched for run {run_idx}, skipping.")
        return None

    # background sample
    if len(df_train_bg) > args.num_background:
        df_bg = df_train_bg.sample(n=args.num_background, random_state=args.random_state + run_idx)
    else:
        df_bg = df_train_bg

    X_bg_raw = np.rint(df_bg[feat_cols].to_numpy()).astype(np.float32)
    X_te_raw = np.rint(df_test[feat_cols].to_numpy()).astype(np.float32)
    y_test = df_test["DX"].map({"CN": 0, "AD": 1, 0: 0, 1: 1}).astype(int).values
    icv_bg = df_bg["ICV"].to_numpy(dtype=np.float32) if "ICV" in df_bg.columns else None
    icv_te = df_test["ICV"].to_numpy(dtype=np.float32) if "ICV" in df_test.columns else None
    X_bg, X_te = _scale_with_saved(X_bg_raw, X_te_raw, scaler, scale_mode, icv_bg, icv_te)
    X_bg = np.asarray(X_bg, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    if args.shap_method == "custom":
        shap_vals = _compute_custom_shap(
            base_model, X_bg, X_te, device=args.device,
            internal_batch_size=args.internal_batch_size
        )
    else:
        explainer = _make_explainer(base_model, X_bg, args.model_output)
        if args.model_output == "log_loss":
            shap_vals = explainer.shap_values(X_te, y=y_test, check_additivity=False)
        else:
            shap_vals = explainer.shap_values(X_te, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = np.asarray(shap_vals)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[1] if shap_vals.shape[0] > 1 else shap_vals[0]
        if shap_vals.shape[1] == len(feat_cols) + 1:
            shap_vals = shap_vals[:, :-1]  # drop bias term

    out_df = pd.DataFrame(shap_vals, columns=feat_cols)
    out_df.insert(0, "DX", df_test["DX"].values)
    out_df.insert(0, "ETHNICITY", df_test["ETHNICITY"].values if "ETHNICITY" in df_test.columns else "unknown")
    out_df.insert(0, "MRI_ID", df_test["MRI_ID"].values if "MRI_ID" in df_test.columns else df_test[id_col].astype(str).values)
    out_df.insert(0, "run_index", run_idx)
    return out_df


def main(argv=None):
    ap = argparse.ArgumentParser("Compute SHAP for trainer_small runs")
    ap.add_argument("--scenario", required=True, choices=["all", "white", "black", "hispanic"])
    ap.add_argument("--method_tag", required=True, help="Folder tag under train_small, e.g., baseline_xgboost_vanilla or regalign_xgboost_vanilla_ce+align+gap")
    ap.add_argument("--data_csv", default=DEFAULT_DATA_CSV)
    ap.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--run_range", default=None, help="Inclusive range like 0-10; if omitted, uses 0..n_repeats-1")
    ap.add_argument("--n_repeats", type=int, default=100)
    ap.add_argument("--num_background", type=int, default=256)
    ap.add_argument("--scale_mode", type=int, default=0, help="Scaling mode to apply if scaler not bundled in model.pkl")
    ap.add_argument("--model_output", default="raw", help="GPUTreeExplainer model_output (e.g., margin, raw, probability, log_loss)")
    ap.add_argument("--shap_method", choices=["gpu", "cpu", "custom"], default="gpu",
                    help="gpu=GPUTreeExplainer, cpu=TreeExplainer, custom=custom_tree_shap_batch_torch")
    ap.add_argument("--device", default="cuda", help="Device for custom SHAP (e.g., cuda or cpu)")
    ap.add_argument("--internal_batch_size", type=int, default=100, help="Internal batch size for custom SHAP predictor")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args(argv)

    run_indices = _parse_run_indices(args.run_range, args.n_repeats)

    if args.data_csv:
        df_full = pd.read_csv(args.data_csv, low_memory=False)
    else:
        df_train = pd.read_csv(args.train_csv, low_memory=False)
        df_test = pd.read_csv(args.test_csv, low_memory=False)
        df_full = pd.concat([df_train, df_test], ignore_index=True)
    os.makedirs(args.output_root, exist_ok=True)
    out_path = os.path.join(args.output_root, f"shap_{args.scenario}.csv")

    collected = []
    for r in tqdm(run_indices, desc="runs"):
        res = compute_one_run(r, args, df_full)
        if res is not None:
            collected.append(res)
    if not collected:
        print("[WARN] No SHAP values computed.")
        return 0
    out_df = pd.concat(collected, ignore_index=True)
    write_header = not os.path.exists(out_path)
    out_df.to_csv(out_path, index=False, mode="a" if not write_header else "w", header=write_header)
    print(f"[INFO] saved SHAP rows={len(out_df)} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
