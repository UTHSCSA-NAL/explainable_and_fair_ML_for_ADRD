#!/usr/bin/env python3
from __future__ import annotations

"""
Prefit KMM weights on the small balanced split used by trainer_small.
This mirrors scripts/prefit_kmm.py but adds conveniences for a small dataset:
- optional per-group subsampling to keep runtime low
- optional explicit param_ids; otherwise uses a compact default grid.
Outputs per-scenario weight tables under --scenario_output_root for reuse by trainer_small.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from dementia_ai.core import ops
from dementia_ai.core import train_utils as U

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = str(ROOT / 'data_splits' / 'train.csv')


def _prepare_features(df: pd.DataFrame, scale_mode: int, icv_col: str) -> Tuple[np.ndarray, pd.Series, pd.Series, str]:
    feat_cols = U.select_feature_columns(df)
    X_raw = np.rint(df[feat_cols].to_numpy()).astype(np.float32)
    icv = df[icv_col].to_numpy(dtype=np.float32) if (scale_mode == 5 and icv_col in df.columns) else None
    if scale_mode == 5 and icv is None:
        raise ValueError(f"scale_mode=5 requires '{icv_col}' column in the dataset")
    X_scaled, scaler = ops.scaling_data(X_raw, scaling_mode=scale_mode, return_scaler=True, icv_train=icv)
    id_col = U.id_column(df)
    if id_col is None:
        raise ValueError("Cannot locate an ID column (MRI_ID/RID/PTID/etc.) in the dataset")
    eth_series = U.canonical_ethnicity(df["ETHNICITY"] if "ETHNICITY" in df.columns else pd.Series(["other"] * len(df)))
    return X_scaled, eth_series, df[id_col].reset_index(drop=True), id_col


def _maybe_subsample(df: pd.DataFrame, per_group: Optional[int], seed: int) -> pd.DataFrame:
    if per_group is None or per_group <= 0:
        return df
    out = []
    rng = np.random.default_rng(seed)
    for eth, grp in df.groupby("ETHNICITY"):
        if len(grp) <= per_group:
            out.append(grp)
        else:
            idx = rng.choice(len(grp), size=per_group, replace=False)
            out.append(grp.iloc[idx])
    return pd.concat(out).reset_index(drop=True)


def _fit_kmm(Xs: np.ndarray, Xt: np.ndarray, params: Dict):
    """Try ADAPT KMM; fallback to custom ops.compute_sample_weight_kmm."""
    gamma = params.get("kmm_gamma")
    if isinstance(gamma, str) and gamma.lower() == "median":
        gamma_val = U._median_gamma(Xs)
    else:
        gamma_val = gamma

    try:
        from adapt.instance_based import KMM as AdaptKMM  # type: ignore

        kmm = AdaptKMM(
            Xt=Xt,
            kernel=params.get("kmm_kernel", "rbf"),
            B=float(params.get("kmm_B", 1000.0)),
            eps=params.get("kmm_eps"),
            max_size=int(params.get("kmm_max_size", 1000)),
            gamma=gamma_val,
            verbose=0,
        )
        kmm.fit(Xs, np.zeros(len(Xs)))
        w = getattr(kmm, "sample_weights_", None)
        if w is None:
            try:
                w = kmm.predict_weights(Xs)
            except TypeError:
                w = kmm.predict_weights()
            except Exception:
                w = None
        if w is not None:
            return np.asarray(w, dtype=float), kmm
    except Exception as e:
        print(f"[WARN] ADAPT KMM failed ({e}); falling back to ops.compute_sample_weight_kmm")

    try:
        w = ops.compute_sample_weight_kmm(
            Xs, Xt,
            kernel=params.get("kmm_kernel", "rbf"),
            B=float(params.get("kmm_B", 1000.0)),
            eps=params.get("kmm_eps"),
            gamma=gamma_val,
            max_size=int(params.get("kmm_max_size", 1000)),
        )
        return np.asarray(w, dtype=float), None
    except Exception as e:
        print(f"[WARN] Custom KMM fallback failed: {e}")
        return None, None


def _compute_pair_weights(
    X_scaled: np.ndarray,
    eth: pd.Series,
    ids: pd.Series,
    src: str,
    tgt: str,
    params: Dict,
    save_model_dir: str,
) -> pd.DataFrame | None:
    mask_src = (eth.str.lower() == src.lower()).values
    mask_tgt = (eth.str.lower() == tgt.lower()).values
    if not mask_src.any() or not mask_tgt.any():
        return None

    Xs = X_scaled[mask_src]
    Xt = X_scaled[mask_tgt]

    w, kmm_model = _fit_kmm(Xs, Xt, params)
    if w is None:
        print(f"[WARN] No weights for {src}->{tgt} (mask sizes src={mask_src.sum()}, tgt={mask_tgt.sum()}); skipping.")
        return None

    w = np.asarray(w, dtype=float) * float(params.get("kmm_scale", 1.0))
    col = U.kmm_column_name(src, tgt)

    # persist model
    if kmm_model is not None:
        try:
            import joblib
            U.ensure_dir(save_model_dir)
            joblib.dump(kmm_model, os.path.join(save_model_dir, "model.joblib"))
        except Exception as e:
            print(f"[WARN] Failed to save KMM model for {src}->{tgt}: {e}")

    return pd.DataFrame({ids.name: ids[mask_src].values, col: w})


def _build_scenario_table(base_tables: Dict[str, pd.DataFrame], id_series: pd.Series, scenario: str) -> pd.DataFrame:
    col_map = U.kmm_columns_for_scenario(scenario)
    df = pd.DataFrame({id_series.name: id_series.values})
    for col_name, group_key in col_map.items():
        tbl = base_tables.get(group_key)
        if tbl is None:
            df[col_name] = np.nan
            continue
        weight_col = [c for c in tbl.columns if c != id_series.name][0]
        df = df.merge(tbl[[id_series.name, weight_col]], how="left", on=id_series.name)
        df.rename(columns={weight_col: col_name}, inplace=True)
    return df


def _default_small_grid() -> List[Dict]:
    """
    Deterministic subset (100 combos) from the full KMM grid in train_utils.method_param_grid(\"kmm\").
    """
    full_grid = list(ParameterGrid(U.method_param_grid("kmm")))
    if len(full_grid) <= 100:
        return full_grid
    idxs = np.linspace(0, len(full_grid) - 1, 100).astype(int)
    return [full_grid[i] for i in idxs]


def _parse_param_ids(param_ids: Optional[str]) -> Optional[List[str]]:
    if not param_ids:
        return None
    return [p.strip() for p in param_ids.split(",") if p.strip()]


def main():
    ap = argparse.ArgumentParser("Prefit KMM weights for the small balanced subset.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="Balanced dataset CSV used in trainer_small.")
    ap.add_argument("--output_root", default="./checkpoints/kmm_small", help="Root for pair-wise KMM weight files.")
    ap.add_argument("--scenario_output_root", default="./checkpoints/kmm_weights_small", help="Root for scenario-level merged weights.")
    ap.add_argument("--scale_mode", type=int, default=0, help="Scaling mode to apply before KMM.")
    ap.add_argument("--icv_col", type=str, default="ICV", help="ICV column name when scale_mode=5.")
    ap.add_argument("--scenarios", type=str, default="all,white+black,white+hispanic,black+hispanic,white,black,hispanic")
    ap.add_argument("--param_ids", type=str, default=None, help="Optional comma list of param_ids to run (must exist in kmm grid).")
    ap.add_argument("--subsample_per_group", type=int, default=None, help="Optional max samples per ethnicity to speed up.")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    df = pd.read_csv(args.dataset, low_memory=False)
    if args.subsample_per_group:
        df = _maybe_subsample(df, args.subsample_per_group, seed=42)
        print(f"[INFO] Subsampled dataset to per-group <= {args.subsample_per_group}; new size={len(df)}")

    X_scaled, eth, ids, id_col = _prepare_features(df, args.scale_mode, args.icv_col)
    grp_counts = eth.value_counts().to_dict()
    print(f"[INFO] ETHNICITY counts (normalized): {grp_counts}")

    full_grid = U.kmm_param_grid_list()
    if args.param_ids:
        keep = set(_parse_param_ids(args.param_ids))
        kmm_grid = [p for p in full_grid if U.kmm_param_id(p) in keep]
    else:
        kmm_grid = _default_small_grid()

    print(f"[INFO] Prefitting {len(kmm_grid)} KMM combos across {len(U.KMM_GROUP_PAIRS)} group pairs")
    success_pairs = 0
    failed_pairs = 0
    for params in tqdm(kmm_grid, desc="param combos"):
        param_id = U.kmm_param_id(params)
        base_tables: Dict[str, pd.DataFrame] = {}

        for src, tgt in U.KMM_GROUP_PAIRS:
            group_key = U.kmm_group_key(src, tgt)
            model_dir = os.path.join(args.output_root, group_key, param_id)
            res = _compute_pair_weights(X_scaled, eth, ids, src, tgt, params, model_dir)
            if res is None:
                failed_pairs += 1
                continue
            success_pairs += 1
            base_tables[group_key] = res
            out_dir = os.path.join(args.output_root, group_key)
            U.ensure_dir(out_dir)
            res.to_csv(os.path.join(out_dir, f"{param_id}.csv"), index=False)

        if not base_tables:
            print(f"[WARN] No KMM weights produced for param {param_id}; skipping scenario tables.")
            continue

        for scn in scenarios:
            scn_df = _build_scenario_table(base_tables, ids, scn)
            scn_dir = os.path.join(args.scenario_output_root, scn.lower())
            U.ensure_dir(scn_dir)
            scn_df.to_csv(os.path.join(scn_dir, f"{param_id}.csv"), index=False)
            if scn_df.drop(columns=[ids.name]).notna().sum().sum() == 0:
                print(f"[WARN] Scenario {scn} param {param_id}: all weights NaN; check source/target groups.")

    meta = {
        "dataset": os.path.abspath(args.dataset),
        "output_root": os.path.abspath(args.output_root),
        "scenario_output_root": os.path.abspath(args.scenario_output_root),
        "scale_mode": args.scale_mode,
        "icv_col": args.icv_col,
        "scenarios": scenarios,
        "total_param_combos": len(kmm_grid),
        "group_pairs": U.KMM_GROUP_PAIRS,
        "success_pairs": success_pairs,
        "failed_pairs": failed_pairs,
        "subsample_per_group": args.subsample_per_group,
    }
    U.ensure_dir(args.output_root)
    with open(os.path.join(args.output_root, "kmm_prefit_meta_small.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] Prefit metadata written. success_pairs={success_pairs}, failed_pairs={failed_pairs}")


if __name__ == "__main__":
    main()
