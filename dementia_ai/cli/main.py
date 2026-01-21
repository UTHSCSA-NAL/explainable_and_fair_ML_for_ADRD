from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from dementia_ai.core import ops
from dementia_ai.core import train_utils as U
from dementia_ai.training import trainer_small as TRS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def _load_model_payload(model_path: str):
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "estimator" in payload:
        estimator = payload["estimator"]
        scaler = payload.get("scaler")
        scale_mode = payload.get("scale_mode")
    else:
        estimator = payload
        scaler = None
        scale_mode = None
    return estimator, scaler, scale_mode


def _load_meta_features(model_path: str) -> Optional[List[str]]:
    meta_path = Path(model_path).with_name("meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r") as f:
            meta = json.load(f)
        feats = meta.get("feature_names")
        return feats if isinstance(feats, list) else None
    except Exception:
        return None


def _align_features(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    data = {}
    for col in feat_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            data[col] = series.fillna(series.mean()).to_numpy(dtype=float)
        else:
            data[col] = np.zeros(len(df), dtype=float)
    return np.rint(pd.DataFrame(data).to_numpy()).astype(np.float32)


def _prepare_features(df: pd.DataFrame, feat_cols: List[str], scaler, scale_mode: int) -> np.ndarray:
    X_raw = _align_features(df, feat_cols)
    icv = None
    if scale_mode == 5:
        if "ICV" not in df.columns:
            raise ValueError("scale_mode=5 requires an ICV column")
        icv = df["ICV"].to_numpy(dtype=np.float32)
    if scaler is None:
        X_scaled = ops.scaling_data(X_raw, scaling_mode=scale_mode, return_scaler=False, icv_train=icv)
    else:
        X_scaled = ops._apply_same_scaling(X_raw, scaler, scale_mode, icv=icv)
    return np.asarray(X_scaled, dtype=np.float32)


def _predict_scores(estimator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_prob = None
    y_score = None

    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            proba = proba[:, None]
        y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    if hasattr(estimator, "decision_function"):
        try:
            y_score = estimator.decision_function(X)
        except Exception:
            y_score = None

    if y_score is None and hasattr(estimator, "predictor"):
        try:
            import xgboost as xgb

            predictor = estimator.predictor
            if hasattr(predictor, "get_booster"):
                booster = predictor.get_booster()
                y_score = booster.predict(xgb.DMatrix(X), output_margin=True)
        except Exception:
            y_score = None

    if y_prob is None and y_score is not None:
        y_prob = _sigmoid(np.asarray(y_score, dtype=float))
    if y_score is None and y_prob is not None:
        y_score = _logit(np.asarray(y_prob, dtype=float))

    if y_prob is None or y_score is None:
        raise ValueError("Model does not expose probability or score outputs")

    return np.asarray(y_score, dtype=float), np.asarray(y_prob, dtype=float)


def cmd_train(argv: List[str]) -> int:
    args = TRS.parse_args(argv)
    summary = TRS.run(args)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_predict(argv: List[str]) -> int:
    ap = argparse.ArgumentParser("Predict labels for a CSV")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--scale_mode", type=int, default=0, help="Fallback scale mode if model payload lacks it")
    ap.add_argument("--feature_cols", default=None, help="Optional path to a txt file with feature columns")
    args = ap.parse_args(argv)

    estimator, scaler, scale_mode = _load_model_payload(args.model_path)
    scale_mode = args.scale_mode if scale_mode is None else int(scale_mode)

    df = pd.read_csv(args.input_csv, low_memory=False)

    feat_cols = None
    if args.feature_cols:
        feat_cols = [c.strip() for c in Path(args.feature_cols).read_text().splitlines() if c.strip()]
    if feat_cols is None:
        feat_cols = _load_meta_features(args.model_path)
    if feat_cols is None:
        feat_cols = U.select_feature_columns(df)

    X = _prepare_features(df, feat_cols, scaler, scale_mode)
    y_score, y_prob = _predict_scores(estimator, X)

    out_df = df.copy()
    out_df["y_score"] = y_score
    out_df["y_prob"] = y_prob

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        in_path = Path(args.input_csv)
        out_path = in_path.with_name(f"{in_path.stem}_pred.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return 0


def _normalize_labels(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        return series.map(U.DX_MAP).astype(int).to_numpy()
    return series.astype(int).to_numpy()


def _group_metrics(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray, thr: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    group_vals = pd.Series(groups).astype(str).fillna("unknown")
    for g in sorted(group_vals.unique()):
        mask = (group_vals == g).values
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        yhat = (yp >= thr).astype(int)

        ba = balanced_accuracy_score(yt, yhat)
        f1 = f1_score(yt, yhat, zero_division=0)
        try:
            auc = roc_auc_score(yt, yp)
        except Exception:
            auc = float("nan")
        acc = accuracy_score(yt, yhat)
        tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        out[g] = {
            "ba": float(ba),
            "f1": float(f1),
            "auc": float(auc),
            "acc": float(acc),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "n": int(mask.sum()),
        }
    return out


def cmd_evaluate(argv: List[str]) -> int:
    ap = argparse.ArgumentParser("Evaluate group metrics and gaps")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--label_col", default=None, help="Column with ground-truth labels (default: DX or y_true)")
    ap.add_argument("--group_col", default="ETHNICITY")
    ap.add_argument("--score_col", default=None, help="Column with probabilities or scores (default: y_prob or y_score)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.data_csv, low_memory=False)
    label_col = args.label_col
    if label_col is None:
        label_col = "y_true" if "y_true" in df.columns else "DX"
    score_col = args.score_col
    if score_col is None:
        score_col = "y_prob" if "y_prob" in df.columns else "y_score"

    if label_col not in df.columns:
        raise ValueError(f"label column not found: {label_col}")
    if score_col not in df.columns:
        raise ValueError(f"score column not found: {score_col}")
    if args.group_col not in df.columns:
        raise ValueError(f"group column not found: {args.group_col}")

    y_true = _normalize_labels(df[label_col])
    scores = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    if score_col == "y_score":
        y_prob = _sigmoid(scores)
    else:
        y_prob = scores

    groups = df[args.group_col].astype(str).values
    group_metrics = _group_metrics(y_true, y_prob, groups, args.threshold)

    # Pairwise gaps
    group_names = sorted(group_metrics.keys())
    pairwise = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            fnr_gap = abs(group_metrics[g1]["fnr"] - group_metrics[g2]["fnr"])
            fpr_gap = abs(group_metrics[g1]["fpr"] - group_metrics[g2]["fpr"])
            pairwise.append({
                "group_a": g1,
                "group_b": g2,
                "fnr_gap": float(fnr_gap),
                "fpr_gap": float(fpr_gap),
            })

    ba_vals = [m["ba"] for m in group_metrics.values() if math.isfinite(m["ba"])]
    fnr_gaps = [p["fnr_gap"] for p in pairwise if math.isfinite(p["fnr_gap"])]
    fpr_gaps = [p["fpr_gap"] for p in pairwise if math.isfinite(p["fpr_gap"])]

    summary = {
        "group_metrics": group_metrics,
        "pairwise_gaps": pairwise,
        "mean_ba": float(np.mean(ba_vals)) if ba_vals else float("nan"),
        "mean_fnr_gap": float(np.mean(fnr_gaps)) if fnr_gaps else float("nan"),
        "mean_fpr_gap": float(np.mean(fpr_gaps)) if fpr_gaps else float("nan"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _parse_pairs(pairs_str: Optional[str]) -> List[Tuple[str, str]]:
    if not pairs_str:
        return [(a, b) for a, b in U.KMM_GROUP_PAIRS]
    pairs = []
    for item in pairs_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" in item:
            src, tgt = item.split("->", 1)
        elif ">" in item:
            src, tgt = item.split(">", 1)
        else:
            raise ValueError(f"Invalid pair spec: {item}")
        pairs.append((src.strip(), tgt.strip()))
    return pairs


def _fit_kmm(Xs: np.ndarray, Xt: np.ndarray, params: Dict):
    gamma = params.get("kmm_gamma")
    if isinstance(gamma, str) and gamma.lower() == "median":
        gamma_val = U._median_gamma(Xs)
    else:
        gamma_val = gamma

    try:
        from adapt.instance_based import KMM as AdaptKMM

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
            w = kmm.predict_weights(Xs)
        return np.asarray(w, dtype=float), kmm
    except Exception as e:
        print(f"[WARN] ADAPT KMM failed ({e}); falling back to ops.compute_sample_weight_kmm")

    try:
        w = ops.compute_sample_weight_kmm(
            Xs,
            Xt,
            kernel=params.get("kmm_kernel", "rbf"),
            B=float(params.get("kmm_B", 1000.0)),
            eps=params.get("kmm_eps"),
            gamma=gamma_val,
            max_size=int(params.get("kmm_max_size", 1000)),
        )
        return np.asarray(w, dtype=float), None
    except Exception as e:
        print(f"[WARN] Custom KMM failed: {e}")
        return None, None


def _kmm_params_from_config() -> Dict:
    cfg = U._load_configs().get("defaults", {})
    kmm = cfg.get("kmm_defaults", {})
    return {
        "kmm_kernel": kmm.get("kernel", "rbf"),
        "kmm_B": kmm.get("B", 1000.0),
        "kmm_eps": kmm.get("eps", None),
        "kmm_max_size": kmm.get("max_size", 1000),
        "kmm_gamma": kmm.get("gamma", None),
        "kmm_clip_min": kmm.get("clip_min", 0.0),
        "kmm_clip_max": kmm.get("clip_max", 50.0),
        "kmm_scale": 1.0,
    }


def cmd_prefit_kmm(argv: List[str]) -> int:
    ap = argparse.ArgumentParser("Prefit or apply KMM weights")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--data_csv", required=True)
    ap_train.add_argument("--group_col", default="ETHNICITY")
    ap_train.add_argument("--pairs", default=None, help="Comma list of src->tgt pairs")
    ap_train.add_argument("--output_root", default="./checkpoints/kmm_models")
    ap_train.add_argument("--scale_mode", type=int, default=0)
    ap_train.add_argument("--icv_col", default="ICV")
    ap_train.add_argument("--param_id", default=None)
    ap_train.add_argument("--kmm_kernel", default=None)
    ap_train.add_argument("--kmm_B", type=float, default=None)
    ap_train.add_argument("--kmm_eps", type=float, default=None)
    ap_train.add_argument("--kmm_max_size", type=int, default=None)
    ap_train.add_argument("--kmm_gamma", default=None)
    ap_train.add_argument("--kmm_clip_min", type=float, default=None)
    ap_train.add_argument("--kmm_clip_max", type=float, default=None)
    ap_train.add_argument("--kmm_scale", type=float, default=None)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--data_csv", required=True)
    ap_apply.add_argument("--group_col", default="ETHNICITY")
    ap_apply.add_argument("--pairs", default=None, help="Comma list of src->tgt pairs")
    ap_apply.add_argument("--models_root", default="./checkpoints/kmm_models")
    ap_apply.add_argument("--param_id", required=True)
    ap_apply.add_argument("--output_csv", required=True)

    args = ap.parse_args(argv)

    if args.mode == "train":
        df = pd.read_csv(args.data_csv, low_memory=False)
        feat_cols = U.select_feature_columns(df)
        X_raw = np.rint(df[feat_cols].to_numpy()).astype(np.float32)
        icv = None
        if args.scale_mode == 5:
            if args.icv_col not in df.columns:
                raise ValueError(f"scale_mode=5 requires {args.icv_col} column")
            icv = df[args.icv_col].to_numpy(dtype=np.float32)
        X_scaled, scaler = ops.scaling_data(X_raw, scaling_mode=args.scale_mode, return_scaler=True, icv_train=icv)

        params = _kmm_params_from_config()
        for key in ["kmm_kernel", "kmm_B", "kmm_eps", "kmm_max_size", "kmm_gamma", "kmm_clip_min", "kmm_clip_max", "kmm_scale"]:
            val = getattr(args, key)
            if val is not None:
                params[key] = val

        param_id = args.param_id or U.kmm_param_id(params)
        pairs = _parse_pairs(args.pairs)

        id_col = U.id_column(df) or "ID"
        ids = df[id_col].astype(str) if id_col in df.columns else pd.Series(df.index.astype(str))
        groups = df[args.group_col].astype(str)

        for src, tgt in pairs:
            mask_src = groups.str.lower() == src.lower()
            mask_tgt = groups.str.lower() == tgt.lower()
            if not mask_src.any() or not mask_tgt.any():
                print(f"[WARN] Skipping {src}->{tgt}: missing source or target rows")
                continue
            Xs = X_scaled[mask_src.values]
            Xt = X_scaled[mask_tgt.values]
            w, kmm_model = _fit_kmm(Xs, Xt, params)
            if w is None:
                continue
            w = np.asarray(w, dtype=float) * float(params.get("kmm_scale", 1.0))
            clip_min = params.get("kmm_clip_min")
            clip_max = params.get("kmm_clip_max")
            if clip_min is not None or clip_max is not None:
                w = np.clip(w, clip_min if clip_min is not None else w.min(), clip_max if clip_max is not None else w.max())

            group_key = U.kmm_group_key(src, tgt)
            out_dir = Path(args.output_root) / group_key / param_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_weights = pd.DataFrame({id_col: ids[mask_src].values, U.kmm_column_name(src, tgt): w})
            weights_path = out_dir.parent / f"{param_id}.csv"
            out_weights.to_csv(weights_path, index=False)

            payload = {
                "kmm_model": kmm_model,
                "scaler": scaler,
                "scale_mode": args.scale_mode,
                "feature_names": feat_cols,
                "target_X": Xt,
                "params": params,
            }
            joblib.dump(payload, out_dir / "model.joblib")
        return 0

    # apply mode
    df = pd.read_csv(args.data_csv, low_memory=False)
    pairs = _parse_pairs(args.pairs)
    id_col = U.id_column(df) or "ID"
    ids = df[id_col].astype(str) if id_col in df.columns else pd.Series(df.index.astype(str))
    groups = df[args.group_col].astype(str)

    out_df = pd.DataFrame({id_col: ids})
    for src, tgt in pairs:
        group_key = U.kmm_group_key(src, tgt)
        model_path = Path(args.models_root) / group_key / args.param_id / "model.joblib"
        if not model_path.exists():
            print(f"[WARN] Missing model: {model_path}")
            out_df[U.kmm_column_name(src, tgt)] = np.nan
            continue
        payload = joblib.load(model_path)
        kmm_model = payload.get("kmm_model") if isinstance(payload, dict) else payload
        scaler = payload.get("scaler") if isinstance(payload, dict) else None
        scale_mode = payload.get("scale_mode", 0) if isinstance(payload, dict) else 0
        feat_cols = payload.get("feature_names") if isinstance(payload, dict) else None
        params = payload.get("params", {}) if isinstance(payload, dict) else {}
        target_X = payload.get("target_X") if isinstance(payload, dict) else None

        if feat_cols is None:
            feat_cols = U.select_feature_columns(df)
        X_raw = _align_features(df, feat_cols)
        icv = None
        if scale_mode == 5 and "ICV" in df.columns:
            icv = df["ICV"].to_numpy(dtype=np.float32)
        if scaler is None:
            X_scaled = ops.scaling_data(X_raw, scaling_mode=scale_mode, return_scaler=False, icv_train=icv)
        else:
            X_scaled = ops._apply_same_scaling(X_raw, scaler, scale_mode, icv=icv)

        mask_src = groups.str.lower() == src.lower()
        if not mask_src.any():
            out_df[U.kmm_column_name(src, tgt)] = np.nan
            continue
        Xs = X_scaled[mask_src.values]
        if kmm_model is not None and hasattr(kmm_model, "predict_weights"):
            w = kmm_model.predict_weights(Xs)
        elif target_X is not None:
            w = ops.compute_sample_weight_kmm(
                Xs,
                target_X,
                kernel=params.get("kmm_kernel", "rbf"),
                B=float(params.get("kmm_B", 1000.0)),
                eps=params.get("kmm_eps"),
                gamma=params.get("kmm_gamma"),
                max_size=int(params.get("kmm_max_size", 1000)),
            )
        else:
            print(f"[WARN] No usable model for {src}->{tgt}")
            w = np.full(mask_src.sum(), np.nan)

        w = np.asarray(w, dtype=float)
        w_full = np.full(len(df), np.nan)
        w_full[mask_src.values] = w
        out_df[U.kmm_column_name(src, tgt)] = w_full

    out_df.to_csv(args.output_csv, index=False)
    print(f"[OK] wrote {args.output_csv}")
    return 0


def _run_script(script_name: str, subargv: List[str]) -> int:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return 1
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *subargv]
        import runpy
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


def cmd_compute_shap(argv: List[str]) -> int:
    return _run_script("compute_shap_small.py", argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Usage: dai <train|predict|evaluate|prefit_kmm|compute_shap> [options]")
        return 2
    cmd, subargv = argv[0], argv[1:]
    if cmd == "train":
        return cmd_train(subargv)
    if cmd == "predict":
        return cmd_predict(subargv)
    if cmd == "evaluate":
        return cmd_evaluate(subargv)
    if cmd == "prefit_kmm":
        return cmd_prefit_kmm(subargv)
    if cmd == "compute_shap":
        return cmd_compute_shap(subargv)
    print(f"Unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
