from __future__ import annotations
import numpy as np


# ---------- numerics ----------

_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _clip01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def _to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64, order="C")


# ---------- focal / CE ----------

def _ce_grad_hess(
    logits: np.ndarray,
    y: np.ndarray,
    w_pos: float = 1.0,
    gamma: float = 0.0,
    class1_boost: float = 1.0,
):
    """
    Binary CE (optionally focal) on margins.
    y in {0,1}. w_pos upweights positive class (AD). class1_boost multiplies y==1.
    """
    p = _clip01(_sigmoid(logits))
    w = np.ones_like(p)
    w[y == 1] *= (w_pos * class1_boost)

    if gamma and gamma > 0.0:
        pt = np.where(y == 1, p, 1.0 - p)         # prob of the true class
        mod = np.power(1.0 - pt, gamma)           # focal modulation
        grad = w * mod * (p - y)
        hess = w * mod * p * (1.0 - p)
    else:
        grad = w * (p - y)
        hess = w * p * (1.0 - p)

    return grad, np.maximum(hess, _EPS)


# ---------- entropy (SSDA) ----------

def _entropy_grad_hess(logits: np.ndarray, weight: float = 1.0):
    p = _clip01(_sigmoid(logits))
    # H(p) = -p log p - (1-p) log(1-p)
    # dH/dp = log((1-p)/p)
    dH_dp = np.log(_clip01(1.0 - p)) - np.log(_clip01(p))
    dp_dz = p * (1.0 - p)
    grad = weight * dH_dp * dp_dz
    # conservative, stabilized second derivative
    hess = weight * ((1.0 - 2.0 * p) * dH_dp - 1.0) * (dp_dz ** 2)
    return grad, np.maximum(np.abs(hess), _EPS)


# ---------- smooth BA gap ----------

def _ba_gap_surrogate_grad(
    logits: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tau_s: float = 2.0,
    lam: float = 1.0,
):
    """
    Smooth Balanced Accuracy per group:
      BA_g ~= 0.5 * ( E[sigmoid(z/tau) | y=1,g] + E[sigmoid(-z/tau) | y=0,g] )
    Penalize sum_{g<h} (BA_g - BA_h)^2. Returns per-sample grad/hess on z.
    """
    z = _to_numpy(logits)
    y = _to_numpy(y).astype(int)
    groups = _to_numpy(groups).astype(int)
    if groups is not None and len(groups) != len(y):
        raise ValueError(f"[CustomObjective] groups(len={len(groups)}) != y(len={len(y)}). Check source/target alignment.")

    s_pos = _sigmoid(z / max(tau_s, 1e-6))
    s_neg = _sigmoid(-z / max(tau_s, 1e-6))  # = 1 - s_pos

    uniq = np.unique(groups)
    G = len(uniq)
    if G <= 1:
        return np.zeros_like(z), np.zeros_like(z) + _EPS

    BA = {}
    dBA_dz = {}
    for g in uniq:
        mask_g = (groups == g)
        m_pos = mask_g & (y == 1)
        m_neg = mask_g & (y == 0)
        n_pos = max(int(m_pos.sum()), 1)
        n_neg = max(int(m_neg.sum()), 1)

        tpr_g = s_pos[m_pos].mean() if m_pos.any() else 0.0
        tnr_g = s_neg[m_neg].mean() if m_neg.any() else 0.0
        BA[g] = 0.5 * (tpr_g + tnr_g)

        d_s_pos = (s_pos * (1.0 - s_pos)) / max(tau_s, 1e-6)
        d_s_neg = -(s_neg * (1.0 - s_neg)) / max(tau_s, 1e-6)  # d/dz sigmoid(-z/tau)
        dBA = np.zeros_like(z)
        if m_pos.any():
            dBA[m_pos] += 0.5 * d_s_pos[m_pos] / n_pos
        if m_neg.any():
            dBA[m_neg] += 0.5 * d_s_neg[m_neg] / n_neg
        dBA_dz[g] = dBA

    grad = np.zeros_like(z)
    for i, gi in enumerate(uniq):
        for gj in uniq[i + 1:]:
            diff = BA[gi] - BA[gj]
            grad += 2.0 * diff * (dBA_dz[gi] - dBA_dz[gj])

    return lam * grad, np.zeros_like(z) + _EPS

def _eqodds_surrogate_grad(
    logits: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tau: float = 2.0,
    lam: float = 0.25,
    w_tpr: float = 0.5,
    w_fpr: float = 0.5
):
    """Smooth, pairwise equalized-odds penalty.
    Penalizes squared gaps in TPR and FPR between all group pairs, with
    sigmoid-smoothed confusion components to enable gradients.
    """
    z = _to_numpy(logits)
    y = _to_numpy(y).astype(int)
    g = _to_numpy(groups).astype(int)

    if z.size == 0:
        return np.zeros_like(z), np.zeros_like(z) + _EPS

    s_pos = _sigmoid(z / max(tau, 1e-6))       # soft TPR proxy when y=1
    s_neg = _sigmoid(-z / max(tau, 1e-6))      # soft TNR proxy when y=0
    d_s_pos = (s_pos * (1.0 - s_pos)) / max(tau, 1e-6)
    d_s_fpr = (s_neg * (1.0 - s_neg)) / max(tau, 1e-6)  # for d/dz of (1 - s_neg)

    uniq = np.unique(g)
    grad = np.zeros_like(z)
    for i, gi in enumerate(uniq):
        for gj in uniq[i+1:]:
            m_pos_i = (g == gi) & (y == 1); npi = max(int(m_pos_i.sum()), 1)
            m_pos_j = (g == gj) & (y == 1); npj = max(int(m_pos_j.sum()), 1)
            m_neg_i = (g == gi) & (y == 0); nni = max(int(m_neg_i.sum()), 1)
            m_neg_j = (g == gj) & (y == 0); nnj = max(int(m_neg_j.sum()), 1)

            # skip gracefully if any class subset missing
            tpr_i = float(s_pos[m_pos_i].mean()) if m_pos_i.any() else 0.0
            tpr_j = float(s_pos[m_pos_j].mean()) if m_pos_j.any() else 0.0
            fpr_i = float((1.0 - s_neg[m_neg_i]).mean()) if m_neg_i.any() else 0.0
            fpr_j = float((1.0 - s_neg[m_neg_j]).mean()) if m_neg_j.any() else 0.0

            # Gradients: squared gaps; split equally across the two groups
            # TPR part
            dgap_tpr = 2.0 * (tpr_i - tpr_j)
            grad[m_pos_i] += lam * w_tpr * dgap_tpr * (0.5 * d_s_pos[m_pos_i] / npi)
            grad[m_pos_j] -= lam * w_tpr * dgap_tpr * (0.5 * d_s_pos[m_pos_j] / npj)
            # FPR part
            dgap_fpr = 2.0 * (fpr_i - fpr_j)
            grad[m_neg_i] += lam * w_fpr * dgap_fpr * (0.5 * d_s_fpr[m_neg_i] / nni)
            grad[m_neg_j] -= lam * w_fpr * dgap_fpr * (0.5 * d_s_fpr[m_neg_j] / nnj)

    return grad, np.zeros_like(z) + _EPS

def _calibration_align_grad(
    logits: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tau: float = 2.0,
    lam: float = 0.2,
    use_logits: bool = False
):
    """Align per-group, per-class means toward a global barycenter.
    If use_logits=False, aligns probabilities σ(z); else aligns logits z.
    """
    z = _to_numpy(logits)
    y = _to_numpy(y).astype(int)
    g = _to_numpy(groups).astype(int)
    if z.size == 0:
        return np.zeros_like(z), np.zeros_like(z) + _EPS

    if use_logits:
        v = z.copy()
        dv_dz = np.ones_like(z)
    else:
        v = _sigmoid(z)
        dv_dz = v * (1.0 - v)

    # Per-class, per-group means
    grad = np.zeros_like(z)
    for c in (0, 1):
        mask_c = (y == c)
        if not mask_c.any():
            continue
        v_c = v[mask_c]
        g_c = g[mask_c]
        uniq = np.unique(g_c)
        if len(uniq) <= 1:
            continue
        mu_c = float(v_c.mean())
        for gi in uniq:
            m = (g_c == gi)
            if not m.any():
                continue
            mu_g = float(v_c[m].mean())
            # gradient contribution for samples in (g=c=gi)
            # d/dz L = 2*(mu_g - mu_c) * d(mu_g)/dz ;  d(mu_g)/dz = dv/dz / n_g
            idx = np.where(mask_c)[0][m]
            grad[idx] += lam * 2.0 * (mu_g - mu_c) * (dv_dz[idx] / max(int(m.sum()), 1))

    return grad, np.zeros_like(z) + _EPS


# ---------- class-conditional alignment on margins ----------

def _class_cond_bary_align_grad(
    logits: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes=(0, 1),
    lam: float = 1.0,
    barycentric: bool = True,
):
    """
    Align per-group class-conditional mean margin to the barycenter (or to a ref group).
    L_bary = sum_c sum_g || mu_{g,c} - mu_c ||^2
    L_pair = sum_c sum_{g>0} || mu_{g,c} - mu_{0,c} ||^2

    Gradient on each sample z_i in (g,c):
      bary: 2*(mu_{g,c} - mu_c) * ((G-1)/(G * n_{g,c}))
      pair: 2*(mu_{g,c} - mu_{0,c}) * (1/n_{g,c})
    """
    z = _to_numpy(logits)
    y = _to_numpy(y).astype(int)
    groups = _to_numpy(groups).astype(int)

    uniq_g = np.unique(groups)
    G = len(uniq_g)
    if G <= 1:
        return np.zeros_like(z), np.zeros_like(z) + _EPS

    valid = (y >= 0)
    grad = np.zeros_like(z, dtype=np.float64)

    for c in classes:
        idx_g = {g: np.where((groups == g) & valid & (y == c))[0] for g in uniq_g}
        mu_g = {g: (z[idx].mean() if idx.size else 0.0) for g, idx in idx_g.items()}
        if barycentric:
            mu_bar = np.mean([mu_g[g] for g in uniq_g])
            for g in uniq_g:
                idx = idx_g[g]
                n = max(idx.size, 1)
                coef = 2.0 * (mu_g[g] - mu_bar) * ((G - 1.0) / (G * n))
                if idx.size:
                    grad[idx] += coef
        else:
            g0 = uniq_g[0]
            for g in uniq_g[1:]:
                idx = idx_g[g]
                n = max(idx.size, 1)
                coef = 2.0 * (mu_g[g] - mu_g[g0]) * (1.0 / n)
                if idx.size:
                    grad[idx] += coef

    return lam * grad, np.zeros_like(z) + _EPS


# ---------- public API ----------

class CustomObjectiveV3:
    """
    loss_mode options (case-insensitive, '+' separated; optional weights in parentheses):
      ce | target | align | gap | any '+' combination, e.g. 'ce+target+align+gap'
      'target' means:
        - if approach == '1O' and target labels present -> target CE
        - else -> target entropy minimization (SSDA)
      Weighted terms: 'align(0.3)+gap(0.5)+target(0.8)'.
      Defaults when weight omitted:
        align -> self.eps ; gap -> self.gap_lambda ; target -> self.target_weight ; ce -> 1.0
    """
    def __init__(
        self,
        total_steps: int,
        n_source: int,
        m_target: int,
        tau: float = 1.0,
        eps: float = 1.0,
        approach: str = "1C",            # '1C' few-shot RegAlign; '2B' SSDA (unlabeled target)
        loss_mode: str | None = None,
        source_focal_gamma: float = 0.0,
        target_pos_weight: float = 1.0,
        update_every: int = 25,
        class1_boost: float = 1.0,
        # v1-style options (restored)
        target_focal_gamma: float = 0.0,
        use_target_reg: bool = False,
        target_weight: float = 1.0,
        lambda_align: float = 0.1,
        # ALL-groups homogenization
        bary_align: bool = False,
        var_align: bool = False,         # placeholder (kept for API compat)
        center_on: str = "barycenter",
        group_ids_source: np.ndarray | None = None,
        group_ids_target: np.ndarray | None = None,
        # extra knobs
        gap_kind: str = "ba",
        gap_lambda: float = 1.0,
        ba_tau: float = 2.0,
        fair_grad_alpha: float = 0.9,   # EMA smoothing for fairness grads
        fair_grad_ema: float = 0.9,     # use EMA on fairness grads
        eq_lambda: float = 0.0,         # equalized-odds penalty strength
        cal_lambda: float = 0.0, 
        # placeholders carried by wrapper
        X_combined=None,
        y_source=None,
    ):
        self.total_steps = int(total_steps)
        self.n_source = int(n_source)
        self.m_target = int(m_target)
        self.tau = float(tau)
        self.eps = float(eps)
        self.approach = str(approach or "1C")
        self.loss_mode = (loss_mode or "ce+align").lower()
        self.source_focal_gamma = float(source_focal_gamma)
        self.target_pos_weight = float(target_pos_weight)
        self.update_every = int(update_every)
        self.class1_boost = float(class1_boost)

        self.target_focal_gamma = float(target_focal_gamma)
        self.use_target_reg = bool(use_target_reg)
        self.target_weight = float(target_weight)
        self.lambda_align = float(lambda_align)

        self.bary_align = bool(bary_align)
        self.var_align = bool(var_align)
        self.center_on = center_on
        self.group_ids_source = None if group_ids_source is None else _to_numpy(group_ids_source).astype(int)
        self.group_ids_target = None if group_ids_target is None else _to_numpy(group_ids_target).astype(int)

        self.ba_tau = float(ba_tau)           # smooth BA slope
        self.gap_lambda = float(gap_lambda)   # BA-gap weight
        
        self.gap_kind = str(gap_kind or "eqodds").lower().replace("/", "+")
        self.fair_grad_alpha = float(fair_grad_alpha)
        self.fair_grad_ema = float(fair_grad_ema)
        self.eq_lambda = float(eq_lambda)
        self.cal_lambda = float(cal_lambda)
        self._ema_ba = {}

    # ---------- parsing helpers ----------
    @staticmethod
    def _parse_loss_mode(loss_mode: str):
        """
        Returns flags and weights dict like:
          flags = {'ce':True,'target':True,'align':True,'gap':True}
          weights = {'ce':1.0,'target':None,'align':None,'gap':None}
        None weight => use default from self (align->eps, gap->gap_lambda, target->target_weight)
        """
        tokens = [t.strip() for t in (loss_mode or "").lower().split("+") if t.strip()]
        flags = {'ce': False, 'target': False, 'align': False, 'gap': False}
        weights = {'ce': None, 'target': None, 'align': None, 'gap': None}
        for tk in tokens:
            if "(" in tk and tk.endswith(")"):
                name, w = tk.split("(", 1)
                name = name.strip()
                try:
                    val = float(w[:-1])
                except Exception:
                    val = None
            else:
                name, val = tk, None
            if name in flags:
                flags[name] = True
                weights[name] = val
        # backward-compat aliases
        if not any(flags.values()):  # empty => default
            flags['ce'] = True; flags['align'] = True
        return flags, weights

    def _grad_hess_approach_1c(self, logits: np.ndarray, labels_all: np.ndarray, flags, weights):
        ns = self.n_source
        nt = self.m_target
        grad = np.zeros_like(logits)
        hess = np.zeros_like(logits)

        labels_src = labels_all[:ns]
        preds_src = logits[:ns]
        probs_src = _clip01(_sigmoid(preds_src), self.eps)

        class_weight_src = np.where(labels_src == 1, self.class1_boost, 1.0)
        if self.source_focal_gamma > 0:
            p_t = np.where(labels_src == 1, probs_src, 1.0 - probs_src)
            mod = np.power(1.0 - p_t, self.source_focal_gamma)
            grad[:ns] = class_weight_src * mod * (probs_src - labels_src)
            hess[:ns] = class_weight_src * mod * (probs_src * (1.0 - probs_src))
        else:
            grad[:ns] = class_weight_src * (probs_src - labels_src)
            hess[:ns] = class_weight_src * probs_src * (1.0 - probs_src)

        if nt > 0 and (flags.get("target") or flags.get("align")):
            preds_tgt = logits[ns:ns + nt]
            probs_tgt = _clip01(_sigmoid(preds_tgt), self.eps)
            grad_target = np.zeros(nt)
            hess_target = np.zeros(nt)

            labels_tgt = labels_all[ns:]
            if flags.get("target"):
                class_weight_tgt = np.where(labels_tgt == 1, self.class1_boost, 1.0)
                if self.target_focal_gamma > 0:
                    p_t_tgt = np.where(labels_tgt == 1, probs_tgt, 1.0 - probs_tgt)
                    mod_tgt = np.power(1.0 - p_t_tgt, self.target_focal_gamma)
                    grad_tgt_ce = class_weight_tgt * mod_tgt * (probs_tgt - labels_tgt)
                    hess_tgt_ce = class_weight_tgt * mod_tgt * (probs_tgt * (1.0 - probs_tgt))
                else:
                    grad_tgt_ce = class_weight_tgt * (probs_tgt - labels_tgt)
                    hess_tgt_ce = class_weight_tgt * probs_tgt * (1.0 - probs_tgt)
                scale_tgt = weights.get("target") if weights.get("target") is not None else self.tau
                grad[ns:] += scale_tgt * grad_tgt_ce
                hess[ns:] += np.maximum(scale_tgt * hess_tgt_ce, 1e-6)

            if flags.get("align"):
                lam = weights.get("align") if weights.get("align") is not None else float(getattr(self, "lambda_align", self.eps))
                for cls in (0, 1):
                    src_mask = (labels_src == cls)
                    tgt_mask = (labels_tgt == cls)
                    if np.any(src_mask) and np.any(tgt_mask):
                        mu_s = float(np.mean(probs_src[src_mask]))
                        mu_t = float(np.mean(probs_tgt[tgt_mask]))
                        diff = mu_t - mu_s
                        denom = np.sum(tgt_mask)
                        if denom > 0:
                            pt = probs_tgt[tgt_mask]
                            grad_target[tgt_mask] += lam * (2.0 * diff / denom) * pt * (1.0 - pt)
                            hess_target[tgt_mask] += lam * (2.0 / denom) * pt * (1.0 - pt)

                grad[ns:] += grad_target
                hess[ns:] += np.maximum(hess_target, 1e-6)

        return grad, np.maximum(hess, 1e-6)

    def __call__(self, preds: np.ndarray, dtrain):
        """
        preds: raw margins (scores) from current boosting iteration
        dtrain: xgb.DMatrix with .get_label()
        """
        z = _to_numpy(preds)
        y_all = _to_numpy(dtrain.get_label()).astype(int)

        n = len(z)
        ns = self.n_source
        nt = self.m_target
        assert ns + nt == n, f"Expected ns+nt == n ({ns}+{nt}!={n}). Did you pass target_X to the wrapper?"

        # parse modes/weights
        flags, weights = self._parse_loss_mode(self.loss_mode)
        # CE must always be on
        flags["ce"] = True

        y_src = y_all[:ns]
        # few-shot labels available only for '1O' path
        y_tgt = None if nt == 0 else (y_all[ns:] if (self.approach.upper() == "1O" and np.unique(y_all[ns:]).size > 1) else None)

        g = np.zeros_like(z)
        h = np.zeros_like(z) + _EPS

        if self.approach.upper() == "1C":
            return self._grad_hess_approach_1c(z, y_all, flags, weights)

        # ---------- source CE ----------
        if flags['ce']:
            w_ce = weights['ce'] if weights['ce'] is not None else 1.0
            g_src, h_src = _ce_grad_hess(
                z[:ns], y_src,
                w_pos=self.target_pos_weight,
                gamma=self.source_focal_gamma,
                class1_boost=self.class1_boost,
            )
            
            if self.fair_grad_alpha > 0.0 and (self.group_ids_source is not None):
                # compute/update EMA of soft BA per group (source only)
                if ns > 0:
                    ba_now = _soft_ba_by_group(z[:ns], y_src, self.group_ids_source, tau=self.tau)
                    # update EMA
                    for gi, val in ba_now.items():
                        prev = self._ema_ba.get(int(gi), val)
                        self._ema_ba[int(gi)] = self.fair_grad_ema * prev + (1.0 - self.fair_grad_ema) * val
                    # build per-sample multipliers
                    mu = np.mean(list(self._ema_ba.values())) if self._ema_ba else 1.0
                    r_map = {gi: (mu / max(v, 1e-6)) ** self.fair_grad_alpha for gi, v in self._ema_ba.items()}
                    r_per_sample = np.ones(ns, dtype=np.float64)
                    for idx in range(ns):
                        gi = int(self.group_ids_source[idx])
                        r_per_sample[idx] = r_map.get(gi, 1.0)
                    g_src[:ns] *= r_per_sample
                    h_src[:ns] *= r_per_sample  # conservative scaling
            g[:ns] += w_ce * g_src
            h[:ns] += w_ce * h_src

        # ---------- target CE / entropy ----------
        if nt > 0 and flags['target']:
            w_tgt = weights['target'] if weights['target'] is not None else self.target_weight
            if self.approach.upper() == "1O" and y_tgt is not None:
                # few-shot labeled target
                g_t, h_t = _ce_grad_hess(
                    z[ns:], y_tgt,
                    w_pos=self.target_pos_weight,
                    gamma=self.target_focal_gamma,
                    class1_boost=1.0,
                )
                g[ns:] += self.tau * w_tgt * g_t
                h[ns:] += self.tau * w_tgt * h_t
            else:
                # SSDA unlabeled: entropy minimization
                g_e, h_e = _entropy_grad_hess(z[ns:], weight=self.tau * w_tgt)
                g[ns:] += g_e
                h[ns:] += h_e

        # build group ids for align/gap if any is on
        need_gid = (flags['align'] or flags['gap']) and (self.group_ids_source is not None)
        if need_gid:
            gid = self.group_ids_source if nt == 0 or self.group_ids_target is None \
                else np.concatenate([self.group_ids_source, self.group_ids_target])
        
        if nt > 0:
            # pad target portion with -1 so class masks skip target rows
            y_for_fair = np.concatenate([y_src, -np.ones(nt, dtype=int)])
        else:
            y_for_fair = y_src
        
        if need_gid and len(gid) != len(z):
            raise ValueError(f"[CustomObjective] group_ids length {len(gid)} != logits length {len(z)}")
        if (flags['align'] or flags['gap']) and len(y_for_fair) != len(z):
            raise ValueError(f"[CustomObjective] labels_for_fair length {len(y_for_fair)} != logits length {len(z)}")

        # ---------- alignment on margins ----------
        if flags['align'] and (self.eps and self.eps > 0.0) and (self.group_ids_source is not None):
            w_align = weights['align'] if weights['align'] is not None else self.eps
            ga, ha = _class_cond_bary_align_grad(
                z,
                y_for_fair,
                gid,
                lam=w_align,
                barycentric=bool(self.bary_align),
            )
            g += ga
            h += ha  # tiny eps inside

        # ---------- fairness gap penalties (BA / EqOdds / Calibration) ----------
        if flags['gap'] and (self.group_ids_source is not None):
            kinds = [k.strip() for k in self.gap_kind.split('+')]
            # Default weights if not explicitly passed in loss_mode (weights['gap'])
            w_gap = weights['gap'] if weights['gap'] is not None else self.gap_lambda

            # (i) BA GAP (uses soft BA per group; supervised on source only)
            if ('ba' in kinds) and (w_gap > 0):
                gb, hb = _ba_gap_surrogate_grad(z, y_for_fair, gid,
                                                tau_s=self.ba_tau, lam=w_gap)
                g += gb; h += hb

            # Threshold for EqOdds/Calibration (cost-aware if available; else 0.5)
            thr = 0.5  # simple default; objective is threshold-free, so keep it stable

            # (ii) Equalized Odds (match TPR/FPR across groups on source)
            if ('eqodds' in kinds) and (self.eq_lambda > 0):
                ge, he = _eqodds_surrogate_grad(z, y_for_fair, gid,
                                                lam=self.eq_lambda, thr=thr)
                g += ge; h += he

            # (iii) Calibration alignment (match calibrated probabilities across groups)
            if ('cal' in kinds) and (self.cal_lambda > 0):
                gc, hc = _calibration_align_grad(z, y_for_fair, gid,
                                                lam=self.cal_lambda, thr=thr)
                g += gc; h += hc

        return g.astype(np.float64), np.maximum(h, _EPS).astype(np.float64)

def _soft_ba_by_group(logits: np.ndarray, y: np.ndarray, groups: np.ndarray, tau: float = 2.0):
    """Return dict {group_id: soft_BA} using sigmoid-smoothed confusion elements."""
    z = _to_numpy(logits); y = _to_numpy(y).astype(int); g = _to_numpy(groups).astype(int)
    s_pos = _sigmoid(z / max(tau, 1e-6))
    s_neg = _sigmoid(-z / max(tau, 1e-6))
    out = {}
    for gi in np.unique(g):
        m_pos = (g == gi) & (y == 1)
        m_neg = (g == gi) & (y == 0)
        tpr = float(s_pos[m_pos].mean()) if m_pos.any() else 0.0
        tnr = float(s_neg[m_neg].mean()) if m_neg.any() else 0.0
        out[int(gi)] = 0.5 * (tpr + tnr)
    return out

# ---------- optional: simpler custom objectives ----------

def focal_binary_object(alpha: float = 0.9, gamma: float = 2.0):
    def _obj(preds, dtrain):
        y = _to_numpy(dtrain.get_label()).astype(int)
        z = _to_numpy(preds)
        # alpha is prior prob of positive class; convert to cost ratio
        w_pos = alpha / max(1.0 - alpha, _EPS)
        g, h = _ce_grad_hess(z, y, w_pos=w_pos, gamma=gamma, class1_boost=1.0)
        return g, h
    return _obj


def cost_sensitive_binary_object(cost_fn: float = 5.0, cost_fp: float = 1.0):
    def _obj(preds, dtrain):
        y = _to_numpy(dtrain.get_label()).astype(int)
        z = _to_numpy(preds)
        w_pos = float(cost_fn) / max(float(cost_fp), _EPS)
        g, h = _ce_grad_hess(z, y, w_pos=w_pos, gamma=0.0, class1_boost=1.0)
        return g, h
    return _obj
