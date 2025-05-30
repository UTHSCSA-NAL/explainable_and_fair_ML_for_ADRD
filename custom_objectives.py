import numpy as np
import xgboost as xgb
from scipy.special import expit as sigmoid

class CustomObjective:
    """
    Single merged objective. Supports only:
      - '1C': few‐shot domain alignment
      - '2B': entropy‐min pseudo-label alignment
    """
    def __init__(self, total_steps, n_source, m_target,
                 tau=1.0, eps=1e-7, approach='1C'):
        if approach not in ('1C','2B'):
            raise ValueError("Only approach '1C' or '2B' supported.")
        self.total_steps   = total_steps
        self.n_source      = n_source
        self.m_target      = m_target
        self.tau           = tau
        self.eps           = eps
        self.approach      = approach
        self.current_step  = 0
        self.pseudo_labels = None

    def update_pseudo_labels(self, p_t):
        # Hard threshold at 0.5
        self.pseudo_labels = (p_t >= 0.5).astype(int)

    def __call__(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        self.current_step += 1
        labels = dtrain.get_label()
        p_all  = np.clip(sigmoid(predt), self.eps, 1-self.eps)

        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)

        # --- source cross-entropy ---
        p_s = p_all[:self.n_source]
        y_s = labels[:self.n_source]
        grad[:self.n_source] = p_s - y_s
        hess[:self.n_source] = p_s * (1 - p_s)

        # --- target few-shot part ---
        if self.m_target > 0:
            p_t = p_all[self.n_source:self.n_source+self.m_target]
            y_t = labels[self.n_source:self.n_source+self.m_target]

            if self.approach == '1C':
                # CE + class‐wise alignment
                grad_t = p_t - y_t
                hess_t = p_t * (1 - p_t)
                for c in (0,1):
                    mask_s = (y_s == c)
                    mask_t = (y_t == c)
                    if mask_s.any() and mask_t.any():
                        mu_s = p_s[mask_s].mean()
                        mu_t = p_t[mask_t].mean()
                        diff = mu_t - mu_s
                        grad_t[mask_t] += self.eps * 2 * diff * p_t[mask_t]*(1-p_t[mask_t]) / mask_t.sum()
                        hess_t[mask_t] += self.eps * 2*(p_t[mask_t]*(1-p_t[mask_t]))/mask_t.sum()
                grad[self.n_source:] = self.tau * grad_t
                hess[self.n_source:] = np.maximum(self.tau*hess_t, 1e-6)

            else:  # '2B'
                # entropy‐min + pseudo‐label alignment
                if self.pseudo_labels is None or self.current_step == 1:
                    self.update_pseudo_labels(p_t)
                grad_t = np.zeros_like(p_t)
                hess_t = np.zeros_like(p_t)
                for c in (0,1):
                    mask_s = (y_s == c)
                    mask_t = (self.pseudo_labels == c)
                    if mask_s.any() and mask_t.any():
                        mu_s = p_s[mask_s].mean()
                        mu_t = p_t[mask_t].mean()
                        diff = mu_t - mu_s
                        grad_t[mask_t] += self.eps * 2 * diff * p_t[mask_t]*(1-p_t[mask_t]) / mask_t.sum()
                        hess_t[mask_t] += self.eps * 2*(p_t[mask_t]*(1-p_t[mask_t]))/mask_t.sum()
                grad[self.n_source:] = grad_t
                hess[self.n_source:] = np.maximum(hess_t, 1e-6)

        return grad, hess
