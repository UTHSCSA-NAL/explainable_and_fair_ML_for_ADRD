import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from scipy.special import expit as sigmoid

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


class CustomObjective:
    def __init__(self, total_steps, n_source, m_target, tau, eps, approach='few_shot'):
        self.total_steps = total_steps
        self.n_source = n_source
        self.m_target = m_target
        self.tau = tau
        self.eps = eps
        self.approach = approach
        self.current_step = 0

    def __call__(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        self.current_step += 1
        labels = dtrain.get_label()
        p = sigmoid(predt)
        p = np.clip(p, self.eps, 1 - self.eps)
        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)
        mu_source = np.mean(p[:self.n_source])
        mu_target = np.mean(p[self.n_source:self.n_source + self.m_target])
        if self.approach == 'few_shot':
            for i in range(self.n_source):
                sup_grad = p[i] - labels[i]
                align_grad = self.tau * (2.0 / self.n_source) * (mu_source - mu_target)
                grad[i] = sup_grad + align_grad
                hess[i] = p[i] * (1 - p[i])
            for i in range(self.n_source, self.n_source + self.m_target):
                sup_grad = p[i] - labels[i]
                align_grad = self.tau * (2.0 / self.m_target) * (mu_target - mu_source)
                grad[i] = sup_grad + align_grad
                hess[i] = p[i] * (1 - p[i])
        elif self.approach == 'unsupervised':
            for i in range(self.n_source):
                sup_grad = p[i] - labels[i]
                align_grad = self.tau * (2.0 / self.n_source) * (mu_source - mu_target)
                grad[i] = sup_grad + align_grad
                hess[i] = p[i] * (1 - p[i])
            for i in range(self.n_source, self.n_source + self.m_target):
                ent_grad = - predt[i] * p[i] * (1 - p[i])
                align_grad = self.tau * (2.0 / self.m_target) * (mu_target - mu_source)
                grad[i] = ent_grad + align_grad
                hess[i] = p[i] * (1 - p[i])
        else:
            raise ValueError("Unsupported approach. Use 'few_shot' or 'unsupervised'.")
        return grad, hess


class CustomObjectiveV2:
    def __init__(self, total_steps, n_source, m_target, tau, eps, approach='1A',
                 source_focal_gamma=0., target_focal_gamma=0., use_target_reg=False, target_weight=1.0,
                 X_combined=None, y_source=None):
        self.total_steps = total_steps
        self.n_source    = n_source
        self.m_target    = m_target
        self.tau      = tau
        self.eps      = eps
        self.approach = approach
        self.class1_boost       = 1.5
        self.current_step       = 0
        self.source_focal_gamma = source_focal_gamma
        self.target_focal_gamma = target_focal_gamma
        self.use_target_reg     = use_target_reg
        self.target_weight      = target_weight
        self.X_combined    = X_combined
        self.y_source      = y_source
        self.pseudo_labels = None
        if approach in ['1A', '1B', '1C'] and m_target > 15:
            print(f"Warning: {m_target} target samples for {approach}, expected fewer than 15 for few-shot.")
        if approach in ['2A', '2B'] and (X_combined is None or y_source is None):
            raise ValueError("X_combined and y_source required for 2A/2B approaches.")
        if approach not in ['1A', '1B', '1C', '2A', '2B']:
            raise ValueError("Unsupported approach. Use '1A', '1B', '1C', '2A', or '2B'.")
    
    def update_pseudo_labels(self):
        if self.approach not in ['2A\'', '2B\'']:
            return
        if self.current_step % 50 == 0 or self.pseudo_labels is None:
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(self.X_combined)
            source_clusters = clusters[:self.n_source]
            cluster_scores = []
            for c in [0, 1]:
                mask = source_clusters == c
                if np.sum(mask) > 0:
                    score = np.mean(self.y_source[mask])
                else:
                    score = 0.5
                cluster_scores.append(score)
            class_1_cluster = 0 if cluster_scores[0] > cluster_scores[1] else 1
            self.pseudo_labels = (clusters[self.n_source:] == class_1_cluster).astype(int)

    def __call__(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        self.current_step += 1
        # Use source labels only for computing the source cross-entropy.
        labels = dtrain.get_label()[:self.n_source]
        predt_source = predt[:self.n_source]
        probs_source = sigmoid(predt_source)
        probs_source = np.clip(probs_source, self.eps, 1 - self.eps)
        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)
        
        # Apply focal weighting on the source if requested.
        if self.source_focal_gamma > 0:
            # For each source sample, define p_t as p if label==1, and (1-p) if label==0.
            p_t = np.where(labels == 1, probs_source, 1 - probs_source)
            modulating_factor = (1 - p_t) ** self.source_focal_gamma
            class_weight = np.where(labels == 1, self.class1_boost, 1.0)
            grad[:self.n_source] = class_weight * modulating_factor * (probs_source - labels)
            hess[:self.n_source] = class_weight * modulating_factor * (probs_source * (1 - probs_source))
        else:
            class_weight = np.where(labels == 1, self.class1_boost, 1.0)
            grad[:self.n_source] = class_weight * (probs_source - labels)
            hess[:self.n_source] = class_weight * probs_source * (1 - probs_source)
        
        if self.m_target > 0:
            predt_target = predt[self.n_source:self.n_source + self.m_target]
            probs_target = sigmoid(predt_target)
            probs_target = np.clip(probs_target, self.eps, 1 - self.eps)
            if self.approach == '1A':
                grad_target = np.zeros(self.m_target)
                hess_target = np.zeros(self.m_target)
                y_target = dtrain.get_label()[self.n_source:]
                grad_target_ce = probs_target - y_target
                hess_target_ce = probs_target * (1 - probs_target)
                for c in [0, 1]:
                    source_mask = labels == c
                    if np.sum(source_mask) > 0:
                        mu_s = np.mean(probs_source[source_mask])
                        target_mask = y_target == c
                        if np.sum(target_mask) > 0:
                            diff = probs_target[target_mask] - mu_s
                            grad_target[target_mask] += self.eps * 2 * diff * probs_target[target_mask] * (1 - probs_target[target_mask])
                            hess_target[target_mask] += self.eps * 2 * (probs_target[target_mask] * (1 - probs_target[target_mask]) + diff * (1 - 2 * probs_target[target_mask]))
                grad_target += self.tau * grad_target_ce
                hess_target += self.tau * hess_target_ce
                grad[self.n_source:] = grad_target
                hess[self.n_source:] = np.maximum(hess_target, 1e-6)
            elif self.approach == '1B':
                grad_target = np.zeros(self.m_target)
                hess_target = np.zeros(self.m_target)
                y_target = dtrain.get_label()[self.n_source:]
                grad_target_ce = probs_target - y_target
                hess_target_ce = probs_target * (1 - probs_target)
                for c in [0, 1]:
                    source_mask = labels == c
                    target_mask = y_target == c
                    if np.sum(source_mask) > 0 and np.sum(target_mask) > 0:
                        mu_s = np.mean(probs_source[source_mask])
                        mu_t = np.mean(probs_target[target_mask])
                        diff = mu_t - mu_s
                        grad_target[target_mask] += self.eps * 2 * diff * probs_target[target_mask] * (1 - probs_target[target_mask]) / np.sum(target_mask)
                        hess_target[target_mask] += self.eps * 2 * (probs_target[target_mask] * (1 - probs_target[target_mask])) / np.sum(target_mask)
                grad_target += self.tau * grad_target_ce
                hess_target += self.tau * hess_target_ce
                grad[self.n_source:] = grad_target
                hess[self.n_source:] = np.maximum(hess_target, 1e-6)
            elif self.approach == '1C':
                # New approach: explicit domain adaptation (like old 1B) but with an extra regularization flag
                # that scales down the contribution of the adaptation term.
                grad_target = np.zeros(self.m_target)
                hess_target = np.zeros(self.m_target)
                y_target = dtrain.get_label()[self.n_source:]
                if self.target_focal_gamma > 0:
                    p_t_target = np.where(y_target == 1, probs_target, 1 - probs_target)
                    modulating_factor_target = (1 - p_t_target) ** self.target_focal_gamma
                    
                    class_target_weight = np.where(y_target == 1, self.class1_boost, 1.0)

                    grad_target_ce = class_target_weight * modulating_factor_target * (probs_target - y_target)
                    hess_target_ce = class_target_weight * modulating_factor_target * (probs_target * (1 - probs_target))
                else:
                    class_target_weight = np.where(y_target == 1, self.class1_boost, 1.0)
                    grad_target_ce = class_target_weight * (probs_target - y_target)
                    hess_target_ce = class_target_weight * probs_target * (1 - probs_target)
                for c in [0, 1]:
                    source_mask = labels == c
                    target_mask = y_target == c
                    if np.sum(source_mask) > 0 and np.sum(target_mask) > 0:
                        mu_s = np.mean(probs_source[source_mask])
                        mu_t = np.mean(probs_target[target_mask])
                        diff = mu_t - mu_s
                        grad_target[target_mask] += self.eps * 2 * diff * probs_target[target_mask] * (1 - probs_target[target_mask]) / np.sum(target_mask)
                        hess_target[target_mask] += self.eps * 2 * (probs_target[target_mask]*(1-probs_target[target_mask])) / np.sum(target_mask)
                # If use_target_reg is True, scale down the adaptation term by target_weight (<1).
                if self.use_target_reg:
                    grad_target = self.target_weight * grad_target + self.tau * grad_target_ce
                    hess_target = self.target_weight * hess_target + self.tau * hess_target_ce
                else:
                    grad_target = grad_target + self.tau * grad_target_ce
                    hess_target = hess_target + self.tau * hess_target_ce
                grad[self.n_source:] = grad_target
                hess[self.n_source:] = np.maximum(hess_target, 1e-6)
            elif self.approach == '2A':
                # Pseudo-labeling: assign pseudo-labels based on threshold 0.5.
                self.update_pseudo_labels()
                grad_target = probs_target - self.pseudo_labels
                hess_target = probs_target * (1 - probs_target)
                grad[self.n_source:] = self.tau * grad_target
                hess[self.n_source:] = self.tau * np.maximum(hess_target, 1e-6)
            elif self.approach == '2B':
                # Entropy minimization.
                self.update_pseudo_labels()
                grad_target = np.zeros(self.m_target)
                hess_target = np.zeros(self.m_target)
                for c in [0, 1]:
                    source_mask = labels == c
                    target_mask = self.pseudo_labels == c
                    if np.sum(source_mask) > 0 and np.sum(target_mask) > 0:
                        mu_s = np.mean(probs_source[source_mask])
                        mu_t = np.mean(probs_target[target_mask])
                        diff = mu_t - mu_s
                        grad_target[target_mask] += self.eps * 2 * diff * probs_target[target_mask] * (1 - probs_target[target_mask]) / np.sum(target_mask)
                        hess_target[target_mask] += self.eps * 2 * (probs_target[target_mask] * (1 - probs_target[target_mask])) / np.sum(target_mask)
                grad[self.n_source:] = grad_target
                hess[self.n_source:] = np.maximum(hess_target, 1e-6)
            else:
                raise ValueError("Unsupported approach. Use '1A', '1B', '2A', or '2B'.")
        return grad, hess
