"""
Estimators for parent vectors and child deltas using LDA in the causal space.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class LDAEstimator:
    def __init__(self, shrinkage: float = 0.1, class_balance: bool = True):
        self.shrinkage = float(shrinkage)
        self.class_balance = class_balance

    def estimate_binary_direction(
        self,
        X_pos: torch.Tensor,
        X_neg: torch.Tensor,
        geometry,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Estimate LDA direction (binary), returning a vector in original space.

        Steps:
          - Whiten pos/neg with provided geometry
          - Compute class means and within-class covariance (with shrinkage)
          - Solve S_w^{-1} (mu_pos - mu_neg) in whitened space
          - Map back to original space and (optionally) normalize under causal norm
        """
        Xp = geometry.whiten(X_pos.to(torch.float32))
        Xn = geometry.whiten(X_neg.to(torch.float32))
        if self.class_balance:
            n = min(len(Xp), len(Xn))
            Xp, Xn = Xp[:n], Xn[:n]
        mu_p = Xp.mean(dim=0)
        mu_n = Xn.mean(dim=0)
        Xpc = Xp - mu_p
        Xnc = Xn - mu_n
        Xc = torch.cat([Xpc, Xnc], dim=0)
        d = Xc.shape[1]
        Sw = (Xc.t() @ Xc) / max(1, len(Xc) - 1) + self.shrinkage * torch.eye(d, dtype=torch.float32)
        try:
            w_w = torch.linalg.solve(Sw, (mu_p - mu_n))
        except torch.linalg.LinAlgError:
            w_w = torch.linalg.pinv(Sw) @ (mu_p - mu_n)
        # Map back to original space using W^{-T}
        # whiten(x) = x W^T ⇒ direction_original = w_w @ (W^T)^{-1}
        # Solve W x = w_w  (since whiten(x) = x W^T ⇒ w_w is in whitened coords)
        direction_orig = torch.linalg.solve(geometry.W.to(torch.float32), w_w)
        if normalize:
            direction_orig = geometry.normalize_causal(direction_orig)
        return direction_orig


class ConceptVectorEstimator:
    def __init__(self, lda: LDAEstimator, geometry):
        self.lda = lda
        self.geometry = geometry

    def estimate_parent_vectors(
        self, parent_acts: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for pid, d in parent_acts.items():
            if "pos" in d and "neg" in d:
                vec = self.lda.estimate_binary_direction(d["pos"], d["neg"], self.geometry, normalize=True)
                out[pid] = vec
        return out

    def estimate_child_deltas(
        self,
        parent_vectors: Dict[str, torch.Tensor],
        child_acts: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        child_vecs: Dict[str, Dict[str, torch.Tensor]] = {}
        child_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        for pid, d in child_acts.items():
            if pid not in parent_vectors:
                continue
            pvec = parent_vectors[pid]
            child_vecs[pid] = {}
            child_deltas[pid] = {}
            # d is mapping child_id -> {pos, neg}
            for cid, splits in d.items():
                if "pos" in splits and "neg" in splits:
                    cvec = self.lda.estimate_binary_direction(splits["pos"], splits["neg"], self.geometry, normalize=True)
                    delta = cvec - pvec
                    delta = self.geometry.normalize_causal(delta)
                    child_vecs[pid][cid] = cvec
                    child_deltas[pid][cid] = delta
        return child_vecs, child_deltas

    def estimate_child_subspace_projectors(
        self,
        child_deltas: Dict[str, Dict[str, torch.Tensor]],
        subspace_dim: int = 32,
        energy_threshold: float = 0.85,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return per-parent (down_proj, up_proj) via SVD of stacked deltas.

        Shapes: down_proj [k, d], up_proj [d, k], where k = min(subspace_dim, k_energy).
        """
        projectors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for pid, deltas in child_deltas.items():
            if not deltas:
                continue
            M = torch.stack(list(deltas.values()))  # [n_children, d]
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            V = Vh.t()  # [d, r]
            cum = torch.cumsum(S, dim=0) / (S.sum() + 1e-12)
            k_energy = int((cum <= energy_threshold).sum().item()) + 1
            k = min(subspace_dim, V.shape[1], k_energy)
            Vk = V[:, :k]
            down = Vk.t()   # [k, d]
            up = Vk         # [d, k]
            projectors[pid] = (down, up)
        return projectors


class MeanDiffEstimator:
    def __init__(self, class_balance: bool = True):
        self.class_balance = class_balance

    def estimate_binary_direction(
        self,
        X_pos: torch.Tensor,
        X_neg: torch.Tensor,
        geometry,
        normalize: bool = True,
    ) -> torch.Tensor:
        Xp = geometry.whiten(X_pos.to(torch.float32))
        Xn = geometry.whiten(X_neg.to(torch.float32))
        if self.class_balance:
            n = min(len(Xp), len(Xn))
            Xp, Xn = Xp[:n], Xn[:n]
        mu_p = Xp.mean(dim=0)
        mu_n = Xn.mean(dim=0)
        w_w = (mu_p - mu_n)
        # Map back to original space using W^{-T}
        direction_orig = torch.linalg.solve(geometry.W.to(torch.float32), w_w)
        if normalize:
            direction_orig = geometry.normalize_causal(direction_orig)
        return direction_orig


class L2ProbeEstimator:
    def __init__(self, C: float = 1.0, max_iter: int = 500):
        self.C = C
        self.max_iter = max_iter

    def estimate_binary_direction(
        self,
        X_pos: torch.Tensor,
        X_neg: torch.Tensor,
        geometry,
        normalize: bool = True,
    ) -> torch.Tensor:
        from sklearn.linear_model import LogisticRegression

        Xp = geometry.whiten(X_pos.to(torch.float32))
        Xn = geometry.whiten(X_neg.to(torch.float32))
        n = min(len(Xp), len(Xn))
        Xp, Xn = Xp[:n], Xn[:n]
        X = torch.cat([Xp, Xn], dim=0).numpy()
        y = torch.cat([torch.ones(n), torch.zeros(n)]).numpy()
        clf = LogisticRegression(max_iter=self.max_iter, C=self.C, solver="lbfgs")
        clf.fit(X, y)
        w_w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        direction_orig = torch.linalg.solve(geometry.W.to(torch.float32), w_w)
        if normalize:
            direction_orig = geometry.normalize_causal(direction_orig)
        return direction_orig
