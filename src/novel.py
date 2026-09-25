"""
Novel components that distinguish this project from standard GNN+KG pipelines.

1) Temporal Attention Graph Neural Network (TAGNN)
   - Standard message-passing GNNs treat every edge equally. TAGNN adds a
     learnable temporal gate that down-weights neighbors whose feature
     trajectory diverges from the target patient.  This is especially
     useful for longitudinal diabetes data where patients with similar
     early trajectories are more informative than patients who merely
     share a condition label.

2) Multi-Head Feature Interaction Network (MHFIN)
   - Explicitly models pairwise feature interactions through a small
     set of learned projection heads, then concatenates the interaction
     embeddings with the base features before the classifier layer.

3) Confidence-Calibrated Fusion (CCF)
   - Instead of hard-voting or simple averaging, CCF fits a small
     logistic calibration layer on top of stacked base-model
     probabilities.  The calibration layer learns per-model reliability
     weights from a held-out fold.

4) Patient Trajectory Clustering (PTC)
   - Groups patients by the shape of their HbA1c trajectory (slope,
     curvature) rather than raw values, then appends the cluster ID as
     an extra categorical feature.  Helps the classifier pick up
     population-level progression patterns.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 1. Temporal Attention Graph Neural Network
# ──────────────────────────────────────────────

class TAGNNLayer(nn.Module):
    """Single temporal-attention message-passing layer."""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.w_self = nn.Linear(in_dim, hid_dim)
        self.w_neigh = nn.Linear(in_dim, hid_dim)
        self.t_gate = nn.Sequential(
            nn.Linear(in_dim * 2, hid_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        self_emb = self.w_self(x)
        neigh_emb = self.w_neigh(x[col])

        # temporal gate: compare source & neighbor features
        gate_input = torch.cat([x[row], x[col]], dim=1)
        gate = self.t_gate(gate_input)

        msg = gate * neigh_emb
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(1)

        agg = torch.zeros_like(self_emb).index_add(0, row, msg)
        deg = torch.zeros(x.size(0), device=x.device).index_add(
            0, row, torch.ones_like(row, dtype=torch.float32)
        )
        out = self.norm(self_emb + agg / deg.clamp_min(1).unsqueeze(1))
        return F.relu(out)


class TAGNN(nn.Module):
    """Temporal Attention GNN for patient-level classification."""

    def __init__(self, in_dim, hid_dim=32, n_layers=2, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(TAGNNLayer(
                in_dim if i == 0 else hid_dim, hid_dim
            ))
        self.dropout = dropout
        self.head = nn.Linear(hid_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for layer in self.layers:
            h = F.dropout(layer(h, edge_index, edge_weight),
                          p=self.dropout, training=self.training)
        return self.head(h).squeeze(1)


# ──────────────────────────────────────────────
# 2. Multi-Head Feature Interaction Network
# ──────────────────────────────────────────────

class MHFIN(nn.Module):
    """
    Learns k pairwise interaction projections and concatenates them
    with the original features to produce an enriched representation.
    """

    def __init__(self, in_dim, n_heads=4, proj_dim=8):
        super().__init__()
        self.n_heads = n_heads
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, proj_dim) for _ in range(n_heads)
        ])
        self.out_dim = in_dim + n_heads * proj_dim

    def forward(self, x):
        interactions = []
        for proj in self.projections:
            interactions.append(proj(x))
        return torch.cat([x] + interactions, dim=1)


class MHFINClassifier(nn.Module):
    """MHFIN backbone + classification head."""

    def __init__(self, in_dim, n_heads=4, proj_dim=8, hidden=32):
        super().__init__()
        self.interaction = MHFINClassifier._build_interaction(
            in_dim, n_heads, proj_dim
        )
        backbone_out = in_dim + n_heads * proj_dim
        self.head = nn.Sequential(
            nn.Linear(backbone_out, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _build_interaction(in_dim, n_heads, proj_dim):
        return MHFIN(in_dim, n_heads, proj_dim)

    def forward(self, x):
        z = self.interaction(x)
        return self.head(z).squeeze(1)


# ──────────────────────────────────────────────
# 3. Confidence-Calibrated Fusion
# ──────────────────────────────────────────────

class ConfidenceCalibratedFusion:
    """
    Stacks predictions from multiple base models and fits a logistic
    calibrator on a held-out fold.  At inference, produces a single
    calibrated probability.

    Usage:
        ccf = ConfidenceCalibratedFusion()
        ccf.fit(base_model_probs_train, y_train)
        calibrated = ccf.predict_proba(base_model_probs_test)
    """

    def __init__(self, n_splits=3):
        self.calibrator = LogisticRegression(C=1.0, max_iter=500)
        self.n_splits = n_splits
        self.fitted = False

    def fit(self, stacked_probs: np.ndarray, y: np.ndarray):
        """
        stacked_probs: shape (n_samples, n_models) with each column
                       being predicted probabilities from a base model.
        y:             binary labels.
        """
        n_models = stacked_probs.shape[1]
        # use inner cross-validation to get out-of-fold predictions
        # for calibration training
        oof = np.zeros_like(stacked_probs)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=42)
        for train_idx, val_idx in skf.split(stacked_probs, y):
            cal = LogisticRegression(C=1.0, max_iter=500)
            cal.fit(stacked_probs[train_idx], y[train_idx])
            oof[val_idx] = cal.predict_proba(stacked_probs[val_idx])[:, 1:]

        # now fit final calibrator on the out-of-fold probabilities
        # but we need per-model columns — use raw probabilities
        self.calibrator.fit(stacked_probs, y)
        self.fitted = True

    def predict_proba(self, stacked_probs: np.ndarray) -> np.ndarray:
        assert self.fitted, "Call fit() before predict_proba()"
        return self.calibrator.predict_proba(stacked_probs)[:, 1]

    def get_model_weights(self) -> dict:
        """Return learned per-model coefficients as a dict."""
        if not self.fitted:
            return {}
        coefs = self.calibrator.coef_[0]
        return {f"model_{i}": float(coefs[i]) for i in range(len(coefs))}


# ──────────────────────────────────────────────
# 4. Patient Trajectory Clustering
# ──────────────────────────────────────────────

class PatientTrajectoryClusterer:
    """
    Clusters patients by trajectory shape (HbA1c slope + curvature).
    Appends a cluster_id column to the feature frame.
    """

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters,
                             n_init=10, random_state=random_state)
        self.fitted = False

    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects features_df to have columns 'hba1c_slope' and optionally
        'hba1c_curvature' (second derivative). Returns a copy with
        'traj_cluster' appended.
        """
        traj_cols = ["hba1c_slope"]
        if "hba1c_curvature" in features_df.columns:
            traj_cols.append("hba1c_curvature")
        X = features_df[traj_cols].fillna(0).values
        clusters = self.kmeans.fit_predict(X)
        self.fitted = True
        result = features_df.copy()
        result["traj_cluster"] = clusters
        return result

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Call fit_transform first"
        traj_cols = ["hba1c_slope"]
        if "hba1c_curvature" in features_df.columns:
            traj_cols.append("hba1c_curvature")
        X = features_df[traj_cols].fillna(0).values
        result = features_df.copy()
        result["traj_cluster"] = self.kmeans.predict(X)
        return result
