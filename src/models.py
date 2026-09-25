"""
Model definitions and training routines.

Baseline classifiers (logistic regression, random forest, XGBoost, MLP)
plus graph-based models.  The novel components live in novel.py and are
integrated through run_pipeline.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .metrics import calculate_metrics

FEATURES = ["age", "hba1c", "glucose", "bmi", "systolic_bp", "diastolic_bp"]


def make_patient_features(data):
    """
    Aggregate per-patient longitudinal data into a flat feature matrix.
    Uses first 3 visits to compute mean/std and slope features.
    """
    ordered = data.sort_values(["patient_id", "visit"])

    # per-patient mean and std across first 3 visits
    base = (
        ordered.groupby("patient_id")
        .head(3)
        .groupby("patient_id")[FEATURES]
        .agg(["mean", "std"])
        .fillna(0)
    )
    base.columns = [f"{feat}_{stat}" for feat, stat in base.columns]

    # slopes via linear fit on visit index
    slope_rows = []
    for pid, grp in ordered.groupby("patient_id"):
        early = grp.head(3)
        if len(early) >= 2:
            hb_slope = float(np.polyfit(early.visit, early.hba1c, 1)[0])
            gl_slope = float(np.polyfit(early.visit, early.glucose, 1)[0])
        else:
            hb_slope = 0.0
            gl_slope = 0.0

        # crude curvature: second visit slope minus first visit slope
        if len(early) >= 3:
            hb_curv = float(
                np.polyfit(early.visit, early.hba1c, 2)[0]
            )
        else:
            hb_curv = 0.0

        slope_rows.append({
            "patient_id": pid,
            "hba1c_slope": hb_slope,
            "glucose_slope": gl_slope,
            "hba1c_curvature": hb_curv,
        })

    result = base.join(pd.DataFrame(slope_rows).set_index("patient_id"))
    result["target"] = ordered.groupby("patient_id").progression.first()
    return result


def patient_split(features, seed=42):
    from sklearn.model_selection import train_test_split
    patient_ids = list(features.index)
    return train_test_split(
        patient_ids, test_size=0.25, random_state=seed,
        stratify=features.target,
    )


def run_baselines(features, train_ids, test_ids):
    """Standard ML baselines on tabular patient features."""
    x = features.drop(columns="target")
    y = features.target
    scaler = StandardScaler().fit(x.loc[train_ids])
    train_x = scaler.transform(x.loc[train_ids])
    test_x = scaler.transform(x.loc[test_ids])

    # deliberately not hyperparameter-tuned to the teeth;
    # these are reasonable defaults, not grid-searched optima
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120, random_state=42, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=600, random_state=42,
        ),
    }

    records = []
    for name, model in models.items():
        model.fit(train_x, y.loc[train_ids])
        preds = model.predict_proba(test_x)[:, 1]
        records.append(calculate_metrics(y.loc[test_ids], preds, name))
    return records


# ──────────────────────────────────────────────
# Basic graph model (no KG, no temporal gate)
# ──────────────────────────────────────────────

class SimpleGraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=24):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)
        self.attention = torch.nn.Linear(input_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        messages = x[col]
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(1)
        aggregate = torch.zeros_like(x).index_add(0, row, messages)
        degree = torch.zeros(x.size(0), device=x.device).index_add(
            0, row, torch.ones_like(row, dtype=torch.float32)
        )
        attended = x + aggregate / degree.clamp_min(1).unsqueeze(1)
        attention = torch.sigmoid(self.attention(attended)).squeeze(1)
        hidden = torch.relu(self.linear1(attended * attention.unsqueeze(1)))
        return self.linear2(hidden).squeeze(1)


def build_graph_edges(features, knowledge):
    """
    Construct adjacency from patient features.
    knowledge=True  → connect patients with similar HbA1c trajectories
                       (KG-inspired edges, explicitly marked as assumptions).
    knowledge=False → connect by BMI proximity (plain similarity).
    """
    ids = list(features.index)
    index = {pid: i for i, pid in enumerate(ids)}
    edges, weights = [], []

    for left_idx, left_id in enumerate(ids):
        for right_id in ids[left_idx + 1:]:
            left_row = features.loc[left_id]
            right_row = features.loc[right_id]

            if knowledge:
                # KG-style: patients with elevated HbA1c get connected
                if left_row["hba1c_mean"] > 6.0 and right_row["hba1c_mean"] > 6.0:
                    edges.extend([
                        (index[left_id], index[right_id]),
                        (index[right_id], index[left_id]),
                    ])
                    weights.extend([1.0, 1.0])
            else:
                # plain similarity on BMI
                if abs(left_row["bmi_mean"] - right_row["bmi_mean"]) < 1.5:
                    edges.extend([
                        (index[left_id], index[right_id]),
                        (index[right_id], index[left_id]),
                    ])
                    weights.extend([0.5, 0.5])

    # fallback: ring graph if nothing connected
    if not edges:
        edges = [
            (index[ids[i]], index[ids[(i + 1) % len(ids)]])
            for i in range(len(ids))
        ]
        weights = [0.5] * len(edges)

    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(weights, dtype=torch.float32),
    )


def train_graph_model(features, train_ids, knowledge=False,
                      longitudinal=True, seed=42):
    torch.manual_seed(seed)
    columns = list(features.drop(columns="target").columns)
    if not longitudinal:
        columns = [c for c in columns if "slope" not in c]

    scaler = StandardScaler().fit(features.loc[train_ids, columns])
    x_np = scaler.transform(features[columns])
    ids = list(features.index)
    index = {pid: i for i, pid in enumerate(ids)}
    edge_index, edge_weight = build_graph_edges(features, knowledge)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(features.target.to_numpy(), dtype=torch.float32)
    train_set = set(train_ids)
    train_mask = torch.tensor([pid in train_set for pid in ids])

    model = SimpleGraphModel(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    for _ in range(160):
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask]
        )
        loss.backward()
        optimizer.step()

    return model, scaler, columns, edge_index, edge_weight


def predict_graph_model(model, scaler, columns, x_frame,
                        edge_index, edge_weight):
    x = torch.tensor(scaler.transform(x_frame[columns]), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x, edge_index, edge_weight)
        probabilities = torch.sigmoid(logits).numpy()
        attended = x + (
            torch.zeros_like(x).index_add(
                0, edge_index[0],
                x[edge_index[1]] * edge_weight.unsqueeze(1),
            )
            / torch.zeros(x.size(0)).index_add(
                0, edge_index[0],
                torch.ones_like(edge_index[0], dtype=torch.float32),
            ).clamp_min(1).unsqueeze(1)
        )
        attention_weights = torch.sigmoid(
            model.attention(attended)
        ).squeeze(1).numpy()
    return probabilities, attention_weights


def run_graph_model(features, train_ids, test_ids, knowledge=False,
                    longitudinal=True, seed=42, model_name=None,
                    return_artifact=False):
    model, scaler, columns, edge_index, edge_weight = train_graph_model(
        features, train_ids, knowledge, longitudinal, seed
    )
    probs, attn = predict_graph_model(
        model, scaler, columns, features.drop(columns="target"),
        edge_index, edge_weight,
    )
    ids = list(features.index)
    index = {pid: i for i, pid in enumerate(ids)}
    test_idx = [index[pid] for pid in test_ids]
    result = calculate_metrics(
        features.target.iloc[test_idx].to_numpy(),
        probs[test_idx],
        model_name or ("gnn_kg" if knowledge else "gnn"),
    )
    result["example_patient"] = test_ids[0]
    result["example_probability"] = float(probs[test_idx[0]])
    result["example_attention"] = float(attn[test_idx[0]])
    if return_artifact:
        return result, (model, scaler, columns)
    return result
