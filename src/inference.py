
import numpy as np
import pandas as pd
import torch

from .knowledge_graph import build_patient_subgraph
from .models import SimpleGraphModel, predict_graph_model


def prepare_inference_features(visits: pd.DataFrame) -> tuple[pd.DataFrame, str]:

    visits = visits.sort_values("visit")
    is_baseline = len(visits) < 2
    window = visits if is_baseline else visits.head(3)
    features = {}
    for col in ["age", "hba1c", "glucose", "bmi", "systolic_bp",
                "diastolic_bp"]:
        features[f"{col}_mean"] = window[col].mean()
        features[f"{col}_std"] = window[col].std(ddof=1) if len(window) > 1 else 0.0
    if len(window) >= 2:
        features["hba1c_slope"] = float(
            np.polyfit(window.visit, window.hba1c, 1)[0]
        )
        features["glucose_slope"] = float(
            np.polyfit(window.visit, window.glucose, 1)[0]
        )
        features["hba1c_curvature"] = float(
            np.polyfit(window.visit, window.hba1c, 2)[0]
        )
    else:
        features["hba1c_slope"] = 0.0
        features["glucose_slope"] = 0.0
        features["hba1c_curvature"] = 0.0

    mode = "baseline" if is_baseline else "longitudinal"
    return pd.DataFrame([features]), mode


def _load_scaler(artifact):
    mean = np.asarray(artifact["scaler_mean"])
    scale = np.asarray(artifact["scaler_scale"])

    class _Scaler:
        def transform(self, frame):
            arr = frame.to_numpy() if hasattr(frame, "to_numpy") else np.asarray(frame)
            return (arr - mean) / scale

    return _Scaler()


def infer_new_patient(
    visits: pd.DataFrame,
    artifact_path: str = "outputs/gnn_kg_longitudinal.pt",
    use_tagnn: bool = False,
) -> dict:
    features, mode = prepare_inference_features(visits)

    if use_tagnn:
        try:
            return _infer_tagnn(features, mode, "outputs/tagnn.pt")
        except FileNotFoundError:
            pass  # fall back to standard model

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)

    model = SimpleGraphModel(artifact["input_dim"])
    model.load_state_dict(artifact["state_dict"])
    scaler = _load_scaler(artifact)

    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0], dtype=torch.float32)

    probs, attn = predict_graph_model(
        model, scaler, artifact["columns"],
        features, edge_index, edge_weight,
    )
    subgraph = build_patient_subgraph(visits)

    return {
        "mode": mode,
        "model": "gnn_kg_longitudinal",
        "probability": float(probs[0]),
        "attention_weight": float(attn[0]),
        "model_contribution": [
            {"feature": "node_attention", "value": float(attn[0])},
        ],
        "subgraph_nodes": subgraph.number_of_nodes(),
        "subgraph_edges": subgraph.number_of_edges(),
        "interpretation": (
            "Baseline/current-state prediction for a single patient. "
            "The contribution value is the model's internal attention weight "
            "for this patient's node."
        ),
    }


def _infer_tagnn(features, mode, artifact_path):
    """Load TAGNN and run inference."""
    from .novel import TAGNN
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    model = TAGNN(artifact["input_dim"], hid_dim=32, n_layers=2)
    model.load_state_dict(artifact["state_dict"])
    scaler = _load_scaler(artifact)

    # TAGNN was trained on features that include trajectory-cluster
    # columns; backfill any that a new patient does not provide.
    inference_frame = features.copy()
    for col in artifact["columns"]:
        if col not in inference_frame.columns:
            inference_frame[col] = 0.0

    x = torch.tensor(
        scaler.transform(inference_frame[artifact["columns"]]),
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        logit = model(x, edge_index, edge_weight)
        prob = torch.sigmoid(logit).item()

    return {
        "mode": mode,
        "model": "tagnn",
        "probability": prob,
        "attention_weight": 0.0,
        "model_contribution": [
            {"feature": "tagnn_logit", "value": float(logit)},
        ],
        "subgraph_nodes": 0,
        "subgraph_edges": 0,
        "interpretation": (
            "TAGNN prediction for this patient. Temporal attention is "
            "computed over the patient's graph context during training."
        ),
    }
