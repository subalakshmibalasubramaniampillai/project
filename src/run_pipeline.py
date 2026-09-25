"""
Main pipeline entry point.

Orchestrates: data loading → feature engineering → graph construction →
model training → evaluation → artifact saving.

Supports three data paths:
  1. Synthetic cohort (default when no data provided)
  2. Real public dataset (Pima Indians)
  3. User-provided CSV (via --input or DATA_SOURCE env)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .data_source import load_dataset, prepare_dataset
from .generate_data import generate_dataset
from .knowledge_graph import build_knowledge_graph
from .metrics import save_json
from .models import (
    make_patient_features, patient_split, run_baselines, run_graph_model,
)
from .novel import (
    ConfidenceCalibratedFusion, MHFIN, PatientTrajectoryClusterer, TAGNN,
)


def _train_tagnn(features, train_ids, test_ids, columns, seed=42):
    """Train the novel TAGNN model and return metrics + artifact."""
    from .novel import TAGNN
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(features.loc[train_ids, columns])
    x_np = scaler.transform(features[columns])
    ids = list(features.index)
    index = {pid: i for i, pid in enumerate(ids)}

    from .models import build_graph_edges
    edge_index, edge_weight = build_graph_edges(features, knowledge=True)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(features.target.to_numpy(), dtype=torch.float32)
    train_set = set(train_ids)
    train_mask = torch.tensor([pid in train_set for pid in ids])

    model = TAGNN(x.shape[1], hid_dim=32, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask]
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(x, edge_index, edge_weight)).numpy()

    test_idx = [index[pid] for pid in test_ids]
    from .metrics import calculate_metrics
    result = calculate_metrics(
        features.target.iloc[test_idx].to_numpy(),
        probs[test_idx],
        "tagnn",
    )
    result["example_patient"] = test_ids[0]
    result["example_probability"] = float(probs[test_idx[0]])
    return result, (model, scaler, columns, edge_index, edge_weight)


def _train_mhfin(features, train_ids, test_ids, columns, seed=42):
    """Train the Multi-Head Feature Interaction Network."""
    from .novel import MHFINClassifier
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(features.loc[train_ids, columns])
    train_x = torch.tensor(
        scaler.transform(features.loc[train_ids, columns]),
        dtype=torch.float32,
    )
    test_x = torch.tensor(
        scaler.transform(features.loc[test_ids, columns]),
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        features.loc[train_ids, "target"].to_numpy(), dtype=torch.float32
    )
    y_test = features.loc[test_ids, "target"].to_numpy()

    model = MHFINClassifier(len(columns), n_heads=4, proj_dim=6, hidden=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

    for _ in range(150):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_train
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(test_x)).numpy()

    from .metrics import calculate_metrics
    result = calculate_metrics(y_test, probs, "mhfin")
    return result, model


def _run_ccf(base_probs_train, base_probs_test, y_train, y_test):
    """Run Confidence-Calibrated Fusion on stacked base model outputs."""
    from .novel import ConfidenceCalibratedFusion
    ccf = ConfidenceCalibratedFusion(n_splits=3)
    ccf.fit(base_probs_train, y_train)
    calibrated = ccf.predict_proba(base_probs_test)
    from .metrics import calculate_metrics
    result = calculate_metrics(y_test, calibrated, "ccf_fusion")
    result["model_weights"] = ccf.get_model_weights()
    return result


def run(source: str | None = None, data=None, use_real_data: bool = False):
    """
    Full pipeline execution.

    Parameters
    ----------
    source : str, optional
        Path or URL to a CSV file.
    data : DataFrame, optional
        Pre-loaded dataframe.
    use_real_data : bool
        If True and no source/data given, try to download Pima dataset.
    """
    # ── 1. data loading ──────────────────────────
    data_source_label = "synthetic"
    if data is not None:
        data = prepare_dataset(data, "data/longitudinal_diabetes.csv")
        data_source_label = "user-provided"
    elif source:
        data = load_dataset(source, "data/longitudinal_diabetes.csv")
        data_source_label = "user-provided"
    elif use_real_data:
        try:
            from .datasets import load_pima
            data = load_pima("data/longitudinal_diabetes.csv")
            data_source_label = "pima-indians"
            print(f"Loaded Pima Indians dataset: "
                  f"{data.patient_id.nunique()} patients, "
                  f"{len(data)} rows")
        except Exception as e:
            print(f"Could not load Pima ({e}), falling back to synthetic")
            data = generate_dataset()
    else:
        data = generate_dataset()

    n_patients = data.patient_id.nunique()
    n_visits = len(data)
    print(f"Data ready: {n_patients} patients, {n_visits} visits "
          f"({data_source_label})")

    # ── 2. knowledge graph ───────────────────────
    graph = build_knowledge_graph(data_path="data/longitudinal_diabetes.csv")
    print(f"Knowledge graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges")

    # ── 3. feature engineering ───────────────────
    features = make_patient_features(data)
    train_ids, test_ids = patient_split(features)
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test patients")

    # ── 4. trajectory clustering (novel) ─────────
    ptc = PatientTrajectoryClusterer(n_clusters=5)
    features = ptc.fit_transform(features)
    # add cluster one-hot to features for tabular models
    cluster_dummies = (
        pd.get_dummies(features["traj_cluster"], prefix="traj")
    )
    features = pd.concat([features, cluster_dummies], axis=1)
    extended_columns = list(features.drop(columns="target").columns)

    # ── 5. baseline classifiers ──────────────────
    baseline_records = run_baselines(features, train_ids, test_ids)
    print(f"Baselines done: {[r['model'] for r in baseline_records]}")

    # ── 6. graph models (standard) ───────────────
    gnn_record = run_graph_model(
        features, train_ids, test_ids,
        knowledge=False, longitudinal=False, model_name="gnn",
    )
    kg_record = run_graph_model(
        features, train_ids, test_ids,
        knowledge=True, longitudinal=False, model_name="gnn_kg",
    )
    long_record = run_graph_model(
        features, train_ids, test_ids,
        knowledge=True, longitudinal=True,
        model_name="gnn_kg_longitudinal", return_artifact=True,
    )
    long_result, long_artifact = long_record

    # ── 7. novel models ──────────────────────────
    # TAGNN
    tagnn_result, tagnn_artifact = _train_tagnn(
        features, train_ids, test_ids, extended_columns,
    )
    print(f"TAGNN F1={tagnn_result['f1']:.3f}  "
          f"ROC-AUC={tagnn_result['roc_auc']:.3f}")

    # MHFIN
    mhfin_result, mhfin_model = _train_mhfin(
        features, train_ids, test_ids, extended_columns,
    )
    print(f"MHFIN F1={mhfin_result['f1']:.3f}  "
          f"ROC-AUC={mhfin_result['roc_auc']:.3f}")

    # ── 8. Confidence-Calibrated Fusion ──────────
    # stack predictions from the best models on test set
    ids = list(features.index)
    index = {pid: i for i, pid in enumerate(ids)}
    test_idx = [index[pid] for pid in test_ids]

    scaler_temp = StandardScaler().fit(
        features.loc[train_ids, extended_columns]
    )
    test_x_np = scaler_temp.transform(features.loc[test_ids, extended_columns])
    test_x_t = torch.tensor(test_x_np, dtype=torch.float32)

    # gather predictions from available models
    stack_test = []
    stack_train = []

    # run each model on train to get stack_train
    train_x_np = scaler_temp.transform(
        features.loc[train_ids, extended_columns]
    )
    train_x_t = torch.tensor(train_x_np, dtype=torch.float32)

    # simple fallback: use baseline probs
    x_all = features.drop(columns="target")
    scaler_full = StandardScaler().fit(x_all.loc[train_ids])
    all_x = scaler_full.transform(x_all)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    import warnings
    warnings.filterwarnings("ignore")

    simple_models = [
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=80, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=80, max_depth=3,
                               eval_metric="logloss", random_state=42)),
    ]

    y_train_np = features.loc[train_ids, "target"].to_numpy()
    y_test_np = features.loc[test_ids, "target"].to_numpy()

    for name, m in simple_models:
        m.fit(all_x[train_idx := [index[pid] for pid in train_ids]],
              y_train_np)
        stack_train.append(m.predict_proba(
            all_x[[index[pid] for pid in train_ids]]
        )[:, 1])
        stack_test.append(
            m.predict_proba(all_x[[index[pid] for pid in test_ids]])[:, 1]
        )

    # add GNN predictions to the stack
    from .models import train_graph_model, predict_graph_model
    gnn_m, gnn_s, gnn_c, gnn_ei, gnn_ew = train_graph_model(
        features, train_ids, knowledge=False, longitudinal=False
    )
    gnn_probs_all, _ = predict_graph_model(
        gnn_m, gnn_s, gnn_c,
        features.drop(columns="target"), gnn_ei, gnn_ew,
    )
    stack_train.append(gnn_probs_all[[index[pid] for pid in train_ids]])
    stack_test.append(gnn_probs_all[[index[pid] for pid in test_ids]])

    stack_train_arr = np.column_stack(stack_train)
    stack_test_arr = np.column_stack(stack_test)

    ccf_result = _run_ccf(
        stack_train_arr, stack_test_arr, y_train_np, y_test_np,
    )
    print(f"CCF Fusion F1={ccf_result['f1']:.3f}  "
          f"ROC-AUC={ccf_result['roc_auc']:.3f}")

    # ── 9. save artifacts ────────────────────────
    all_records = (
        baseline_records
        + [gnn_record, kg_record, long_result, tagnn_result, mhfin_result,
           ccf_result]
    )

    # save model metrics
    save_json(all_records, "outputs/model_metrics.json")

    # save ablation
    stage_map = {
        "logistic_regression": "baseline", "random_forest": "baseline",
        "xgboost": "baseline", "mlp": "baseline",
        "gnn": "gnn", "gnn_kg": "gnn+kg",
        "gnn_kg_longitudinal": "gnn+kg+longitudinal",
        "tagnn": "novel-tagnn", "mhfin": "novel-mhfin",
        "ccf_fusion": "novel-ccf",
    }
    save_json(
        [{**r, "ablation_stage": stage_map.get(r["model"], "other")}
         for r in all_records],
        "outputs/ablation_metrics.json",
    )

    # save GNN+KG+longitudinal artifact
    model, scaler, columns = long_artifact
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": len(columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "columns": columns,
    }, "outputs/gnn_kg_longitudinal.pt")

    # save TAGNN artifact
    tagnn_m, tagnn_s, tagnn_c, tagnn_ei, tagnn_ew = tagnn_artifact
    torch.save({
        "state_dict": tagnn_m.state_dict(),
        "input_dim": len(tagnn_c),
        "scaler_mean": tagnn_s.mean_.tolist(),
        "scaler_scale": tagnn_s.scale_.tolist(),
        "columns": tagnn_c,
    }, "outputs/tagnn.pt")

    features.reset_index().to_csv(
        "outputs/patient_features.csv", index=False
    )

    Path("outputs/split.json").write_text(
        json.dumps({
            "split": "patient-level stratified 75/25",
            "train_patients": len(train_ids),
            "test_patients": len(test_ids),
            "data_source": data_source_label,
            "n_patients": n_patients,
            "n_visits": n_visits,
        }),
        encoding="utf-8",
    )

    # explanation for one test patient
    sample_id = str(test_ids[0])
    sample = features.loc[sample_id]
    explanation = {
        "patient_id": sample_id,
        "model": "tagnn",
        "prediction_probability": tagnn_result["example_probability"],
        "model_contribution": [
            {"feature": "hba1c_slope",
             "value": float(sample.hba1c_slope)},
            {"feature": "glucose_slope",
             "value": float(sample.glucose_slope)},
            {"feature": "traj_cluster",
             "value": int(sample.traj_cluster) if "traj_cluster" in sample.index else 0},
        ],
        "interpretation": (
            "Feature contribution probe for this patient. Values are the "
            "feature values used by the model for this prediction."
        ),
    }
    Path("outputs/explanation.json").write_text(
        json.dumps(explanation, indent=2), encoding="utf-8"
    )

    # save CCF weights
    save_json([ccf_result], "outputs/ccf_weights.json")

    print(f"\nPipeline complete. {len(all_records)} models evaluated.")
    print(f"Best F1: {max(r['f1'] for r in all_records):.3f} "
          f"({max(all_records, key=lambda r: r['f1'])['model']})")
    return all_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train from generated, real, or user-provided data."
    )
    parser.add_argument(
        "--input",
        help="CSV path or HTTP CSV URL; omit for generated demo cohort",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Use Pima Indians Diabetes dataset instead of synthetic",
    )
    args = parser.parse_args()

    if args.real:
        run(use_real_data=True)
    else:
        run(source=args.input)
