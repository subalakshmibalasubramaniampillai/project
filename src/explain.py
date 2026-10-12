"""
Model explanations on the held-out test partition (grouped split,
repeat 0) using the saved deployment models.

* TKGN: permutation importance of clinically meaningful input groups
  (AUROC drop when the group is shuffled across encounters), plus the
  history-attention weights and reliability gates of example patients.
* LightGBM component of TKGN-B: exact TreeSHAP attributions
  (``pred_contrib``), summarised as mean |SHAP| per feature.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .analysis import OUT, RUNS
from .data import CATEGORICAL_COLUMNS
from .evaluation import save_json

CAT_GROUPS = {
    "Admission / discharge context": ["admission_type_id",
                                      "discharge_disposition_id",
                                      "admission_source_id",
                                      "medical_specialty", "payer_code"],
    "Demographics": ["race", "gender", "age"],
    "Glycaemic tests (HbA1c, glucose)": ["A1Cresult", "max_glu_serum"],
    "Medication change flags": ["change", "diabetesMed"],
}


def _load(task, model_name="tkgn"):
    import joblib
    from .neural import build_model
    context = joblib.load(OUT / "models" / f"{task}_{model_name}_context.joblib")
    model = build_model("tkgn", context["kg"], context["seq"],
                        max_history=context["max_history"],
                        dropout=context["dropout"])
    model.load_state_dict(torch.load(OUT / "models" / f"{task}_{model_name}.pt",
                                     map_location="cpu"))
    model.eval()
    return model, context


def permutation_importance(task, n_samples=8000, repeats=3, seed=0):
    from .data import build_cohort, history_index, task_labels
    from .neural import SequenceData, predict

    model, ctx = _load(task)
    cohort = build_cohort()
    history = history_index(cohort, ctx["max_history"])
    y = np.clip(task_labels(cohort, task), 0, 1).astype(np.float32)
    arrays = ctx["seq"].transform(cohort, ctx["kg"].encode_diagnoses(cohort))
    rows = np.load(RUNS / f"{task}_grouped_r0.npz")["test_rows"]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, min(n_samples, len(rows)), replace=False))

    def score(arr, hist=history, prior=True):
        data = SequenceData(arr, hist, y, prior_outcomes=prior)
        return roc_auc_score(y[rows], predict(model, data, rows))

    base = score(arrays)
    cat_index = {c: i for i, c in enumerate(CATEGORICAL_COLUMNS)}
    groups = {
        "Diagnoses (ICD-9 codes)": [("diag", None)],
        "Anti-diabetic medications": [("drugs", None)],
        "Utilisation counts": [("nums", None), ("num_bins", None)],
    }
    for label, cols in CAT_GROUPS.items():
        groups[label] = [("cats", [cat_index[c] for c in cols])]

    results = []
    n = len(cohort)
    for label, fields in groups.items():
        drops = []
        for r in range(repeats):
            perm = np.random.default_rng(seed + r + 1).permutation(n)
            arr = {k: v.copy() for k, v in arrays.items()}
            for key, cols in fields:
                if cols is None:
                    arr[key] = arr[key][perm]
                else:
                    arr[key][:, cols] = arr[key][perm][:, cols]
            drops.append(base - score(arr))
        results.append({"group": label, "auroc_drop": float(np.mean(drops)),
                        "sd": float(np.std(drops))})

    # history: give every encounter the history window of a random other one
    drops = []
    for r in range(repeats):
        perm = np.random.default_rng(seed + 100 + r).permutation(n)
        shuffled = history[perm]
        drops.append(base - score(arrays, hist=shuffled))
    results.append({"group": "Prior encounters (sequence)",
                    "auroc_drop": float(np.mean(drops)), "sd": float(np.std(drops))})
    drop = base - score(arrays, prior=False)
    results.append({"group": "Outcomes of prior encounters",
                    "auroc_drop": float(drop), "sd": 0.0})
    return {"task": task, "base_auroc": float(base), "n": int(len(rows)),
            "groups": sorted(results, key=lambda d: -d["auroc_drop"])}


def treeshap(task, n_samples=5000, seed=0, top=20):
    import joblib
    from .data import build_cohort
    ctx = joblib.load(OUT / "models" / f"{task}_tkgn_b_context.joblib")
    gbdt, tab = ctx["gbdt"], ctx["tab"]
    cohort = build_cohort()
    rows = np.load(RUNS / f"{task}_grouped_r0.npz")["test_rows"]
    rows = np.random.default_rng(seed).choice(rows, min(n_samples, len(rows)),
                                              replace=False)
    # history aggregates need every encounter of a patient, so transform
    # the full cohort and then select the sampled rows
    x = tab.transform(cohort).to_numpy()[rows]
    contrib = gbdt.predict_proba(x, pred_contrib=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(0)
    order = np.argsort(-mean_abs)[:top]
    return [{"feature": tab.columns[i], "mean_abs_shap": float(mean_abs[i]),
             "mean_shap_when_present": float(
                 contrib[x[:, i] != 0, i].mean()) if (x[:, i] != 0).any() else 0.0}
            for i in order]


def example_patients(task, k=3, seed=0):
    """Attention over history and gate values for a few test patients."""
    from .data import build_cohort, history_index
    from .neural import SequenceData

    model, ctx = _load(task)
    cohort = build_cohort()
    history = history_index(cohort, ctx["max_history"])
    arrays = ctx["seq"].transform(cohort, ctx["kg"].encode_diagnoses(cohort))
    rows = np.load(RUNS / f"{task}_grouped_r0.npz")["test_rows"]
    candidates = rows[cohort["order"].to_numpy()[rows] >= 3]
    chosen = np.random.default_rng(seed).choice(candidates, k, replace=False)
    data = SequenceData(arrays, history, np.zeros(len(cohort), np.float32))
    out = []
    with torch.no_grad():
        batch = data.batch(np.sort(chosen))
        logits, extras = model(batch)
    for i, row in enumerate(np.sort(chosen)):
        hist_rows = history[row][history[row] >= 0]
        att = extras["attention"][i].numpy()[-len(hist_rows):]
        out.append({
            "patient_nbr": int(cohort["patient_nbr"].iloc[row]),
            "encounter_id": int(cohort["encounter_id"].iloc[row]),
            "probability": float(torch.sigmoid(logits[i])),
            "gate_history": float(extras["gate_history"][i]),
            "gate_delta": float(extras["gate_delta"][i]),
            "history": [{"encounter_id": int(cohort["encounter_id"].iloc[h]),
                         "primary_diagnosis": str(cohort["diag_1"].iloc[h]),
                         "readmitted": str(cohort["readmitted"].iloc[h]),
                         "attention": float(a)}
                        for h, a in zip(hist_rows, att)],
        })
    return out


def run(tasks=("readmit30", "escalation")):
    out = {}
    for task in tasks:
        if not (OUT / "models" / f"{task}_tkgn.pt").exists():
            continue
        out[task] = {
            "permutation_importance": permutation_importance(task),
            "treeshap": treeshap(task),
            "examples": example_patients(task),
        }
        print(f"explanations for {task} done")
    save_json(out, OUT / "explanations.json")


if __name__ == "__main__":
    run()
