"""
Aggregate experiment runs into the summary artefacts used by the API,
the web dashboard and the manuscript.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .evaluation import (
    bootstrap_ci, calibration_curve, delong_test, evaluate,
    expected_calibration_error, paired_bootstrap, save_json,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNS = OUT / "runs"

METRICS = ["auroc", "auprc", "brier", "ece", "calibration_intercept",
           "calibration_slope", "sensitivity", "specificity", "ppv", "f1",
           "balanced_accuracy"]
DISPLAY = {
    "logistic_regression": "Logistic regression", "random_forest": "Random forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
    "gru": "GRU", "retain": "RETAIN", "transformer": "Transformer",
    "tkgn": "TKGN (proposed)", "tkgn_b": "TKGN-B (proposed)",
    "tkgn_no_kg": "TKGN - knowledge graph",
    "tkgn_no_cooccurrence": "TKGN - co-morbidity edges",
    "tkgn_no_history": "TKGN - history",
    "tkgn_no_gate": "TKGN - reliability gate",
    "tkgn_no_delta": "TKGN - delta encoding",
    "tkgn_no_prior_outcomes": "TKGN - prior outcomes",
    "tkgn_no_aux": "TKGN - auxiliary head",
}
MAIN_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
               "mlp", "gru", "retain", "transformer", "tkgn", "tkgn_b"]


def _load_runs():
    runs = []
    for path in sorted(RUNS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["name"] = path.stem
        runs.append(record)
    return runs


def _npz(name):
    return np.load(RUNS / f"{name}.npz")


def _summary(runs):
    rows = []
    groups = defaultdict(list)
    for r in runs:
        if r["train_fraction"] < 1.0:
            continue
        for model, res in r["models"].items():
            if "test" in res:
                groups[(r["task"], r["split"], model)].append(res["test"])
    for (task, split, model), results in sorted(groups.items()):
        row = {"task": task, "split": split, "model": model,
               "label": DISPLAY.get(model, model), "n_runs": len(results)}
        for m in METRICS:
            values = np.array([res[m] for res in results], dtype=float)
            row[m] = float(values.mean())
            row[f"{m}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def _repeat_tests(runs, reference):
    """Per-repeat paired comparison of AUROC (reference vs each model)."""
    out = []
    by_key = defaultdict(dict)
    for r in runs:
        if r["train_fraction"] < 1.0:
            continue
        for model, res in r["models"].items():
            if "test" in res:
                by_key[(r["task"], r["split"])].setdefault(model, {})[r["repeat"]] = \
                    res["test"]["auroc"]
    for (task, split), models in sorted(by_key.items()):
        if reference not in models:
            continue
        ref = models[reference]
        for model, values in models.items():
            if model == reference or model not in MAIN_MODELS:
                continue
            common = sorted(set(ref) & set(values))
            diffs = np.array([ref[k] - values[k] for k in common])
            if len(diffs) < 2:
                continue
            t = stats.ttest_1samp(diffs, 0.0)
            out.append({"task": task, "split": split, "reference": reference,
                        "model": model, "n_repeats": len(diffs),
                        "mean_delta_auroc": float(diffs.mean()),
                        "wins": int((diffs > 0).sum()),
                        "t_test_p": float(t.pvalue)})
    return out


def _significance(runs, cohort):
    out = {}
    for r in runs:
        if r["repeat"] != 0 or r["train_fraction"] < 1.0:
            continue
        data = _npz(r["name"])
        y = data["y"].astype(int)
        groups = cohort["patient_nbr"].to_numpy()[data["test_rows"]]
        preds = {k[2:]: data[k] for k in data.files if k.startswith("p_")}
        entry = {"ci": {}, "vs_tkgn_b": {}, "vs_tkgn": {}}
        for model, p in preds.items():
            entry["ci"][model] = bootstrap_ci(y, p, groups, n_boot=1000)
        for ref in ("tkgn_b", "tkgn"):
            if ref not in preds:
                continue
            for model, p in preds.items():
                if model == ref or model not in MAIN_MODELS:
                    continue
                entry[f"vs_{ref}"][model] = {
                    "delong": delong_test(y, preds[ref], p),
                    "bootstrap": paired_bootstrap(y, preds[ref], p, groups,
                                                  n_boot=1000),
                }
        out[f"{r['task']}_{r['split']}"] = entry
    return out


def _age_band(age):
    lower = int(age.strip("[)").split("-")[0])
    if lower < 50:
        return "<50"
    if lower < 70:
        return "50-69"
    return "70+"


def _subgroups(runs, cohort):
    out = []
    for r in runs:
        if r["repeat"] != 0 or r["split"] != "grouped" or r["train_fraction"] < 1.0:
            continue
        data = _npz(r["name"])
        rows = data["test_rows"]
        y = data["y"].astype(int)
        sub = cohort.iloc[rows]
        definitions = {
            "race": sub["race"].fillna("Unknown").to_numpy(),
            "gender": sub["gender"].to_numpy(),
            "age": sub["age"].map(_age_band).to_numpy(),
            "prior_encounters": np.where(sub["order"] == 0, "0",
                                         np.where(sub["order"] == 1, "1", "2+")),
            # the release records "None" when HbA1c was not measured
            "hba1c_measured": np.where(sub["A1Cresult"].isin(["None"])
                                       | sub["A1Cresult"].isna(),
                                       "not measured", "measured"),
        }
        for model in ["logistic_regression", "lightgbm", "gru", "tkgn", "tkgn_b"]:
            if f"p_{model}" not in data.files:
                continue
            p = data[f"p_{model}"]
            thr = r["models"][model]["test"]["threshold"]
            for attribute, values in definitions.items():
                for level in sorted(set(values)):
                    sel = values == level
                    if sel.sum() < 100 or y[sel].min() == y[sel].max():
                        continue
                    res = evaluate(y[sel], p[sel], thr)
                    out.append({
                        "task": r["task"], "model": model, "attribute": attribute,
                        "group": str(level), "n": int(sel.sum()),
                        "prevalence": res["prevalence"],
                        "mean_predicted": float(p[sel].mean()),
                        "auroc": res["auroc"], "auprc": res["auprc"],
                        "ece": res["ece"], "sensitivity": res["sensitivity"],
                        "specificity": res["specificity"],
                    })
    return out


def _calibration(runs):
    out = {}
    for r in runs:
        if r["repeat"] != 0 or r["train_fraction"] < 1.0:
            continue
        data = _npz(r["name"])
        y = data["y"].astype(int)
        out[f"{r['task']}_{r['split']}"] = {
            k[2:]: calibration_curve(y, data[k]) for k in data.files
            if k.startswith("p_") and k[2:] in MAIN_MODELS
        }
    return out


def _ablation(runs):
    rows = []
    groups = defaultdict(list)
    for r in runs:
        if r["split"] != "grouped" or r["train_fraction"] < 1.0:
            continue
        if "tkgn_no_kg" not in r["models"]:
            continue
        for model, res in r["models"].items():
            if model.startswith("tkgn") and model != "tkgn_b":
                groups[(r["task"], model)].append(
                    (r["repeat"], res["test"]["auroc"], res["test"]["auprc"]))
    for (task, model), values in sorted(groups.items()):
        full = {rep: (a, p) for rep, a, p in groups[(task, "tkgn")]}
        deltas = [a - full[rep][0] for rep, a, _ in values if rep in full]
        rows.append({
            "task": task, "model": model, "label": DISPLAY.get(model, model),
            "n_runs": len(values),
            "auroc": float(np.mean([v[1] for v in values])),
            "auroc_sd": float(np.std([v[1] for v in values], ddof=1))
            if len(values) > 1 else 0.0,
            "auprc": float(np.mean([v[2] for v in values])),
            "delta_auroc_vs_full": float(np.mean(deltas)) if deltas else 0.0,
        })
    return rows


def _learning_curve(runs, cohort):
    rows = []
    for r in runs:
        if r["split"] != "grouped" or r["repeat"] != 0:
            continue
        data = _npz(r["name"])
        y = data["y"].astype(int)
        order = cohort["order"].to_numpy()[data["test_rows"]]
        for model in ["lightgbm", "gru", "tkgn", "tkgn_b"]:
            if f"p_{model}" not in data.files:
                continue
            p = data[f"p_{model}"]
            res = {"task": r["task"], "fraction": r["train_fraction"],
                   "n_train": r["n"]["train"], "model": model,
                   "auroc": evaluate(y, p, 0.5)["auroc"]}
            for label, sel in (("no_history", order == 0),
                               ("with_history", order > 0)):
                res[f"auroc_{label}"] = float(
                    evaluate(y[sel], p[sel], 0.5)["auroc"])
            rows.append(res)
    return sorted(rows, key=lambda d: (d["task"], d["model"], d["fraction"]))


def _gates(runs, cohort):
    out = []
    for r in runs:
        if r["repeat"] != 0 or r["split"] != "grouped" or r["train_fraction"] < 1.0:
            continue
        data = _npz(r["name"])
        if "tkgn_gates" not in data.files:
            continue
        gates = data["tkgn_gates"]
        order = cohort["order"].to_numpy()[data["test_rows"]]
        bucket = np.minimum(order, 5)
        for b in range(6):
            sel = bucket == b
            if not sel.any():
                continue
            out.append({"task": r["task"], "prior_encounters":
                        f"{b}" if b < 5 else "5+", "n": int(sel.sum()),
                        "gate_history": float(gates[sel, 0].mean()),
                        "gate_delta": float(gates[sel, 1].mean())})
    return out


def _explorer(runs, cohort):
    frames = []
    for r in runs:
        if r["repeat"] != 0 or r["split"] != "grouped" or r["train_fraction"] < 1.0:
            continue
        data = _npz(r["name"])
        rows = data["test_rows"]
        frame = pd.DataFrame({
            "task": r["task"],
            "patient_nbr": cohort["patient_nbr"].to_numpy()[rows],
            "encounter_id": cohort["encounter_id"].to_numpy()[rows],
            "order": cohort["order"].to_numpy()[rows],
            "y": data["y"].astype(int),
        })
        for model in ["lightgbm", "tkgn", "tkgn_b"]:
            if f"p_{model}" in data.files:
                frame[f"p_{model}"] = np.round(data[f"p_{model}"], 4)
        if "tkgn_gates" in data.files:
            frame["gate_history"] = np.round(data["tkgn_gates"][:, 0], 4)
            frame["gate_delta"] = np.round(data["tkgn_gates"][:, 1], 4)
        counts = frame.groupby("patient_nbr")["encounter_id"].transform("count")
        frames.append(frame[counts >= 2])
    if frames:
        pd.concat(frames).to_csv(OUT / "explorer_predictions.csv", index=False)


def dataset_summary(cohort):
    from .data import load_raw
    raw_n = len(load_raw(verify=False))
    per_patient = cohort.groupby("patient_nbr").size()
    esc = cohort[cohort["has_next"]]

    def dist(col, frame=cohort):
        vc = frame[col].fillna("Unknown").value_counts()
        return {str(k): int(v) for k, v in vc.items()}

    return {
        "source": "Diabetes 130-US Hospitals 1999-2008 (UCI id 296; "
                  "Strack et al., 2014)",
        "raw_encounters": int(raw_n),
        "cohort_encounters": int(len(cohort)),
        "patients": int(cohort["patient_nbr"].nunique()),
        "patients_with_repeat_encounters": int((per_patient > 1).sum()),
        "encounters_with_history": int((cohort["order"] > 0).sum()),
        "max_encounters_per_patient": int(per_patient.max()),
        "readmit30_rate": float(cohort["y_readmit30"].mean()),
        "escalation_samples": int(len(esc)),
        "escalation_rate": float((esc["y_escalation"] == 1).mean()),
        "encounters_per_patient": {str(k): int(v) for k, v in
                                   per_patient.clip(upper=6).value_counts()
                                   .sort_index().items()},
        "race": dist("race"), "gender": dist("gender"), "age": dist("age"),
        "a1c": dist("A1Cresult"),
        "time_in_hospital_mean": float(cohort["time_in_hospital"].mean()),
        "num_medications_mean": float(cohort["num_medications"].mean()),
        "insulin_any": float(cohort["on_insulin"].mean()),
    }


def aggregate():
    from .data import build_cohort, grouped_split
    from .knowledge_graph import ClinicalKnowledgeGraph

    runs = _load_runs()
    if not runs:
        print("no runs found")
        return
    cohort = build_cohort()
    save_json(dataset_summary(cohort), OUT / "dataset_summary.json")
    kg = ClinicalKnowledgeGraph().fit(
        cohort, grouped_split(cohort, "readmit30", seed=1000)["train"])
    kg.save(OUT)

    save_json(_summary(runs), OUT / "summary.json")
    save_json(_repeat_tests(runs, "tkgn_b") + _repeat_tests(runs, "tkgn"),
              OUT / "repeat_tests.json")
    save_json(_significance(runs, cohort), OUT / "significance.json")
    save_json(_subgroups(runs, cohort), OUT / "subgroups.json")
    save_json(_calibration(runs), OUT / "calibration.json")
    save_json(_ablation(runs), OUT / "ablation.json")
    save_json(_learning_curve(runs, cohort), OUT / "learning_curve.json")
    save_json(_gates(runs, cohort), OUT / "gates.json")
    _explorer(runs, cohort)
    splits = {}
    for r in runs:
        if r["repeat"] == 0 and r["train_fraction"] == 1.0:
            splits[f"{r['task']}_{r['split']}"] = {
                "n": r["n"], "prevalence": r["prevalence"]}
    save_json(splits, OUT / "splits.json")
    print(f"aggregated {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    aggregate()
