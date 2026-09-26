"""
Single-patient inference with the deployed TKGN-B model.

Input: a list of encounters for one patient in the original UCI schema
(any missing field takes a neutral default), ordered oldest first.  The
last encounter is the index encounter; its ``readmitted`` value is never
used.  Output: calibrated risk from TKGN-B, the GBDT-only anchor risk,
history reliability gates, attention over earlier encounters and the
largest TreeSHAP contributions of the GBDT anchor.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CATEGORICAL_COLUMNS, DIAG_COLUMNS, NUMERIC_COLUMNS
from .icd9 import DRUG_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"

DEFAULTS = {
    **{c: 0 for c in NUMERIC_COLUMNS},
    **{c: np.nan for c in CATEGORICAL_COLUMNS},
    **{c: np.nan for c in DIAG_COLUMNS},
    **{d: "No" for d in DRUG_COLUMNS},
    "readmitted": "NO", "change": "No", "diabetesMed": "No",
    "gender": "Female", "age": "[60-70)", "race": "Caucasian",
    "admission_type_id": 1, "discharge_disposition_id": 1,
    "admission_source_id": 7, "number_diagnoses": 1, "time_in_hospital": 1,
}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@lru_cache(maxsize=4)
def load_predictor(task: str):
    import joblib
    from .neural import build_model
    ctx_path = MODELS_DIR / f"{task}_tkgn_b_context.joblib"
    if not ctx_path.exists():
        raise FileNotFoundError(
            f"{ctx_path.name} not found - run `python -m src.run_pipeline` first")
    ctx = joblib.load(ctx_path)
    model = build_model("tkgn", ctx["kg"], ctx["seq"],
                        max_history=ctx["max_history"], dropout=ctx["dropout"])
    model.load_state_dict(torch.load(MODELS_DIR / f"{task}_tkgn_b.pt",
                                     map_location="cpu"))
    model.eval()
    return model, ctx


def _frame(encounters: list[dict]) -> pd.DataFrame:
    rows = []
    for i, enc in enumerate(encounters):
        row = dict(DEFAULTS)
        for key, value in enc.items():
            if value is None or value == "":
                continue
            row[key] = value
        row["encounter_id"] = i + 1
        row["patient_nbr"] = 1
        rows.append(row)
    frame = pd.DataFrame(rows)
    for col in NUMERIC_COLUMNS + ["admission_type_id",
                                  "discharge_disposition_id",
                                  "admission_source_id"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    for col in DIAG_COLUMNS:
        frame[col] = frame[col].map(lambda v: str(v).strip()
                                    if isinstance(v, (str, int, float))
                                    and str(v).strip() not in ("", "nan") else np.nan)
    return frame


def predict_patient(encounters: list[dict], task: str = "readmit30") -> dict:
    from .data import build_cohort, history_index
    from .neural import SequenceData

    if not encounters:
        raise ValueError("at least one encounter is required")
    model, ctx = load_predictor(task)
    cohort = build_cohort(raw=_frame(encounters), apply_exclusions=False)
    index = len(cohort) - 1

    x = ctx["tab"].transform(cohort).to_numpy()
    gbdt = ctx["gbdt"]
    p_gbdt = float(gbdt.predict_proba(x[index:index + 1])[:, 1][0])
    contrib = gbdt.predict_proba(x[index:index + 1], pred_contrib=True)[0, :-1]
    top = np.argsort(-np.abs(contrib))[:8]

    arrays = ctx["seq"].transform(cohort, ctx["kg"].encode_diagnoses(cohort))
    history = history_index(cohort, ctx["max_history"])
    offset = np.full(len(cohort), _logit(p_gbdt), dtype=np.float32)
    data = SequenceData(arrays, history, np.zeros(len(cohort), np.float32),
                        offset=offset)
    with torch.no_grad():
        batch = data.batch(np.array([index]))
        logit, extras = model(batch)
        prob = float(torch.sigmoid(logit + batch["offset"])[0])

    n_hist = int((history[index] >= 0).sum())
    attention = []
    if n_hist and "attention" in extras:
        weights = extras["attention"][0].numpy()[-n_hist:]
        attention = [{"encounter": i + 1, "weight": float(w)}
                     for i, w in enumerate(weights[-n_hist:])]
    return {
        "task": task,
        "model": "TKGN-B",
        "probability": prob,
        "gbdt_anchor_probability": p_gbdt,
        "prior_encounters": n_hist,
        "gate_history": float(extras.get("gate_history", torch.zeros(1))[0]),
        "gate_delta": float(extras.get("gate_delta", torch.zeros(1))[0]),
        "history_attention": attention,
        "top_contributions": [
            {"feature": ctx["tab"].columns[i], "value": float(x[index, i]),
             "shap": float(contrib[i])} for i in top],
        "note": "Research prototype for decision support; not a diagnostic device.",
    }
