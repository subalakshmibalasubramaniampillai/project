"""
FastAPI backend for the diabetes readmission / treatment-escalation
research system built on the Diabetes 130-US Hospitals data.

Read-only result endpoints (produced by ``python -m src.run_pipeline``):
  GET  /api/overview          dataset summary + split sizes
  GET  /api/summary           mean +/- SD test metrics per task/split/model
  GET  /api/significance      bootstrap CIs, DeLong and paired bootstrap tests
  GET  /api/repeat-tests      per-repeat paired AUROC comparisons
  GET  /api/ablation          TKGN component ablation
  GET  /api/subgroups         fairness / subgroup performance
  GET  /api/calibration       reliability curves
  GET  /api/learning-curve    performance vs training-set size
  GET  /api/gates             TKGN history gates vs number of prior stays
  GET  /api/explanations      permutation importance, TreeSHAP, examples
  GET  /api/graph             knowledge-graph summary
Patient-level endpoints:
  GET  /api/patients          test-set patients with >= 2 encounters
  GET  /api/patients/{id}     their encounters + stored model predictions
  GET  /api/schema            allowed values for the risk form
  POST /api/predict           TKGN-B risk for a posted encounter history

The built React app in ../web/dist is served at / when present.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

app = FastAPI(
    title="Diabetes Encounter Risk Research API",
    description="Real-world EHR research system (Diabetes 130-US Hospitals). "
                "Research and decision support only; not a diagnostic tool.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173"],
    allow_methods=["*"], allow_headers=["*"],
)


def _json(name: str):
    path = OUT / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found - run the pipeline first")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok",
            "purpose": "research and decision support, not diagnosis"}


@app.get("/api/overview")
def overview():
    return {"dataset": _json("dataset_summary.json"),
            "splits": _json("splits.json")}


@app.get("/api/summary")
def summary():
    return _json("summary.json")


@app.get("/api/significance")
def significance():
    return _json("significance.json")


@app.get("/api/repeat-tests")
def repeat_tests():
    return _json("repeat_tests.json")


@app.get("/api/ablation")
def ablation():
    return _json("ablation.json")


@app.get("/api/subgroups")
def subgroups():
    return _json("subgroups.json")


@app.get("/api/calibration")
def calibration():
    return _json("calibration.json")


@app.get("/api/learning-curve")
def learning_curve():
    return _json("learning_curve.json")


@app.get("/api/gates")
def gates():
    return _json("gates.json")


@app.get("/api/explanations")
def explanations():
    return _json("explanations.json")


@app.get("/api/graph")
def graph():
    return _json("knowledge_graph_summary.json")


# ── patients ─────────────────────────────────────
@lru_cache(maxsize=1)
def _cohort():
    from .data import build_cohort
    return build_cohort()


@lru_cache(maxsize=1)
def _explorer():
    path = OUT / "explorer_predictions.csv"
    if not path.exists():
        raise HTTPException(404, "explorer_predictions.csv not found")
    return pd.read_csv(path)


@app.get("/api/patients")
def patients(task: str = "readmit30", limit: int = 300):
    frame = _explorer()
    frame = frame[frame.task == task]
    counts = (frame.groupby("patient_nbr").size()
              .sort_values(ascending=False).head(limit))
    return [{"patient_nbr": int(p), "encounters": int(n)}
            for p, n in counts.items()]


ENCOUNTER_FIELDS = [
    "encounter_id", "order", "age", "gender", "race", "admission_type_id",
    "discharge_disposition_id", "time_in_hospital", "num_medications",
    "num_lab_procedures", "number_inpatient", "number_emergency",
    "number_diagnoses", "diag_1", "diag_2", "diag_3", "A1Cresult",
    "insulin", "change", "readmitted", "y_escalation",
]


@app.get("/api/patients/{patient_nbr}")
def patient(patient_nbr: int, task: str = "readmit30"):
    cohort = _cohort()
    rows = cohort[cohort.patient_nbr == patient_nbr]
    if rows.empty:
        raise HTTPException(404, "patient not found")
    preds = _explorer()
    preds = preds[(preds.task == task) & (preds.patient_nbr == patient_nbr)]
    merged = rows[ENCOUNTER_FIELDS].merge(
        preds.drop(columns=["task", "patient_nbr", "order"]),
        on="encounter_id", how="left")
    merged = merged.replace({np.nan: None})
    return {"patient_nbr": patient_nbr, "task": task,
            "encounters": merged.to_dict(orient="records")}


@app.get("/api/schema")
def schema():
    from .data import CATEGORICAL_COLUMNS
    from .icd9 import DRUG_COLUMNS, DRUG_STATES
    cohort = _cohort()
    options = {}
    for col in CATEGORICAL_COLUMNS:
        values = cohort[col].dropna().value_counts().head(25).index
        options[col] = [v.item() if hasattr(v, "item") else v for v in values]
    common_dx = (pd.concat([cohort.diag_1, cohort.diag_2, cohort.diag_3])
                 .dropna().value_counts().head(60).index.tolist())
    return {"categorical": options, "drugs": DRUG_COLUMNS,
            "drug_states": DRUG_STATES, "common_diagnoses": common_dx}


class PredictRequest(BaseModel):
    task: str = Field("readmit30", pattern="^(readmit30|escalation)$")
    encounters: list[dict]


@app.post("/api/predict")
def predict(req: PredictRequest):
    from .inference import predict_patient
    if not req.encounters:
        raise HTTPException(400, "no encounters provided")
    if len(req.encounters) > 50:
        raise HTTPException(400, "at most 50 encounters per request")
    try:
        return predict_patient(req.encounters, req.task)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))


# ── static React build ───────────────────────────
WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")),
              name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(str(WEB_DIST / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = (WEB_DIST / full_path).resolve()
        if candidate.is_file() and WEB_DIST.resolve() in candidate.parents:
            return FileResponse(str(candidate))
        return FileResponse(str(WEB_DIST / "index.html"))
