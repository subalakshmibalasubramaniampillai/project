"""
FastAPI backend for the diabetes progression research system.

Provides REST endpoints that the React frontend consumes:
  - /health
  - /api/metrics            → all model evaluation metrics
  - /api/ablation           → ablation-stage metrics
  - /api/split              → dataset + split summary
  - /api/dataset            → patient-level raw data
  - /api/dataset/{patient_id} → one patient's visits
  - /api/patients           → the list of patient IDs
  - /api/explanation        → saved feature-contribution probe
  - /api/clusters           → trajectory clustering summary
  - /api/fusion             → CCF model weights
  - /api/graph              → knowledge graph nodes/edges
  - /api/predict            → single / multi-visit inference
  - /api/ingest             → upload CSV / use real / generate + retrain

Static build of the React app is served from ../web/dist when present.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data_source import prepare_dataset
from .inference import infer_new_patient
from .models import make_patient_features, patient_split
from .novel import PatientTrajectoryClusterer

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "longitudinal_diabetes.csv"

app = FastAPI(
    title="Diabetes Progression Research API",
    description="Longitudinal diabetes progression research and "
                "decision-support system. Not a diagnostic tool.",
    version="2.0.0",
)

# CORS for the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_data():
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Run `python -m src.run_pipeline` or ingest data first.",
        )
    return pd.read_csv(DATA_PATH)


def _safe_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    return json.loads(path.read_text(encoding="utf-8"))


# ── health ───────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "purpose": "research and decision support, not diagnosis",
    }


# ── metrics ──────────────────────────────────────
@app.get("/api/metrics")
def get_metrics():
    return _safe_json(ROOT / "outputs" / "model_metrics.json")


@app.get("/api/ablation")
def get_ablation():
    return _safe_json(ROOT / "outputs" / "ablation_metrics.json")


@app.get("/api/split")
def get_split():
    return _safe_json(ROOT / "outputs" / "split.json")


@app.get("/api/fusion")
def get_fusion():
    return _safe_json(ROOT / "outputs" / "ccf_weights.json")


@app.get("/api/explanation")
def get_explanation():
    return _safe_json(ROOT / "outputs" / "explanation.json")


# ── dataset ──────────────────────────────────────
@app.get("/api/patients")
def list_patients():
    data = _require_data()
    return {"patients": sorted(data.patient_id.astype(str).unique().tolist())}


@app.get("/api/dataset")
def get_dataset(patient_id: Optional[str] = None):
    data = _require_data()
    if patient_id:
        frame = data[data.patient_id == patient_id]
        if frame.empty:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {"patient_id": patient_id,
                "visits": frame.to_dict(orient="records")}
    return data.to_dict(orient="records")


@app.get("/api/clusters")
def get_clusters():
    feat_path = ROOT / "outputs" / "patient_features.csv"
    if not feat_path.exists():
        raise HTTPException(status_code=404, detail="No patient features")
    feats = pd.read_csv(feat_path)
    if "traj_cluster" not in feats.columns:
        return {"clusters": None}
    summary = (
        feats.groupby("traj_cluster")
        .agg({"hba1c_slope": "mean", "target": "mean",
              "patient_id": "count"})
        .rename(columns={"patient_id": "n_patients"})
        .reset_index()
    )
    return {
        "clusters": summary.to_dict(orient="records"),
        "counts": feats["traj_cluster"]
        .value_counts().sort_index().to_dict(),
    }


# ── knowledge graph ──────────────────────────────
@app.get("/api/graph")
def get_graph():
    gml_path = ROOT / "outputs" / "knowledge_graph.graphml"
    if not gml_path.exists():
        raise HTTPException(status_code=404, detail="No graph file")
    import networkx as nx
    graph = nx.read_graphml(str(gml_path))
    edges = []
    for src, tgt, attr in graph.edges(data=True):
        edges.append({
            "source": src, "target": tgt,
            "relation": attr.get("relation", ""),
            "justification": attr.get("justification", ""),
        })
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes": list(graph.nodes),
        "edges": edges[:5000],  # keep payload bounded for the UI
    }


# ── inference ────────────────────────────────────
class PredictionRequest(BaseModel):
    visits: list
    use_tagnn: bool = True


@app.post("/api/predict")
def predict(req: PredictionRequest):
    if not req.visits:
        raise HTTPException(status_code=400, detail="No visits provided")
    frame = pd.DataFrame(req.visits)
    use_tagnn = req.use_tagnn and (ROOT / "outputs" / "tagnn.pt").exists()
    model_path = (
        "outputs/tagnn.pt"
        if use_tagnn
        else "outputs/gnn_kg_longitudinal.pt"
    )
    result = infer_new_patient(
        frame, str(ROOT / model_path), use_tagnn=use_tagnn,
    )
    return result


# ── ingest / retrain ─────────────────────────────
class IngestRequest(BaseModel):
    use_real: bool = False
    source: Optional[str] = None


@app.post("/api/ingest")
def ingest(req: IngestRequest = None, file: UploadFile = File(default=None)):
    from .run_pipeline import run as run_pipeline

    try:
        if file is not None:
            content = pd.read_csv(file.file)
            run_pipeline(data=content)
        elif req is not None and req.source:
            import io
            run_pipeline(source=req.source)
        elif req is not None and req.use_real:
            run_pipeline(use_real_data=True)
        else:
            run_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "detail": "Re-trained successfully"}


# ── static React build (production) ──────────────
WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")),
              name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(str(WEB_DIST / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = WEB_DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(WEB_DIST / "index.html"))
