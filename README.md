# Diabetes Progression Research System

A longitudinal research and decision-support system for diabetes progression.
Not a diagnostic tool.

Built with **React + FastAPI** (no Streamlit).

## Features

- Loads real public datasets (Pima Indians) or generates synthetic cohorts
- Knowledge graph construction with observed and assumed edges
- 10 evaluated models including 4 novel techniques (TAGNN, MHFIN, CCF, PTC)
- React web dashboard for exploring models, patients, and predictions
- REST API for data, metrics, inference, and retraining

## Architecture

```
┌──────────────────┐     HTTP      ┌──────────────────────┐
│  React frontend  │  ─────────►   │  FastAPI backend     │
│  web/ (Vite)     │  /api/*       │  src/api.py          │
└──────────────────┘               │  + pipeline, models  │
                                   └──────────────────────┘
```

- Dev mode: Vite dev server on port 5173, proxies `/api` to FastAPI on 8000.
- Prod mode: FastAPI serves the built React app from `web/dist`.

## Quick start

### 1. Train the models

```powershell
pip install -r requirements.txt

# train on synthetic data
python -m src.run_pipeline

# train on the real Pima dataset
python -m src.run_pipeline --real
```

### 2. Start the backend

```powershell
uvicorn src.api:app --reload
# http://127.0.0.1:8000  → served React app (after build)
# http://127.0.0.1:8000/docs → API docs
```

### 3. Build the frontend (first time only)

```powershell
cd web
npm install
npm run build
```

FastAPI now serves the built React app at `http://127.0.0.1:8000`.

### 4. Development mode (optional)

```powershell
cd web
npm run dev
# http://localhost:5173  with hot reload; proxies /api to port 8000
```

If you change React code, run `npm run build` again to refresh the
production bundle served by FastAPI.

## Custom data

Provide a CSV with these columns: `patient_id`, `visit`, `date`, `age`,
`hba1c`, `glucose`, `bmi`, `systolic_bp`, `diastolic_bp`, `conditions`.

```powershell
python -m src.run_pipeline --input path/to/data.csv
```

Or set the `DATA_SOURCE` environment variable to a URL. Alternatively use
`POST /api/ingest` with `{"source": "..."}` or `{"use_real": true}`.

## Updating the data without retraining

The "new patient" form runs inference only. To retrain on new data, run the
pipeline again or call `POST /api/ingest`.

## Model evaluation

The pipeline trains and evaluates these models:

| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Baseline |
| XGBoost | Baseline |
| MLP | Baseline |
| GNN | Graph |
| GNN + KG | Graph + knowledge |
| GNN + KG + longitudinal | Graph + knowledge + temporal |
| TAGNN | Novel (temporal attention) |
| MHFIN | Novel (feature interactions) |
| CCF Fusion | Novel (calibrated stacking) |

Metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score,
log loss) are written to `outputs/model_metrics.json`.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/metrics` | All model metrics |
| GET | `/api/ablation` | Ablation-stage metrics |
| GET | `/api/split` | Dataset + split summary |
| GET | `/api/dataset` | All visits, or `?patient_id=` for one patient |
| GET | `/api/patients` | List of patient IDs |
| GET | `/api/explanation` | Saved feature-contribution probe |
| GET | `/api/clusters` | Trajectory clustering summary |
| GET | `/api/fusion` | CCF model weights |
| GET | `/api/graph` | Knowledge graph nodes/edges |
| POST | `/api/predict` | Single-patient inference |
| POST | `/api/ingest` | Retrain from CSV/URL/real dataset |

## Project structure

```
src/
  run_pipeline.py      — orchestrates the full pipeline
  models.py            — baseline + graph model definitions
  novel.py             — TAGNN, MHFIN, CCF, PTC
  datasets.py          — real dataset loaders
  data_source.py       — validation and loading
  knowledge_graph.py   — graph construction
  inference.py         — single-patient inference
  metrics.py           — evaluation utilities
  generate_data.py     — synthetic cohort fallback
  api.py               — FastAPI REST backend + static web serving
web/
  src/                 — React source (pages, styles, API client)
  dist/                — built bundle served by FastAPI
data/                  — datasets
outputs/               — metrics, graphs, model artifacts
```