# TKGN: temporal knowledge-gated risk models on real diabetes inpatient records

A research and decision-support system that predicts, at hospital discharge,

1. **30-day readmission**, and
2. **escalation of glucose-lowering treatment at the patient's next stay**

from **real, de-identified electronic health records**: the
[Diabetes 130-US Hospitals 1999–2008](https://archive.ics.uci.edu/dataset/296)
database (Strack et al., *BioMed Research International* 2014; UCI dataset 296,
CC BY 4.0). 101,766 inpatient encounters from 130 US hospitals; a persistent
patient number links repeat stays, which makes the data genuinely
longitudinal. **No synthetic, simulated or augmented records are used
anywhere.**

Not a diagnostic tool.

## What is new

| Component | Idea |
|---|---|
| **Hybrid clinical knowledge graph** | ICD-9-CM hierarchy (code → subcategory → category → chapter) + drug → pharmacological class + a co-morbidity graph (PPMI between diagnosis categories) estimated on training encounters only |
| **Knowledge-grounded code encoder** | each code keeps its own embedding and adds an attention-weighted mix of its ancestors; category embeddings are refined by zero-initialised co-morbidity propagation, so rare codes borrow strength while frequent codes stay specific |
| **TKGN — Temporal Knowledge-Gated Network** | GRU + recency attention over earlier stays, an explicit change-since-last-stay (delta) signal, and **history-reliability gates** conditioned on the amount of history, plus an auxiliary 3-level outcome head |
| **TKGN-B** | TKGN trained as a residual on top of an **out-of-fold LightGBM logit**, i.e. one extra "boosting stage" implemented by a knowledge-aware sequence model |
| **Leakage-controlled protocol** | patient-disjoint repeated splits (5×), prospective temporal split, validation-only tuning and thresholds, clustered-bootstrap CIs, DeLong tests, calibration, ablations, learning curves, subgroup/fairness audit |

## Results at a glance

Test AUROC, patient-disjoint splits (mean ± SD over 5 repeats):

| | 30-day readmission | Treatment escalation |
|---|---|---|
| Logistic regression | 0.669 ± 0.005 | 0.666 ± 0.013 |
| LightGBM | 0.688 ± 0.004 | 0.687 ± 0.012 |
| XGBoost | 0.688 ± 0.004 | 0.687 ± 0.013 |
| GRU / RETAIN / Transformer | 0.682–0.683 | 0.677–0.681 |
| TKGN | 0.682 ± 0.004 | 0.678 ± 0.012 |
| **TKGN-B** | **0.690 ± 0.005** | **0.688 ± 0.013** |

* TKGN-B is best on both tasks and beats every deep sequence baseline in
  every repeat; its edge over tuned gradient boosting is small (≈0.002
  AUROC, not statistically significant on a single test set).
* On the prospective temporal split TKGN-B is best for escalation
  (0.680 vs 0.673) but a random forest is best for readmission (0.700).
* Earlier stays are the most useful input. The knowledge graph did **not**
  improve discrimination at any training-set size tested — reported as a
  negative result.

All numbers are produced by `python -m src.run_pipeline` and written to
`outputs/`. See `outputs/summary.json` (mean ± SD over runs),
`outputs/significance.json` (CIs and tests) and the manuscript in
`paper/`. A short digest is kept in [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md).

## Quick start

```bash
pip install -r requirements.txt

# 1. verify data integrity and leakage guards (13 tests)
python -m pytest -q

# 2. run the complete experimental protocol (≈4–5 h on 4 CPU cores)
python -m src.run_pipeline
#    or a quick smoke run (1 repeat, no ablations / temporal / learning curve)
python -m src.run_pipeline --quick

# 3. knowledge-graph data-scarcity follow-up and explanations
python -m src.kg_efficiency
python -m src.explain

# 4. manuscript tables, macros and figures from the outputs
python paper/make_tables.py
python paper/make_figures.py
python paper/make_summary.py
cd paper && latexmk -pdf main.tex
```

The raw file ships in `data/raw/diabetic_data.csv` and is verified by SHA-256
before use; if it is missing it is downloaded from UCI (or an identical
mirror) and verified again.

## Web application (React + FastAPI)

```bash
cd web && npm install && npm run build && cd ..
uvicorn src.api:app --reload        # http://127.0.0.1:8000
```

Pages: results, statistics, ablation, data efficiency, calibration,
fairness, explainability, knowledge graph, patient explorer (real held-out
test patients) and a risk calculator that scores a posted encounter
history with TKGN-B.

For development: `cd web && npm run dev` (http://localhost:5173, proxies
`/api` to port 8000).

### REST endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/overview` | dataset summary and split sizes |
| GET | `/api/summary` | test metrics (mean ± SD) per task / split / model |
| GET | `/api/significance` | bootstrap CIs, DeLong and paired bootstrap tests |
| GET | `/api/repeat-tests` | paired comparison across repeats |
| GET | `/api/ablation` | TKGN component ablation |
| GET | `/api/kg-efficiency` | knowledge graph under data scarcity |
| GET | `/api/learning-curve` | AUROC vs training-set size |
| GET | `/api/calibration` | reliability curves |
| GET | `/api/subgroups` | subgroup / fairness metrics |
| GET | `/api/gates` | history gates vs number of earlier stays |
| GET | `/api/explanations` | permutation importance, TreeSHAP, examples |
| GET | `/api/graph` | knowledge-graph summary |
| GET | `/api/patients`, `/api/patients/{id}` | held-out patients and their predictions |
| GET | `/api/schema` | allowed values for the risk form |
| POST | `/api/predict` | `{"task": "readmit30", "encounters": [...]}` → TKGN-B risk |

## Project structure

```
src/
  data.py            real-data loader (checksum), cohort, labels, splits, history index
  icd9.py            ICD-9-CM chapters/hierarchy, drug classes
  knowledge_graph.py hybrid ontology + co-morbidity graph (train-only)
  features.py        sequence arrays and tabular design matrix
  neural.py          TKGN, GRU, RETAIN, Transformer; batching and training
  baselines.py       LR, RF, XGBoost, LightGBM, MLP with validation tuning
  evaluation.py      metrics, calibration, clustered bootstrap, DeLong
  run_pipeline.py    experiment runner (resumable jobs)
  analysis.py        aggregation into outputs/*.json
  explain.py         permutation importance, TreeSHAP, attention examples
  kg_efficiency.py   knowledge graph vs flat codes with scarce training data
  inference.py       single-patient TKGN-B inference
  api.py             FastAPI backend + static web app
tests/               data-integrity and leakage tests
web/                 React dashboard (Vite)
paper/               journal manuscript (LaTeX), figure/table generators
data/raw/            the unmodified UCI release
outputs/             results (JSON/CSV), deployable models, logs
```

## Limitations

* One public data source; external validation on an independent hospital
  system is still needed.
* The release has no calendar dates; stays are ordered by encounter
  identifier and time gaps between stays are unknown.
* Only three diagnosis codes per stay are recorded, and most diagnosis
  codes are truncated to three characters (diabetes codes excepted).
* Readmission is recorded only for encounters within the participating
  systems.
