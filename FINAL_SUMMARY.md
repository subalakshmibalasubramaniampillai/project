# Project report

## Goal

Predict, from the first three visits of a patient's record, whether diabetes
progression will be observed by the fourth visit. The target is defined as a
final HbA1c that is at least 0.5 points higher than the first-visit value, or
a final HbA1c of at least 6.5%.

## Data

- **Pima Indians Diabetes Database** (UCI / NIDDK). 768 patients, 8 medical
  predictor variables, binary outcome. Downloaded and cached in `data/`.
- Synthetic longitudinal cohort (240 patients, 4 visits each) used when no
  real data is provided, for method development.
- Because Pima is a cross-sectional table, the records were expanded into a
  4-visit longitudinal form with modest synthetic drift between visits.

## Approach

Feature engineering aggregates the early trajectory (mean, std, linear slope,
curvature of HbA1c/glucose) into a per-patient row. A patient-level stratified
75/25 split keeps one patient's visits from crossing train and test.

Ten models are compared:

1. Four tabular baselines: logistic regression, random forest, XGBoost, MLP.
2. Three message-passing graph models on a knowledge graph built from shared
   conditions (plain GNN, KG-augmented GNN, KG + longitudinal features).
3. Three added components, each implemented from scratch:
   - **TAGNN** — graph layers with a learnable temporal gate that adjusts
     neighbor message weights by trajectory divergence.
   - **MHFIN** — learned pairwise feature-interaction projections concatenated
     with the base features before classification.
   - **CCF** — logistic calibration over stacked base-model probabilities,
     fit with inner cross-validation.

Patient trajectory clustering (KMeans on HbA1c slope + curvature) is used as a
preprocessing step; the cluster id is added to the feature set for all models.

## Results

Exact numbers are written to `outputs/model_metrics.json` after every run.
On the Pima run the novel TAGNN model reached the highest F1 of the set, ahead
of the baselines and the standard graph variants. MHFIN and CCF were close to
the best tabular models on ROC-AUC.

## Running

```powershell
# train (synthetic by default, --real for Pima)
python -m src.run_pipeline [--real]

# serve the web app (React frontend, built in web/dist)
uvicorn src.api:app --reload     # http://127.0.0.1:8000
```

See `README.md` for the full setup and the list of REST endpoints.

## Limitations

- The Pima conversion adds synthetic visits, so results are not a clinical
  evaluation of the method.
- Graph similarity edges encode the assumption that shared-condition patients
  are related; this holds only as a modeling prior.
- Calibration and confidence intervals were not computed.