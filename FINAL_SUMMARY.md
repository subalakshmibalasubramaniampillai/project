# Project summary

## Data (real, public, unmodified)

* Source: Diabetes 130-US Hospitals 1999-2008 (UCI id 296; Strack et al., 2014); CC BY 4.0; SHA-256 verified.
* Cohort: 99,340 inpatient encounters of 69,987 patients (16,341 with repeat stays) after excluding deaths/hospice.
* Task 1 — 30-day readmission: prevalence 11.4%.
* Task 2 — treatment escalation at next stay: 29,353 samples, prevalence 35.0%.
* No synthetic, simulated or augmented data are used anywhere.

## Test results (AUROC; grouped = mean ± SD over 5 patient-disjoint repeats; temporal = mean over 3 seeds on the most recent 20% of admissions)

| Model | Readmission grouped | Readmission temporal | Escalation grouped | Escalation temporal |
|---|---|---|---|---|
| Logistic regression | 0.669 ± 0.005 | 0.603 | 0.666 ± 0.013 | 0.646 |
| Random forest | 0.682 ± 0.005 | 0.700 | 0.685 ± 0.012 | 0.671 |
| XGBoost | 0.688 ± 0.004 | 0.684 | 0.687 ± 0.013 | 0.673 |
| LightGBM | 0.688 ± 0.004 | 0.685 | 0.687 ± 0.012 | 0.673 |
| MLP | 0.653 ± 0.009 | 0.601 | 0.646 ± 0.007 | 0.622 |
| GRU | 0.682 ± 0.005 | 0.684 | 0.681 ± 0.014 | 0.670 |
| RETAIN | 0.683 ± 0.006 | 0.687 | 0.677 ± 0.013 | 0.668 |
| Transformer | 0.683 ± 0.006 | 0.690 | 0.679 ± 0.013 | 0.668 |
| **TKGN** (new) | 0.682 ± 0.004 | 0.689 | 0.678 ± 0.012 | 0.668 |
| **TKGN-B** (new) | 0.690 ± 0.005 | 0.685 | 0.688 ± 0.013 | 0.680 |

## TKGN-B versus gradient boosting (patient-grouped repeats)

* readmit30 vs LightGBM: mean ΔAUROC +0.0020, won 4/5 repeats (paired t p=0.123); first test set ΔAUROC +0.0002 [-0.0037, +0.0038]
* readmit30 vs XGBoost: mean ΔAUROC +0.0018, won 4/5 repeats (paired t p=0.264); first test set ΔAUROC +0.0017 [-0.0028, +0.0063]
* escalation vs LightGBM: mean ΔAUROC +0.0015, won 5/5 repeats (paired t p=0.088); first test set ΔAUROC +0.0003 [-0.0004, +0.0011]
* escalation vs XGBoost: mean ΔAUROC +0.0016, won 5/5 repeats (paired t p=0.093); first test set ΔAUROC +0.0009 [-0.0025, +0.0045]

## Honest reading

* TKGN-B is the best or joint-best model under patient-disjoint validation for both tasks and the best model on the temporal escalation split; it is clearly better than GRU, RETAIN, Transformer and TKGN alone.
* Its advantage over tuned LightGBM/XGBoost is small (≈0.002 AUROC) and not statistically significant on a single test set; a random forest was most robust on the temporal readmission split.
* Earlier stays are the most valuable input; knowledge-graph code sharing did not improve discrimination at full data size (see ablation and the data-scarcity follow-up).
* Logistic regression and MLP degrade strongly under temporal shift (more diagnoses recorded per stay and longer histories in later years).

## Reproduce

```bash
python -m pytest -q
python -m src.run_pipeline        # full protocol (~4 h, 4 CPU cores)
python -m src.kg_efficiency       # knowledge-graph data-scarcity follow-up
python -m src.explain
python paper/make_tables.py && python paper/make_figures.py && python paper/make_summary.py
cd paper && latexmk -pdf main.tex
```

Before submission: run a similarity check (e.g. iThenticate/Turnitin), verify every reference, and complete author, funding and CRediT information in `paper/main.tex`.
