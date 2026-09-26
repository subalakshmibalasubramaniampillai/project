"""Write FINAL_SUMMARY.md from the pipeline outputs (no hand-copied numbers)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
LABELS = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
          "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP", "gru": "GRU",
          "retain": "RETAIN", "transformer": "Transformer", "tkgn": "**TKGN** (new)",
          "tkgn_b": "**TKGN-B** (new)"}


def load(name):
    return json.loads((OUT / name).read_text())


d = load("dataset_summary.json")
summary = load("summary.json")
reps = load("repeat_tests.json")
sig = load("significance.json")
lines = [
    "# Project summary",
    "",
    "## Data (real, public, unmodified)",
    "",
    f"* Source: {d['source']}; CC BY 4.0; SHA-256 verified.",
    f"* Cohort: {d['cohort_encounters']:,} inpatient encounters of {d['patients']:,} patients "
    f"({d['patients_with_repeat_encounters']:,} with repeat stays) after excluding deaths/hospice.",
    f"* Task 1 — 30-day readmission: prevalence {100 * d['readmit30_rate']:.1f}%.",
    f"* Task 2 — treatment escalation at next stay: {d['escalation_samples']:,} samples, "
    f"prevalence {100 * d['escalation_rate']:.1f}%.",
    "* No synthetic, simulated or augmented data are used anywhere.",
    "",
    "## Test results (AUROC; grouped = mean ± SD over 5 patient-disjoint repeats; "
    "temporal = mean over 3 seeds on the most recent 20% of admissions)",
    "",
    "| Model | Readmission grouped | Readmission temporal | Escalation grouped | Escalation temporal |",
    "|---|---|---|---|---|",
]
for m, label in LABELS.items():
    cells = [label]
    for task in ("readmit30", "escalation"):
        for split in ("grouped", "temporal"):
            r = next((x for x in summary if x["task"] == task and x["split"] == split
                      and x["model"] == m), None)
            if r is None:
                cells.append("–")
            elif split == "grouped":
                cells.append(f"{r['auroc']:.3f} ± {r['auroc_sd']:.3f}")
            else:
                cells.append(f"{r['auroc']:.3f}")
    lines.append("| " + " | ".join(cells) + " |")

lines += ["", "## TKGN-B versus gradient boosting (patient-grouped repeats)", ""]
for task in ("readmit30", "escalation"):
    for m in ("lightgbm", "xgboost"):
        r = next(x for x in reps if x["task"] == task and x["split"] == "grouped"
                 and x["reference"] == "tkgn_b" and x["model"] == m)
        b = sig[f"{task}_grouped"]["vs_tkgn_b"][m]["bootstrap"]
        lines.append(f"* {task} vs {LABELS[m]}: mean ΔAUROC {r['mean_delta_auroc']:+.4f}, "
                     f"won {r['wins']}/{r['n_repeats']} repeats (paired t p={r['t_test_p']:.3f}); "
                     f"first test set ΔAUROC {b['delta_auroc']:+.4f} "
                     f"[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")

kg = OUT / "kg_efficiency.json"
if kg.exists():
    lines += ["", "## Knowledge graph under data scarcity (TKGN minus TKGN without KG)", ""]
    for r in json.loads(kg.read_text()):
        lines.append(f"* {r['task']} at {int(100 * r['fraction'])}% of training patients: "
                     f"ΔAUROC {r['delta_auroc']:+.4f} ± {r['delta_sd']:.4f} "
                     f"(KG better in {r['wins']}/{r['n_runs']} repeats)")

lines += [
    "",
    "## Honest reading",
    "",
    "* TKGN-B is the best or joint-best model under patient-disjoint validation for both tasks "
    "and the best model on the temporal escalation split; it is clearly better than GRU, RETAIN, "
    "Transformer and TKGN alone.",
    "* Its advantage over tuned LightGBM/XGBoost is small (≈0.002 AUROC) and not statistically "
    "significant on a single test set; a random forest was most robust on the temporal readmission split.",
    "* Earlier stays are the most valuable input; knowledge-graph code sharing did not improve "
    "discrimination at full data size (see ablation and the data-scarcity follow-up).",
    "* Logistic regression and MLP degrade strongly under temporal shift (more diagnoses recorded "
    "per stay and longer histories in later years).",
    "",
    "## Reproduce",
    "",
    "```bash",
    "python -m pytest -q",
    "python -m src.run_pipeline        # full protocol (~4 h, 4 CPU cores)",
    "python -m src.kg_efficiency       # knowledge-graph data-scarcity follow-up",
    "python -m src.explain",
    "python paper/make_tables.py && python paper/make_figures.py && python paper/make_summary.py",
    "cd paper && latexmk -pdf main.tex",
    "```",
    "",
    "Before submission: run a similarity check (e.g. iThenticate/Turnitin), verify every "
    "reference, and complete author, funding and CRediT information in `paper/main.tex`.",
]
(ROOT / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("FINAL_SUMMARY.md written")
