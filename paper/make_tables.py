"""
Generate LaTeX tables and number macros for the manuscript directly from
the pipeline outputs, so that no result is transcribed by hand.

    python paper/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TAB = Path(__file__).resolve().parent / "tables"
TAB.mkdir(exist_ok=True)

LABELS = {
    "logistic_regression": "Logistic regression", "random_forest": "Random forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP", "gru": "GRU",
    "retain": "RETAIN", "transformer": "Transformer",
    "tkgn": "TKGN (proposed)", "tkgn_b": "TKGN-B (proposed)",
}
MAIN = list(LABELS)
MACRO_MODEL = {
    "logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB",
    "lightgbm": "LGBM", "mlp": "MLP", "gru": "GRU", "retain": "RETAIN",
    "transformer": "TRF", "tkgn": "TKGN", "tkgn_b": "TKGNB",
}
MACRO_TASK = {"readmit30": "Readmit", "escalation": "Esc"}
MACRO_SPLIT = {"grouped": "Grp", "temporal": "Tmp"}
TASK_NAMES = {"readmit30": "30-day readmission",
              "escalation": "treatment escalation at the next stay"}


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def f3(x):
    return f"{x:.3f}"


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


macros: list[str] = []


def macro(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


# ──────────────────────────────────────────────
def cohort_table():
    d = load("dataset_summary.json")
    n = d["cohort_encounters"]

    def share(counts, keys):
        return sum(counts.get(k, 0) for k in keys) / n * 100

    ages = d["age"]
    young = [a for a in ages if int(a.strip("[)").split("-")[0]) < 50]
    mid = [a for a in ages if 50 <= int(a.strip("[)").split("-")[0]) < 70]
    old = [a for a in ages if int(a.strip("[)").split("-")[0]) >= 70]
    # "None" in the release means the test was not performed
    a1c_measured = 100 - (d["a1c"].get("None", 0) + d["a1c"].get("Unknown", 0)) / n * 100
    rows = [
        ("Inpatient encounters", f"{n:,}"),
        ("Unique patients", f"{d['patients']:,}"),
        ("Patients with $\\geq$2 encounters", f"{d['patients_with_repeat_encounters']:,}"),
        ("Encounters with $\\geq$1 earlier stay", f"{d['encounters_with_history']:,}"),
        ("Maximum encounters per patient", f"{d['max_encounters_per_patient']}"),
        ("Female, \\%", f"{share(d['gender'], ['Female']):.1f}"),
        ("Age $<$50 / 50--69 / $\\geq$70 years, \\%",
         f"{share(ages, young):.1f} / {share(ages, mid):.1f} / {share(ages, old):.1f}"),
        ("Race: Caucasian / African American / other or unknown, \\%",
         f"{share(d['race'], ['Caucasian']):.1f} / {share(d['race'], ['AfricanAmerican']):.1f} / "
         f"{100 - share(d['race'], ['Caucasian', 'AfricanAmerican']):.1f}"),
        ("HbA1c measured during stay, \\%", f"{a1c_measured:.1f}"),
        ("Insulin prescribed, \\%", f"{100 * d['insulin_any']:.1f}"),
        ("Length of stay, mean days", f"{d['time_in_hospital_mean']:.2f}"),
        ("Medications administered, mean", f"{d['num_medications_mean']:.1f}"),
        ("Readmitted within 30 days, \\%", f"{100 * d['readmit30_rate']:.1f}"),
        ("Encounters with a later stay (escalation task)", f"{d['escalation_samples']:,}"),
        ("\\quad of which escalated at next stay, \\%", f"{100 * d['escalation_rate']:.1f}"),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    (TAB / "tab_cohort.tex").write_text(
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\nCharacteristic & Value \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n", encoding="utf-8")
    macro("NEnc", f"{n:,}")
    macro("NPat", f"{d['patients']:,}")
    macro("NRaw", f"{d['raw_encounters']:,}")
    macro("NRepeat", f"{d['patients_with_repeat_encounters']:,}")
    macro("NHist", f"{d['encounters_with_history']:,}")
    macro("NEsc", f"{d['escalation_samples']:,}")
    macro("RateReadmit", f"{100 * d['readmit30_rate']:.1f}")
    macro("RateEsc", f"{100 * d['escalation_rate']:.1f}")
    macro("PctHist", f"{100 * d['encounters_with_history'] / n:.1f}")


def results_tables():
    rows = load("summary.json")
    for task in ["readmit30", "escalation"]:
        grp = {r["model"]: r for r in rows if r["task"] == task and r["split"] == "grouped"}
        tmp = {r["model"]: r for r in rows if r["task"] == task and r["split"] == "temporal"}
        best_g = max((grp[m]["auroc"] for m in MAIN if m in grp), default=0)
        best_t = max((tmp[m]["auroc"] for m in MAIN if m in tmp), default=0)
        lines = []
        for m in MAIN:
            if m not in grp:
                continue
            g = grp[m]

            def cell(key, digits=3, bold=False):
                txt = f"{g[key]:.{digits}f}$\\pm${g[key + '_sd']:.{digits}f}"
                return f"\\textbf{{{txt}}}" if bold else txt

            auc = cell("auroc", bold=abs(g["auroc"] - best_g) < 1e-12)
            t = tmp.get(m)
            t_auc = (f"\\textbf{{{f3(t['auroc'])}}}" if t and abs(t["auroc"] - best_t) < 1e-12
                     else (f3(t["auroc"]) if t else "--"))
            lines.append(
                f"{LABELS[m]} & {auc} & {cell('auprc')} & {cell('brier')} & "
                f"{cell('ece')} & {g['calibration_slope']:.2f} & {g['sensitivity']:.2f} & "
                f"{g['specificity']:.2f} & {t_auc} & {f3(t['auprc']) if t else '--'} \\\\")
            if m == "mlp" or m == "transformer":
                lines.append("\\midrule")
            for split, src in (("grouped", g), ("temporal", t)):
                if not src:
                    continue
                key = f"{MACRO_TASK[task]}{MACRO_SPLIT[split]}{MACRO_MODEL[m]}"
                macro(f"Auc{key}", f3(src["auroc"]))
                macro(f"Prc{key}", f3(src["auprc"]))
                macro(f"Bri{key}", f"{src['brier']:.4f}")
                macro(f"Ece{key}", f3(src["ece"]))
                macro(f"Slope{key}", f"{src['calibration_slope']:.2f}")
                macro(f"Sens{key}", f"{src['sensitivity']:.2f}")
                macro(f"Spec{key}", f"{src['specificity']:.2f}")
                if split == "grouped":
                    macro(f"AucSd{key}", f3(src["auroc_sd"]))
        header = ("\\begin{tabular}{@{}lccccccc|cc@{}}\n\\toprule\n"
                  " & \\multicolumn{7}{c|}{Patient-grouped split (5 repeats, mean$\\pm$SD)} & "
                  "\\multicolumn{2}{c}{Temporal split} \\\\\n"
                  "Model & AUROC & AUPRC & Brier & ECE & Slope & Sens. & Spec. & AUROC & AUPRC \\\\\n"
                  "\\midrule\n")
        (TAB / f"tab_results_{task}.tex").write_text(
            header + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def significance_table():
    sig = load("significance.json")
    rep = load("repeat_tests.json")
    lines = []
    for m in MAIN:
        if m == "tkgn_b":
            continue
        cells = [LABELS[m]]
        for task in ["readmit30", "escalation"]:
            entry = sig.get(f"{task}_grouped", {}).get("vs_tkgn_b", {}).get(m)
            r = next((x for x in rep if x["task"] == task and x["split"] == "grouped"
                      and x["reference"] == "tkgn_b" and x["model"] == m), None)
            tentry = sig.get(f"{task}_temporal", {}).get("vs_tkgn_b", {}).get(m)
            if entry is None:
                cells += ["--"] * 4
                continue
            b = entry["bootstrap"]
            cells += [f"{b['delta_auroc']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]",
                      pv(entry["delong"]["p_value"]),
                      f"{r['wins']}/{r['n_repeats']}" if r else "--",
                      pv(tentry["delong"]["p_value"]) if tentry else "--"]
            key = f"{MACRO_TASK[task]}{MACRO_MODEL[m]}"
            macro(f"Dauc{key}", f"{b['delta_auroc']:.4f}")
            macro(f"DaucLo{key}", f"{b['ci'][0]:.4f}")
            macro(f"DaucHi{key}", f"{b['ci'][1]:.4f}")
            macro(f"PDelong{key}", pv(entry["delong"]["p_value"]))
            macro(f"PBoot{key}", pv(b["p_value"]))
            if r:
                macro(f"Wins{key}", f"{r['wins']}/{r['n_repeats']}")
                macro(f"RepDauc{key}", f"{r['mean_delta_auroc']:+.4f}")
                macro(f"RepP{key}", pv(r["t_test_p"]))
            if tentry:
                macro(f"DaucTmp{key}", f"{tentry['bootstrap']['delta_auroc']:.4f}")
                macro(f"PDelongTmp{key}", pv(tentry["delong"]["p_value"]))
        lines.append(" & ".join(cells) + " \\\\")
    header = ("\\begin{tabular}{@{}l cccc cccc@{}}\n\\toprule\n"
              " & \\multicolumn{4}{c}{30-day readmission} & \\multicolumn{4}{c}{Treatment escalation} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
              "Comparator & $\\Delta$AUROC [95\\% CI] & $p_\\text{DeLong}$ & Wins & $p_\\text{temp}$ "
              "& $\\Delta$AUROC [95\\% CI] & $p_\\text{DeLong}$ & Wins & $p_\\text{temp}$ \\\\\n\\midrule\n")
    (TAB / "tab_significance.tex").write_text(
        header + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def ablation_table():
    rows = load("ablation.json")
    labels = {
        "tkgn": "Full TKGN",
        "tkgn_no_kg": "$-$ knowledge graph (flat code embeddings)",
        "tkgn_no_cooccurrence": "$-$ co-morbidity (PPMI) propagation",
        "tkgn_no_history": "$-$ earlier encounters (index stay only)",
        "tkgn_no_gate": "$-$ history-reliability gates",
        "tkgn_no_delta": "$-$ delta encoding",
        "tkgn_no_prior_outcomes": "$-$ outcomes of earlier stays",
        "tkgn_no_aux": "$-$ auxiliary outcome head",
    }
    lines = []
    for m, label in labels.items():
        cells = [label]
        for task in ["readmit30", "escalation"]:
            r = next((x for x in rows if x["task"] == task and x["model"] == m), None)
            if r is None:
                cells += ["--", "--"]
                continue
            cells.append(f"{r['auroc']:.4f}$\\pm${r['auroc_sd']:.4f}")
            cells.append("--" if m == "tkgn" else f"{r['delta_auroc_vs_full']:+.4f}")
            macro(f"Abl{MACRO_TASK[task]}{m.replace('tkgn', 'T').replace('_', '')}",
                  f"{r['delta_auroc_vs_full']:+.4f}")
        lines.append(" & ".join(cells) + " \\\\")
    n_runs = rows[0]["n_runs"] if rows else 0
    macro("NAblRuns", str(n_runs))
    (TAB / "tab_ablation.tex").write_text(
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        " & \\multicolumn{2}{c}{30-day readmission} & \\multicolumn{2}{c}{Treatment escalation} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Variant & AUROC & $\\Delta$ & AUROC & $\\Delta$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def subgroup_table():
    rows = load("subgroups.json")
    names = {"race": "Race", "gender": "Gender", "age": "Age (years)",
             "prior_encounters": "Earlier stays", "hba1c_measured": "HbA1c"}
    lines = []
    for attr, title in names.items():
        groups = sorted({r["group"] for r in rows if r["attribute"] == attr})
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for g in groups:
            cells = [f"\\quad {g}"]
            for task in ["readmit30", "escalation"]:
                def get(m):
                    return next((r for r in rows if r["task"] == task and r["attribute"] == attr
                                 and r["group"] == g and r["model"] == m), None)
                t, l = get("tkgn_b"), get("lightgbm")
                cells += [f"{t['n']:,}" if t else "--",
                          f3(t["auroc"]) if t else "--", f3(l["auroc"]) if l else "--"]
                if t and attr == "race" and g in ("AfricanAmerican", "Caucasian"):
                    macro(f"SubRace{MACRO_TASK[task]}{'AA' if g == 'AfricanAmerican' else 'CA'}",
                          f3(t["auroc"]))
                if t and attr == "prior_encounters":
                    tag = {"0": "Zero", "1": "One", "2+": "Two"}[g]
                    macro(f"Sub{MACRO_TASK[task]}{tag}TKGNB", f3(t["auroc"]))
                    if l:
                        macro(f"Sub{MACRO_TASK[task]}{tag}LGBM", f3(l["auroc"]))
                    tk = get("tkgn")
                    if tk:
                        macro(f"Sub{MACRO_TASK[task]}{tag}TKGN", f3(tk["auroc"]))
            lines.append(" & ".join(cells) + " \\\\")
    (TAB / "tab_subgroups.tex").write_text(
        "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        " & \\multicolumn{3}{c}{30-day readmission} & \\multicolumn{3}{c}{Treatment escalation} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Subgroup & $n$ & TKGN-B & LightGBM & $n$ & TKGN-B & LightGBM \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")

    big = [r["ece"] for r in rows if r["model"] == "tkgn_b" and r["n"] >= 300]
    macro("MaxSubEce", f"{max(big):.3f}")
    # fairness gaps
    for task in ["readmit30", "escalation"]:
        for attr in ["race", "gender", "age"]:
            # groups with fewer than 300 test encounters give unstable AUROCs
            vals = [r["auroc"] for r in rows if r["task"] == task and r["attribute"] == attr
                    and r["model"] == "tkgn_b" and r["n"] >= 300]
            if len(vals) > 1:
                macro(f"Gap{MACRO_TASK[task]}{attr.capitalize()}", f3(max(vals) - min(vals)))


def learning_macros():
    rows = load("learning_curve.json")
    for r in rows:
        frac = {0.05: "Five", 0.1: "Ten", 0.25: "TwentyFive", 0.5: "Fifty", 1.0: "Full"}.get(
            r["fraction"])
        if frac is None:
            continue
        key = f"{MACRO_TASK[r['task']]}{frac}{MACRO_MODEL[r['model']]}"
        macro(f"Lc{key}", f3(r["auroc"]))
        macro(f"LcHist{key}", f3(r["auroc_with_history"]))
    lines = []
    fractions = sorted({r["fraction"] for r in rows})
    for task in ["readmit30", "escalation"]:
        for m in ["lightgbm", "gru", "tkgn", "tkgn_b"]:
            cells = [f"{TASK_NAMES[task].split(' at')[0].capitalize()}" if m == "lightgbm" else "",
                     LABELS[m].replace(" (proposed)", "")]
            for f in fractions:
                r = next((x for x in rows if x["task"] == task and x["model"] == m
                          and x["fraction"] == f), None)
                cells.append(f3(r["auroc"]) if r else "--")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    cols = " & ".join(f"{int(round(100 * f))}\\%" for f in fractions)
    (TAB / "tab_learning.tex").write_text(
        f"\\begin{{tabular}}{{@{{}}ll{'c' * len(fractions)}@{{}}}}\n\\toprule\n"
        f"Task & Model & {cols} \\\\\n\\midrule\n" + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def gate_macros():
    rows = load("gates.json")
    for r in rows:
        tag = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
               "5+": "Five"}[r["prior_encounters"]]
        macro(f"Gate{MACRO_TASK[r['task']]}{tag}", f"{r['gate_history']:.3f}")
        macro(f"GateDelta{MACRO_TASK[r['task']]}{tag}", f"{r['gate_delta']:.3f}")


def explanation_macros():
    ex = load("explanations.json")
    for task, e in ex.items():
        top = e["permutation_importance"]["groups"][0]
        macro(f"TopGroup{MACRO_TASK[task]}", top["group"])
        macro(f"TopGroupDrop{MACRO_TASK[task]}", f"{top['auroc_drop']:.3f}")
        macro(f"TopShap{MACRO_TASK[task]}", e["treeshap"][0]["feature"].replace("_", "\\_"))


def shift_macros():
    sh = load("temporal_shift.json")
    for part in ("train", "test"):
        tag = part.capitalize()
        macro(f"Shift{tag}DxMax", str(sh[part]["number_diagnoses_max"]))
        macro(f"Shift{tag}DxMean", f"{sh[part]['number_diagnoses_mean']:.1f}")
        macro(f"Shift{tag}Hist", f"{100 * sh[part]['share_with_history']:.1f}")
        macro(f"Shift{tag}MaxPrior", str(sh[part]["max_prior_stays"]))
        macro(f"Shift{tag}Emerg", f"{sh[part]['number_emergency_mean']:.2f}")


def kg_table():
    rows = load("kg_efficiency.json")
    lines = []
    for task in ["readmit30", "escalation"]:
        for r in [x for x in rows if x["task"] == task]:
            frac = f"{int(round(100 * r['fraction']))}\\%"
            lines.append(
                f"{TASK_NAMES[task].split(' at')[0].capitalize() if r is rows[0] or r['fraction'] == 0.05 else ''} & {frac} & "
                f"{r['tkgn_auroc']:.4f} & {r['no_kg_auroc']:.4f} & {r['delta_auroc']:+.4f}$\\pm${r['delta_sd']:.4f} & "
                f"{r['wins']}/{r['n_runs']} \\\\")
            tag = {0.05: "Five", 0.1: "Ten", 0.25: "TwentyFive", 1.0: "Full"}[r["fraction"]]
            macro(f"Kg{MACRO_TASK[task]}{tag}", f"{r['delta_auroc']:+.4f}")
            macro(f"KgWins{MACRO_TASK[task]}{tag}", f"{r['wins']}/{r['n_runs']}")
        lines.append("\\midrule")
    (TAB / "tab_kg.tex").write_text(
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
        "Task & Training share & TKGN & TKGN $-$ KG & $\\Delta$AUROC (mean$\\pm$SD) & Wins \\\\\n\\midrule\n"
        + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def split_macros():
    sp = load("splits.json")
    for key, v in sp.items():
        task, split = key.split("_")
        for part in ("train", "val", "test"):
            macro(f"N{MACRO_TASK[task]}{MACRO_SPLIT[split]}{part.capitalize()}",
                  f"{v['n'][part]:,}")


if __name__ == "__main__":
    for fn in (cohort_table, results_tables, significance_table, ablation_table,
               subgroup_table, learning_macros, gate_macros, explanation_macros,
               split_macros, shift_macros, kg_table):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")
    (TAB / "results_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"{len(macros)} macros and tables written to {TAB}")
