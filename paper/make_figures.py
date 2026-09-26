"""
Generate every manuscript figure from the pipeline outputs.

    python paper/make_figures.py

Figures are written to paper/figures/ as PDF (vector, for LaTeX) and PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

LABELS = {
    "logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "mlp": "MLP", "gru": "GRU", "retain": "RETAIN",
    "transformer": "Transformer", "tkgn": "TKGN", "tkgn_b": "TKGN-B",
}
COLORS = {
    "logistic_regression": "#9aa3b5", "random_forest": "#7c89a3",
    "xgboost": "#8f6bd1", "lightgbm": "#3d6fd1", "mlp": "#6d7f9e",
    "gru": "#4fb3b3", "retain": "#2f8f8b", "transformer": "#236e6b",
    "tkgn": "#e08a1e", "tkgn_b": "#1f9e74",
}
MAIN = list(LABELS)
TASK_NAMES = {"readmit30": "30-day readmission",
              "escalation": "Treatment escalation"}


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ──────────────────────────────────────────────
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, color="#eef2fa", edge="#3d4b66", size=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=color, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.8, color="#3d4b66"))

    # inputs
    box(1, 34, 17, 9, "Encounter $e_{t-k}$\n(dx, drugs, labs,\nadmin, outcome)")
    box(1, 22, 17, 9, "Encounter $e_{t-1}$")
    box(1, 8, 17, 10, "Index encounter $e_t$\n(outcome masked)")
    # KG
    box(22, 34, 22, 10, "Hybrid knowledge graph:\nICD-9 hierarchy + PPMI\nco-morbidity (train)",
        color="#fff4e3", edge="#e08a1e", size=6.6)
    box(22, 15, 22, 15, "Shared encounter\nencoder: ancestor-\nattention codes, drug-\nclass tokens, binned\nnumerics",
        color="#eef2fa", size=6.6)
    arrow(33, 34, 33, 29.5)
    for y in (38.5, 26.5, 13):
        arrow(18.5, y, 21.5, 22.5)
    # temporal
    box(48, 30, 20, 11, "History summary:\nGRU + recency\nattention", color="#e8f6f4",
        edge="#2f8f8b")
    box(48, 16, 20, 10, "Delta encoding\n$h_t - h_{t-1}$", color="#e8f6f4", edge="#2f8f8b")
    box(48, 3, 20, 9, "History length\nembedding $n_t$", color="#e8f6f4", edge="#2f8f8b")
    arrow(44.5, 24, 47.5, 35)
    arrow(44.5, 22, 47.5, 21)
    # gate
    box(72, 16, 12, 18, "History-\nreliability\ngates\n$g_s,\\ g_\\delta$",
        color="#fff4e3", edge="#e08a1e")
    arrow(68.5, 35, 71.5, 29)
    arrow(68.5, 21, 71.5, 24)
    arrow(68.5, 7.5, 71.5, 19)
    # heads
    box(87, 27, 12, 11, "Risk head\n+ aux. 3-level\noutcome head", size=6.8)
    box(87, 8, 12, 12, "TKGN-B:\n+ out-of-fold\nGBDT logit\n(anchor)", color="#e3f5ee",
        edge="#1f9e74", size=6.8)
    arrow(84.5, 28, 86.5, 32)
    arrow(93, 20.5, 93, 26.5)
    _save(fig, "fig_framework")


def fig_cohort():
    d = _load("dataset_summary.json")
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    items = [
        (8.6, f"UCI Diabetes 130-US Hospitals\n{d['raw_encounters']:,} encounters"),
        (6.2, f"Study cohort\n{d['cohort_encounters']:,} encounters, {d['patients']:,} patients\n"
              f"({d['patients_with_repeat_encounters']:,} with repeat stays)"),
    ]
    for y, text in items:
        ax.add_patch(FancyBboxPatch((1, y - 0.9), 8, 1.8, boxstyle="round,pad=0.1",
                                    fc="#eef2fa", ec="#3d4b66", lw=0.8))
        ax.text(5, y, text, ha="center", va="center", fontsize=7.5)
    ax.annotate("", (5, 7.1), (5, 7.7), arrowprops=dict(arrowstyle="-|>", lw=0.8))
    excluded = d["raw_encounters"] - d["cohort_encounters"]
    ax.text(5.3, 7.4, f"excluded {excluded:,}: death/hospice\ndischarge, unknown sex",
            fontsize=6.5, va="center")
    for x, text in [(0.6, f"Task 1: 30-day readmission\nall {d['cohort_encounters']:,} stays\n"
                          f"positive {100 * d['readmit30_rate']:.1f}%"),
                    (5.2, f"Task 2: escalation at next stay\n{d['escalation_samples']:,} stays with a\n"
                          f"later stay; positive {100 * d['escalation_rate']:.1f}%")]:
        ax.add_patch(FancyBboxPatch((x, 1.4), 4.2, 2.4, boxstyle="round,pad=0.1",
                                    fc="#fff4e3", ec="#e08a1e", lw=0.8))
        ax.text(x + 2.1, 2.6, text, ha="center", va="center", fontsize=6.8)
        ax.annotate("", (x + 2.1, 3.9), (5, 5.2), arrowprops=dict(arrowstyle="-|>", lw=0.8))
    _save(fig, "fig_cohort")


def fig_auroc():
    rows = _load("summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    for ax, task in zip(axes, ["readmit30", "escalation"]):
        for offset, split, hatch in [(-0.2, "grouped", None), (0.2, "temporal", "///")]:
            sub = {r["model"]: r for r in rows if r["task"] == task and r["split"] == split}
            xs = [i + offset for i, m in enumerate(MAIN) if m in sub]
            vals = [sub[m]["auroc"] for m in MAIN if m in sub]
            errs = [sub[m]["auroc_sd"] for m in MAIN if m in sub]
            ax.bar(xs, vals, width=0.38, yerr=errs, capsize=1.5,
                   color=[COLORS[m] for m in MAIN if m in sub], hatch=hatch,
                   edgecolor="white", linewidth=0.3,
                   label="patient-grouped" if split == "grouped" else "temporal")
        ax.set_xticks(range(len(MAIN)))
        ax.set_xticklabels([LABELS[m] for m in MAIN], rotation=45, ha="right")
        ax.set_title(TASK_NAMES[task])
        lo = min(r["auroc"] for r in rows if r["task"] == task and r["model"] in MAIN)
        hi = max(r["auroc"] for r in rows if r["task"] == task and r["model"] in MAIN)
        ax.set_ylim(lo - 0.02, hi + 0.012)
        ax.set_ylabel("Test AUROC")
    axes[0].legend(loc="lower right", frameon=False)
    _save(fig, "fig_auroc")


def fig_calibration():
    cal = _load("calibration.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, task in zip(axes, ["readmit30", "escalation"]):
        curves = cal.get(f"{task}_grouped", {})
        top = 0
        for m in ["logistic_regression", "lightgbm", "gru", "tkgn", "tkgn_b"]:
            if m not in curves:
                continue
            xs = [b["mean_predicted"] for b in curves[m]]
            ys = [b["observed"] for b in curves[m]]
            top = max(top, max(xs), max(ys))
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.1, color=COLORS[m], label=LABELS[m])
        ax.plot([0, top], [0, top], ls="--", lw=0.8, color="grey")
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed frequency")
        ax.set_title(TASK_NAMES[task])
    axes[0].legend(frameon=False)
    _save(fig, "fig_calibration")


def fig_learning_curve():
    rows = _load("learning_curve.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, task in zip(axes, ["readmit30", "escalation"]):
        for m in ["lightgbm", "gru", "tkgn", "tkgn_b"]:
            pts = sorted([r for r in rows if r["task"] == task and r["model"] == m],
                         key=lambda r: r["fraction"])
            if not pts:
                continue
            ax.plot([100 * p["fraction"] for p in pts], [p["auroc"] for p in pts],
                    marker="o", ms=3, color=COLORS[m], label=LABELS[m])
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 25, 50, 100])
        ax.set_xticklabels(["5", "10", "25", "50", "100"])
        ax.set_xlabel("Training patients used (%)")
        ax.set_ylabel("Test AUROC")
        ax.set_title(TASK_NAMES[task])
    axes[0].legend(frameon=False)
    _save(fig, "fig_learning_curve")


def fig_ablation():
    rows = [r for r in _load("ablation.json") if r["model"] != "tkgn"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    order = sorted({r["label"] for r in rows})
    for ax, task in zip(axes, ["readmit30", "escalation"]):
        sub = {r["label"]: r for r in rows if r["task"] == task}
        vals = [sub[l]["delta_auroc_vs_full"] if l in sub else 0 for l in order]
        ax.barh(range(len(order)), vals,
                color=["#1f9e74" if v < 0 else "#c9483f" for v in vals])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([l.replace("TKGN - ", "without ") for l in order])
        ax.set_xlabel("ΔAUROC vs. full TKGN")
        ax.set_title(TASK_NAMES[task])
    _save(fig, "fig_ablation")


def fig_gates():
    rows = _load("gates.json")
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for task, ls in [("readmit30", "-"), ("escalation", "--")]:
        sub = [r for r in rows if r["task"] == task]
        xs = [r["prior_encounters"] for r in sub]
        ax.plot(xs, [r["gate_history"] for r in sub], ls=ls, marker="o", ms=3,
                color="#e08a1e", label=f"history gate, {TASK_NAMES[task]}")
        ax.plot(xs, [r["gate_delta"] for r in sub], ls=ls, marker="s", ms=3,
                color="#2f8f8b", label=f"delta gate, {TASK_NAMES[task]}")
    ax.set_xlabel("Earlier stays of the patient")
    ax.set_ylabel("Mean gate activation")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, "fig_gates")


def fig_importance():
    ex = _load("explanations.json")
    tasks = [t for t in ["readmit30", "escalation"] if t in ex]
    fig, axes = plt.subplots(len(tasks), 2, figsize=(7.2, 2.6 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]
    for row, task in zip(axes, tasks):
        perm = ex[task]["permutation_importance"]["groups"][::-1]
        row[0].barh([g["group"] for g in perm], [g["auroc_drop"] for g in perm],
                    xerr=[g["sd"] for g in perm], color="#e08a1e")
        row[0].set_xlabel("AUROC drop when shuffled")
        row[0].set_title(f"TKGN input groups — {TASK_NAMES[task]}")
        shap = ex[task]["treeshap"][:12][::-1]
        row[1].barh([s["feature"][:34] for s in shap], [s["mean_abs_shap"] for s in shap],
                    color="#3d6fd1")
        row[1].set_xlabel("mean |SHAP| (log-odds)")
        row[1].set_title(f"GBDT anchor features — {TASK_NAMES[task]}")
    fig.tight_layout()
    _save(fig, "fig_importance")


def fig_subgroups():
    rows = _load("subgroups.json")
    attrs = ["race", "gender", "age", "prior_encounters"]
    fig, axes = plt.subplots(1, len(attrs), figsize=(7.2, 2.4), sharey=True)
    for ax, attr in zip(axes, attrs):
        sub = [r for r in rows if r["task"] == "readmit30" and r["attribute"] == attr]
        groups = sorted({r["group"] for r in sub})
        width = 0.26
        for k, m in enumerate(["lightgbm", "tkgn", "tkgn_b"]):
            vals = [next((r["auroc"] for r in sub if r["group"] == g and r["model"] == m), 0)
                    for g in groups]
            ax.bar([i + (k - 1) * width for i in range(len(groups))], vals, width,
                   color=COLORS[m], label=LABELS[m])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(attr.replace("_", " "))
        ax.set_ylim(0.55, 0.78)
    axes[0].set_ylabel("AUROC (30-day readmission)")
    axes[0].legend(frameon=False, fontsize=6)
    _save(fig, "fig_subgroups")


if __name__ == "__main__":
    fig_framework()
    for fn in (fig_cohort, fig_auroc, fig_calibration, fig_learning_curve,
               fig_ablation, fig_gates, fig_importance, fig_subgroups):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")
    print(f"figures written to {FIG}")
