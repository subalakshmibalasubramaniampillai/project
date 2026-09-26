"""
Pre-specified follow-up: does the knowledge graph help when training
data are scarce?  TKGN and TKGN without the knowledge graph are retrained
on 5/10/25% of training patients in three grouped repeats (paired: same
patients, same seed) and compared on the full test partitions.

    python -m src.kg_efficiency
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .run_pipeline import OUT, RUNS, run_job

FRACTIONS = (0.05, 0.1, 0.25)
REPEATS = (0, 1, 2)
TASKS = ("readmit30", "escalation")


def _job(args):
    task, repeat, frac = args
    return run_job(task, "grouped", repeat, ["tkgn", "tkgn_no_kg"],
                   train_fraction=frac, tag=f"_kg_f{frac:g}", threads=2)


def summarise():
    rows = []
    for task in TASKS:
        for frac in FRACTIONS + (1.0,):
            deltas, full, flat = [], [], []
            for r in REPEATS:
                name = (f"{task}_grouped_r{r}" if frac == 1.0
                        else f"{task}_grouped_r{r}_kg_f{frac:g}")
                path = RUNS / f"{name}.json"
                if not path.exists():
                    continue
                models = json.loads(path.read_text())["models"]
                a = models["tkgn"]["test"]["auroc"]
                b = models["tkgn_no_kg"]["test"]["auroc"]
                full.append(a)
                flat.append(b)
                deltas.append(a - b)
            if deltas:
                rows.append({"task": task, "fraction": frac, "n_runs": len(deltas),
                             "tkgn_auroc": float(np.mean(full)),
                             "no_kg_auroc": float(np.mean(flat)),
                             "delta_auroc": float(np.mean(deltas)),
                             "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                             "wins": int(sum(d > 0 for d in deltas))})
    (OUT / "kg_efficiency.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    jobs = [(t, r, f) for t in TASKS for r in REPEATS for f in FRACTIONS]
    with ProcessPoolExecutor(2) as pool:
        for name in pool.map(_job, jobs):
            print("finished", name, flush=True)
    for row in summarise():
        print(row)
