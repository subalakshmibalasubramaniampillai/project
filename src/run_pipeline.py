"""
End-to-end experiment runner on the real Diabetes 130-US Hospitals data.

    python -m src.run_pipeline                 # full protocol
    python -m src.run_pipeline --quick         # small smoke run

Protocol
    tasks      : readmit30, escalation
    splits     : patient-grouped 70/10/20 (``--repeats`` different random
                 partitions) and a prospective temporal split
                 (``--temporal-seeds`` model seeds)
    models     : logistic regression, random forest, XGBoost, LightGBM,
                 MLP, GRU, RETAIN, Transformer, TKGN (proposed) and
                 TKGN-B (proposed, GBDT-anchored)
    ablations  : TKGN component removals on the first ``--ablation-repeats``
                 grouped repeats
    extra      : learning curve over training-set fractions

Every job writes ``outputs/runs/<job>.json`` (metrics) and ``.npz`` (test
predictions).  Completed jobs are skipped, so the runner can resume.
``aggregate()`` then builds the summary files served by the API.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNS = OUT / "runs"
MODELS_DIR = OUT / "models"

CLASSICAL = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
             "mlp"]
SEQUENCE = ["gru", "retain", "transformer", "tkgn"]
PROPOSED = ["tkgn", "tkgn_b"]
ABLATIONS = ["tkgn_no_kg", "tkgn_no_cooccurrence", "tkgn_no_history",
             "tkgn_no_gate", "tkgn_no_delta", "tkgn_no_prior_outcomes",
             "tkgn_no_aux"]
MAX_HISTORY = 10

# one optimisation recipe for every neural model; TKGN additionally uses
# the auxiliary outcome head (removed in the ``tkgn_no_aux`` ablation)
NEURAL_HP = {"lr": 5e-4, "weight_decay": 1e-3, "max_epochs": 30,
             "patience": 4, "batch_size": 512}
DROPOUT = 0.3
AUX_WEIGHT = 0.5

_CACHE: dict = {}


def _cohort():
    if "cohort" not in _CACHE:
        from .data import build_cohort, history_index
        cohort = build_cohort()
        _CACHE["cohort"] = cohort
        _CACHE["history"] = history_index(cohort, MAX_HISTORY)
    return _CACHE["cohort"], _CACHE["history"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _split(cohort, task, kind, repeat):
    from .data import grouped_split, temporal_split
    if kind == "grouped":
        return grouped_split(cohort, task, seed=1000 + repeat)
    return temporal_split(cohort, task)


def _oof_gbdt_offset(cohort, x, y, split, params, n_iter, seed, folds=5):
    """Out-of-fold LightGBM logits on train; full-train logits elsewhere."""
    import lightgbm as lgb
    train = split["train"]
    pids = cohort["patient_nbr"].to_numpy()[train]
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(unique, rng.integers(0, folds, len(unique))))
    fold = np.array([fold_of[p] for p in pids])
    offset = np.zeros(len(cohort), dtype=np.float32)
    for k in range(folds):
        model = lgb.LGBMClassifier(
            n_estimators=n_iter, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.6, n_jobs=2,
            random_state=seed, verbose=-1, **params)
        model.fit(x[train[fold != k]], y[train[fold != k]])
        offset[train[fold == k]] = _logit(
            model.predict_proba(x[train[fold == k]])[:, 1])
    return offset


def run_job(task: str, kind: str, repeat: int, models: list[str],
            train_fraction: float = 1.0, tag: str = "", save_models=False,
            threads: int = 2, verbose=False) -> str:
    import torch
    torch.set_num_threads(threads)
    from .baselines import fit_classical
    from .data import task_labels
    from .evaluation import choose_threshold, evaluate
    from .features import SequenceEncoder, TabularEncoder
    from .knowledge_graph import ClinicalKnowledgeGraph
    from .neural import (
        SequenceData, build_model, predict, train_model, uses_prior_outcomes,
    )

    name = f"{task}_{kind}_r{repeat}{tag}"
    out_json = RUNS / f"{name}.json"
    if out_json.exists():
        return name
    started = time.time()
    cohort, history = _cohort()
    y = np.clip(task_labels(cohort, task), 0, 1).astype(np.float32)
    split = _split(cohort, task, kind, repeat)
    if train_fraction < 1.0:
        pids = cohort["patient_nbr"].to_numpy()[split["train"]]
        unique = np.unique(pids)
        rng = np.random.default_rng(500 + repeat)
        keep = rng.choice(unique, int(train_fraction * len(unique)),
                          replace=False)
        split = dict(split, train=split["train"][np.isin(pids, keep)])
    tr, va, te = split["train"], split["val"], split["test"]
    seed = repeat

    record = {"task": task, "split": kind, "repeat": repeat,
              "train_fraction": train_fraction,
              "n": {k: int(len(v)) for k, v in split.items()},
              "prevalence": {k: float(y[v].mean()) for k, v in split.items()},
              "models": {}}
    preds_val, preds_test, extras = {}, {}, {}

    tab = TabularEncoder().fit(cohort, tr)
    x = tab.transform(cohort).to_numpy()
    lgbm_info = None
    for model_name in [m for m in models if m in CLASSICAL]:
        model, info = fit_classical(model_name, x[tr], y[tr], x[va], y[va], seed)
        preds_val[model_name] = model.predict_proba(x[va])[:, 1]
        preds_test[model_name] = model.predict_proba(x[te])[:, 1]
        record["models"][model_name] = {"train": info}
        if model_name == "lightgbm":
            lgbm_info = (model, info)
        if verbose:
            print(f"  [{name}] {model_name} done ({info['train_seconds']:.0f}s)",
                  flush=True)

    neural = [m for m in models if m not in CLASSICAL]
    if neural:
        kg = ClinicalKnowledgeGraph().fit(cohort, tr)
        seq = SequenceEncoder().fit(cohort, tr)
        arrays = seq.transform(cohort, kg.encode_diagnoses(cohort))
        offset = None
        if "tkgn_b" in neural:
            if lgbm_info is None:
                lgbm_info = fit_classical("lightgbm", x[tr], y[tr], x[va],
                                          y[va], seed)
            lgbm_model, info = lgbm_info
            offset = _oof_gbdt_offset(cohort, x, y, split, info["params"],
                                      lgbm_model.best_iteration_ or 200, seed)
            offset[va] = _logit(lgbm_model.predict_proba(x[va])[:, 1])
            offset[te] = _logit(lgbm_model.predict_proba(x[te])[:, 1])

        for model_name in neural:
            arch = {"tkgn_b": "tkgn", "tkgn_no_aux": "tkgn"}.get(model_name,
                                                                 model_name)
            aux = AUX_WEIGHT if (arch.startswith("tkgn")
                                 and model_name != "tkgn_no_aux") else 0.0
            data = SequenceData(arrays, history, y,
                                prior_outcomes=uses_prior_outcomes(arch),
                                offset=offset if model_name == "tkgn_b" else None)
            torch.manual_seed(seed)
            model = build_model(arch, kg, seq, max_history=MAX_HISTORY,
                                dropout=DROPOUT)
            info = train_model(model, data, tr, va, seed=seed,
                               aux_weight=aux, **NEURAL_HP)
            preds_val[model_name] = predict(model, data, va)
            p_test, gates = predict(model, data, te, with_extras=True)
            preds_test[model_name] = p_test
            if gates is not None and model_name in ("tkgn", "tkgn_b"):
                extras[f"{model_name}_gates"] = gates.astype(np.float32)
            record["models"][model_name] = {"train": info}
            if verbose:
                print(f"  [{name}] {model_name} done "
                      f"({info['train_seconds']:.0f}s, "
                      f"val={info['best_val_auroc']:.4f})", flush=True)
            if save_models and model_name in ("tkgn", "tkgn_b"):
                _save_deployment(task, model_name, model, kg, seq, tab,
                                 lgbm_info[0] if model_name == "tkgn_b" else None)

    for model_name in preds_test:
        threshold = choose_threshold(y[va], preds_val[model_name])
        record["models"][model_name]["test"] = evaluate(
            y[te], preds_test[model_name], threshold)
        record["models"][model_name]["val_auroc"] = float(
            evaluate(y[va], preds_val[model_name], threshold)["auroc"])

    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RUNS / f"{name}.npz", test_rows=te, y=y[te],
        **{f"p_{k}": v.astype(np.float32) for k, v in preds_test.items()},
        **extras)
    record["seconds"] = time.time() - started
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return name


def _save_deployment(task, model_name, model, kg, seq, tab, gbdt):
    import joblib
    import torch
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{task}_{model_name}.pt")
    joblib.dump({"kg": kg, "seq": seq, "tab": tab, "gbdt": gbdt,
                 "max_history": MAX_HISTORY, "dropout": DROPOUT},
                MODELS_DIR / f"{task}_{model_name}_context.joblib")


def build_jobs(args) -> list[dict]:
    tasks = args.tasks.split(",")
    base = CLASSICAL + SEQUENCE + ["tkgn_b"]
    jobs = []
    for task in tasks:
        for r in range(args.repeats):
            models = base + (ABLATIONS if r < args.ablation_repeats else [])
            jobs.append(dict(task=task, kind="grouped", repeat=r,
                             models=models, save_models=(r == 0)))
        for r in range(args.temporal_seeds):
            jobs.append(dict(task=task, kind="temporal", repeat=r,
                             models=base))
        for frac in args.fractions:
            jobs.append(dict(task=task, kind="grouped", repeat=0,
                             models=["lightgbm", "gru", "tkgn", "tkgn_b"],
                             train_fraction=frac, tag=f"_f{frac:g}"))
    return jobs


def _run(job, threads, verbose):
    t0 = time.time()
    name = run_job(threads=threads, verbose=verbose, **job)
    return name, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tasks", default="readmit30,escalation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ablation-repeats", type=int, default=3)
    parser.add_argument("--temporal-seeds", type=int, default=3)
    parser.add_argument("--fractions", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--quick", action="store_true",
                        help="1 repeat, no ablations/temporal/learning curve")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.fractions = [float(f) for f in args.fractions.split(",") if f]
    if args.quick:
        args.repeats, args.ablation_repeats, args.temporal_seeds = 1, 0, 0
        args.fractions = []

    if not args.aggregate_only:
        jobs = build_jobs(args)
        print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
        if args.workers <= 1:
            for job in jobs:
                print(_run(job, args.threads, args.verbose), flush=True)
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                futures = [pool.submit(_run, job, args.threads, args.verbose)
                           for job in jobs]
                for future in futures:
                    name, secs = future.result()
                    print(f"finished {name} in {secs / 60:.1f} min", flush=True)

    from .analysis import aggregate
    aggregate()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
