"""
Classical baselines on the tabular view (current encounter + history
aggregates).  Hyper-parameters are selected on the validation partition
only; the test partition is never touched during selection.
"""
from __future__ import annotations

import itertools
import time
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

CLASSICAL_MODELS = ["logistic_regression", "random_forest", "xgboost",
                    "lightgbm", "mlp"]


def _grid(name):
    if name == "logistic_regression":
        return [{"C": c} for c in (0.01, 0.1, 1.0)]
    if name == "random_forest":
        return [{"min_samples_leaf": m} for m in (5, 20)]
    if name == "xgboost":
        return [{"max_depth": d, "min_child_weight": w}
                for d, w in itertools.product((4, 6), (1, 10))]
    if name == "lightgbm":
        return [{"num_leaves": l, "min_child_samples": m}
                for l, m in itertools.product((15, 63), (20, 100))]
    if name == "mlp":
        return [{"hidden_layer_sizes": h, "alpha": a}
                for h, a in itertools.product(((128, 64), (64,)), (1e-4, 1e-2))]
    raise ValueError(name)


def _make(name, params, seed):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=params["C"], max_iter=3000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=2,
            random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=2000, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.6, tree_method="hist", eval_metric="auc",
            early_stopping_rounds=100, n_jobs=2, random_state=seed, **params)
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.6, n_jobs=2,
            random_state=seed, verbose=-1, **params)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            early_stopping=True, max_iter=200, random_state=seed, **params))
    raise ValueError(name)


def _fit(model, name, x_tr, y_tr, x_va, y_va):
    if name == "xgboost":
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    elif name == "lightgbm":
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(x_tr, y_tr)
    return model


def fit_classical(name, x_tr, y_tr, x_va, y_va, seed):
    """Validation-set model selection; returns (model, info)."""
    started = time.time()
    best = (-1.0, None, None)
    for params in _grid(name):
        model = _fit(_make(name, params, seed), name, x_tr, y_tr, x_va, y_va)
        auc = roc_auc_score(y_va, model.predict_proba(x_va)[:, 1])
        if auc > best[0]:
            best = (auc, params, model)
    return best[2], {"best_val_auroc": float(best[0]), "params": best[1],
                     "train_seconds": time.time() - started}
