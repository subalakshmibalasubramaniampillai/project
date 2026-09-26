"""
Evaluation utilities.

* Discrimination: AUROC, AUPRC.
* Calibration: Brier score, expected calibration error (ECE, 10 equal-
  width bins), calibration intercept and slope (logistic recalibration).
* Threshold metrics at an operating point chosen on the *validation*
  partition (maximum F1), then frozen and applied to the test partition.
* Uncertainty: patient-clustered bootstrap confidence intervals, paired
  bootstrap and DeLong tests for AUROC differences.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score,
)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def calibration_slope_intercept(y, p):
    """Cox recalibration: logit P(y) = a + b * logit(p)."""
    z = _logit(p)

    def nll(theta):
        a, b = theta
        eta = a + b * z
        return np.sum(np.logaddexp(0, eta) - y * eta)

    slope_fit = optimize.minimize(nll, x0=[0.0, 1.0], method="BFGS")

    def nll_offset(a):
        eta = a[0] + z
        return np.sum(np.logaddexp(0, eta) - y * eta)

    intercept_fit = optimize.minimize(nll_offset, x0=[0.0], method="BFGS")
    return float(intercept_fit.x[0]), float(slope_fit.x[1])


def calibration_curve(y, p, bins=10):
    """Equal-frequency reliability curve."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return [{"mean_predicted": float(p[c].mean()),
             "observed": float(y[c].mean()), "n": int(len(c))}
            for c in chunks if len(c)]


def choose_threshold(y_val, p_val):
    """Threshold maximising F1 on the validation partition."""
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def threshold_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    return {"threshold": float(threshold), "sensitivity": sens,
            "specificity": spec, "ppv": ppv, "npv": npv, "f1": f1,
            "balanced_accuracy": (sens + spec) / 2,
            "accuracy": (tp + tn) / len(y)}


def evaluate(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    intercept, slope = calibration_slope_intercept(y, p)
    result = {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    result.update(threshold_metrics(y, p, threshold))
    return result


# ──────────────────────────────────────────────
# Uncertainty
# ──────────────────────────────────────────────

def _cluster_resamples(groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [[] for _ in unique]
    for i, g in enumerate(inverse):
        members[g].append(i)
    members = [np.asarray(m) for m in members]
    for _ in range(n_boot):
        pick = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([members[k] for k in pick])


def bootstrap_ci(y, p, groups, n_boot=1000, seed=0):
    """Patient-clustered percentile 95% CIs for AUROC, AUPRC and Brier."""
    y = np.asarray(y)
    p = np.asarray(p)
    draws = {"auroc": [], "auprc": [], "brier": []}
    for idx in _cluster_resamples(groups, n_boot, seed):
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        draws["auroc"].append(roc_auc_score(yy, pp))
        draws["auprc"].append(average_precision_score(yy, pp))
        draws["brier"].append(brier_score_loss(yy, pp))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items()}


def paired_bootstrap(y, p_a, p_b, groups, n_boot=1000, seed=0):
    """Patient-clustered bootstrap of AUROC(a) - AUROC(b)."""
    y = np.asarray(y)
    diffs = []
    for idx in _cluster_resamples(groups, n_boot, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    diffs = np.asarray(diffs)
    observed = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta_auroc": float(observed),
            "ci": [float(np.percentile(diffs, 2.5)),
                   float(np.percentile(diffs, 97.5))],
            "p_value": float(min(max(p_value, 1.0 / len(diffs)), 1.0))}


def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y, p_a, p_b):
    """
    Paired DeLong test for two correlated AUROCs, using the fast
    mid-rank formulation of Sun & Xu (2014).
    """
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p_a, p_b])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return {"auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]),
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float),
                          encoding="utf-8")
