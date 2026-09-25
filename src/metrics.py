"""
Evaluation metrics.

All metrics are computed at the patient level using the test split.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)


def calculate_metrics(y_true, probabilities, model_name,
                      split_name="patient-level"):
    """
    Compute standard classification metrics from predicted probabilities.

    Returns a dict with accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
    Brier score, and log loss.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "model": model_name,
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )
        result["brier"] = float(brier_score_loss(y_true, probabilities))
        result["log_loss"] = float(
            log_loss(y_true, np.column_stack([1 - probabilities, probabilities])
            )
        )
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
        result["brier"] = None
        result["log_loss"] = None

    return result


def save_json(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
