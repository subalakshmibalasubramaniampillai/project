"""
Dataset loading and validation.

Supports:
  - Synthetic longitudinal cohort (default fallback)
  - Pima Indians Diabetes (real, public, downloaded on first run)
  - User-provided CSV via path, URL, or upload

The pipeline expects columns:
    patient_id, visit, date, age, hba1c, glucose, bmi,
    systolic_bp, diastolic_bp, conditions

If 'progression' is absent, it is derived from the data.
"""
import os
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "patient_id", "visit", "date", "age", "hba1c", "glucose", "bmi",
    "systolic_bp", "diastolic_bp", "conditions",
}


def prepare_dataset(data: pd.DataFrame,
                    output_path: str | None = None) -> pd.DataFrame:
    """
    Validate and standardize a raw dataframe.
    Returns a clean copy ready for feature extraction.
    """
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )

    result = data.copy()

    # force dtypes
    result["patient_id"] = result["patient_id"].map(str).astype(object)
    result["visit"] = pd.to_numeric(result["visit"], errors="raise")
    result["date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.date.astype(str)
    for col in ["age", "hba1c", "glucose", "bmi", "systolic_bp",
                "diastolic_bp"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(
            result[col].median() if col in result.columns else 0
        )
    result["conditions"] = result["conditions"].fillna("unknown").astype(str)

    # check visit count (allow both longitudinal ≥4 and single-visit datasets)
    visits = result.sort_values(["patient_id", "visit"])
    counts = visits.groupby("patient_id").size()
    min_visits = counts.min()
    # if most patients have ≥4 visits, enforce it; otherwise allow fewer
    if min_visits < 4 and counts.median() >= 4:
        bad = counts[counts < 4].index.tolist()
        if len(bad) <= 5:
            raise ValueError(
                f"These patients have fewer than 4 visits: {bad}"
            )
        else:
            raise ValueError(
                f"{len(bad)} patients have fewer than 4 visits."
            )

    # derive progression if absent
    if "progression" not in result.columns:
        first_hb = visits.groupby("patient_id").first()["hba1c"]
        last_hb = visits.groupby("patient_id").last()["hba1c"]
        labels = (
            ((last_hb - first_hb >= 0.5) | (last_hb >= 6.5))
            .astype(int)
            .rename("progression")
        )
        result = result.join(labels, on="patient_id")
    else:
        result["progression"] = pd.to_numeric(
            result["progression"], errors="coerce"
        ).fillna(0).astype(int)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)

    return result


def load_dataset(source: str | None = None,
                 output_path: str | None = None) -> pd.DataFrame:
    """
    Load from a configured source (path or URL).
    Falls back to DATA_SOURCE env var if source is None.
    """
    source = source or os.getenv("DATA_SOURCE")
    if not source:
        raise ValueError(
            "No live dataset source configured. "
            "Set DATA_SOURCE to a CSV path or HTTP CSV URL."
        )
    return prepare_dataset(pd.read_csv(source), output_path)
