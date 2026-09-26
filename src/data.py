"""
Real-world data layer: Diabetes 130-US Hospitals (1999-2008).

Source
    Strack et al., BioMed Research International, 2014, article 781670.
    UCI Machine Learning Repository, dataset 296 (CC BY 4.0).
    101,766 de-identified inpatient encounters of patients with diabetes
    from 130 US hospitals, with a stable patient identifier
    (``patient_nbr``) that links repeat admissions of the same person.

The file is bundled in ``data/raw`` and verified by SHA-256. If it is
missing it is downloaded from the UCI archive (or a GitHub mirror of the
identical file) and verified before use.  Nothing in this module
generates, imputes or perturbs clinical values.

Prediction tasks (one sample = one index encounter, features use only
that encounter and the patient's *earlier* encounters):

``readmit30``
    Unplanned readmission within 30 days of discharge
    (``readmitted == "<30"``).
``escalation``
    Diabetes treatment intensification at the patient's next recorded
    encounter: a new insulin start, any anti-diabetic dose increase, or
    more concurrently prescribed anti-diabetic drugs than at the index
    encounter.  Only encounters that have a following encounter are
    eligible.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .icd9 import DRUG_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "diabetic_data.csv"
RAW_SHA256 = "da21942c25a10a51a8842907ab01a8530b06232616b17d317441d592a20c4359"

UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/296/"
               "diabetes+130-us+hospitals+for+years+1999-2008.zip")
MIRROR_URLS = [
    "https://raw.githubusercontent.com/fairlearn/talks/main/"
    "2021_scipy_tutorial/data/diabetic_data.csv",
    "https://raw.githubusercontent.com/andrewwlong/diabetes_readmission/"
    "master/diabetic_data.csv",
]

# discharge dispositions after which readmission is impossible
# (expired, hospice), excluded following Strack et al.
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

TASKS = ("readmit30", "escalation")

NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "payer_code",
    "medical_specialty", "max_glu_serum", "A1Cresult", "change",
    "diabetesMed",
]
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]
OUTCOME_LEVELS = {"NO": 0, ">30": 1, "<30": 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = urllib.request.urlopen(UCI_ZIP_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            name = next(n for n in archive.namelist()
                        if n.endswith("diabetic_data.csv"))
            RAW_FILE.write_bytes(archive.read(name))
        if _sha256(RAW_FILE) == RAW_SHA256:
            return
    except Exception:
        pass
    for url in MIRROR_URLS:
        try:
            urllib.request.urlretrieve(url, RAW_FILE)
            if _sha256(RAW_FILE) == RAW_SHA256:
                return
        except Exception:
            continue
    raise RuntimeError(
        "Could not obtain diabetic_data.csv with the expected checksum. "
        f"Download it from {UCI_ZIP_URL} and place it in {RAW_DIR}."
    )


def load_raw(verify: bool = True) -> pd.DataFrame:
    """Load the untouched UCI file (downloading it if necessary)."""
    if not RAW_FILE.exists():
        _download()
    if verify and _sha256(RAW_FILE) != RAW_SHA256:
        raise RuntimeError(f"{RAW_FILE} does not match the UCI release "
                           "checksum; refusing to train on altered data.")
    return pd.read_csv(RAW_FILE, na_values="?", keep_default_na=False,
                       low_memory=False)


def build_cohort(raw: pd.DataFrame | None = None,
                 apply_exclusions: bool = True) -> pd.DataFrame:
    """
    Apply cohort rules and derive per-encounter labels.

    Encounters are ordered within each patient by ``encounter_id``, which
    increases with admission time in this release (the dataset carries no
    calendar dates).  The returned frame is sorted by patient and order.
    """
    frame = load_raw() if raw is None else raw.copy()
    if apply_exclusions:
        frame = frame[~frame.discharge_disposition_id.isin(EXCLUDED_DISPOSITIONS)]
        frame = frame[frame.gender != "Unknown/Invalid"]
    frame = frame.sort_values(["patient_nbr", "encounter_id"])
    frame = frame.reset_index(drop=True)

    frame["patient_nbr"] = frame["patient_nbr"].astype(np.int64)
    frame["order"] = frame.groupby("patient_nbr").cumcount()
    frame["n_prior"] = frame["order"]
    frame["outcome_level"] = frame["readmitted"].map(OUTCOME_LEVELS)

    # task 1: 30-day readmission
    frame["y_readmit30"] = (frame["readmitted"] == "<30").astype(int)

    # task 2: treatment escalation at the next encounter
    active = frame[DRUG_COLUMNS].ne("No")
    frame["n_active_drugs"] = active.sum(axis=1)
    frame["any_dose_up"] = frame[DRUG_COLUMNS].eq("Up").any(axis=1)
    frame["on_insulin"] = frame["insulin"].ne("No")
    grouped = frame.groupby("patient_nbr")
    next_active = grouped["n_active_drugs"].shift(-1)
    next_up = grouped["any_dose_up"].shift(-1)
    next_insulin = grouped["on_insulin"].shift(-1)
    has_next = next_active.notna()
    escalation = (
        next_up.fillna(False).astype(bool)
        | (~frame["on_insulin"] & next_insulin.fillna(False).astype(bool))
        | (next_active > frame["n_active_drugs"])
    )
    frame["has_next"] = has_next
    frame["y_escalation"] = np.where(has_next, escalation.astype(int), -1)
    return frame


def task_mask(cohort: pd.DataFrame, task: str) -> np.ndarray:
    """Boolean mask of encounters that are valid samples for ``task``."""
    if task == "readmit30":
        return np.ones(len(cohort), dtype=bool)
    if task == "escalation":
        return cohort["has_next"].to_numpy()
    raise ValueError(f"unknown task {task}")


def task_labels(cohort: pd.DataFrame, task: str) -> np.ndarray:
    return cohort[f"y_{task}"].to_numpy()


# ──────────────────────────────────────────────
# Splits
# ──────────────────────────────────────────────

def grouped_split(cohort: pd.DataFrame, task: str, seed: int,
                  fractions=(0.7, 0.1, 0.2)) -> dict[str, np.ndarray]:
    """
    Patient-disjoint train/validation/test split.

    Patients (not encounters) are shuffled, stratified by whether the
    patient has any positive sample, so that no person contributes to
    more than one partition.
    Returns encounter-row indices (into ``cohort``) per partition,
    restricted to encounters that are valid samples for ``task``.
    """
    mask = task_mask(cohort, task)
    labels = task_labels(cohort, task)
    eligible = cohort.loc[mask, "patient_nbr"]
    positive = pd.Series(labels[mask] == 1, index=eligible.index)
    patient_pos = positive.groupby(eligible).any()

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for flag in (False, True):
        pids = patient_pos.index[patient_pos.to_numpy() == flag].to_numpy().copy()
        rng.shuffle(pids)
        n_train = int(round(fractions[0] * len(pids)))
        n_val = int(round(fractions[1] * len(pids)))
        parts["train"].append(pids[:n_train])
        parts["val"].append(pids[n_train:n_train + n_val])
        parts["test"].append(pids[n_train + n_val:])

    out = {}
    pid_series = cohort["patient_nbr"]
    for name, arrays in parts.items():
        chosen = np.isin(pid_series.to_numpy(), np.concatenate(arrays))
        out[name] = np.flatnonzero(chosen & mask)
    return out


def temporal_split(cohort: pd.DataFrame, task: str,
                   fractions=(0.7, 0.1, 0.2)) -> dict[str, np.ndarray]:
    """
    Prospective (forward-in-time) split on the encounter axis.

    Encounters are cut by ``encounter_id`` quantiles: the model is fit on
    the earliest 70%, tuned on the next 10% and tested on the most recent
    20%.  A test encounter may belong to a patient whose earlier stays
    were in the training period, exactly as in deployment; its history
    is still built only from strictly earlier encounters.  For the
    escalation task the label depends on the *next* encounter, so an
    index encounter is only placed in a partition when its next
    encounter also falls in or before that partition's time window, and
    training/validation samples whose next encounter lies in a later
    window are dropped to avoid label leakage.
    """
    mask = task_mask(cohort, task)
    ids = cohort["encounter_id"].to_numpy()
    cut1, cut2 = np.quantile(ids, [fractions[0], fractions[0] + fractions[1]])
    period = np.where(ids < cut1, 0, np.where(ids < cut2, 1, 2))

    if task == "escalation":
        next_ids = cohort.groupby("patient_nbr")["encounter_id"].shift(-1)
        next_ids = next_ids.to_numpy()
        next_period = np.where(np.isnan(next_ids), -1,
                               np.where(next_ids < cut1, 0,
                                        np.where(next_ids < cut2, 1, 2)))
        mask = mask & (next_period == period)

    return {
        "train": np.flatnonzero(mask & (period == 0)),
        "val": np.flatnonzero(mask & (period == 1)),
        "test": np.flatnonzero(mask & (period == 2)),
    }


def history_index(cohort: pd.DataFrame, max_history: int) -> np.ndarray:
    """
    For every encounter row, the row indices of the same patient's
    previous ``max_history`` encounters (oldest first, left-padded with
    -1).  Only strictly earlier encounters are included, so an encounter
    can never see its own or any future outcome.
    """
    n = len(cohort)
    hist = np.full((n, max_history), -1, dtype=np.int64)
    order = cohort["order"].to_numpy()
    for lag in range(1, max_history + 1):
        valid = order >= lag
        rows = np.flatnonzero(valid)
        hist[rows, max_history - lag] = rows - lag
    return hist
