"""
Data-integrity and leakage tests.

Run with:  python -m pytest -q
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data import (
    RAW_FILE, RAW_SHA256, _sha256, build_cohort, grouped_split,
    history_index, task_labels, temporal_split,
)
from src.evaluation import delong_test, expected_calibration_error
from src.features import UNKNOWN_OUTCOME
from src.icd9 import ancestors, chapter_of


@pytest.fixture(scope="module")
def cohort():
    return build_cohort()


def test_raw_file_is_the_uci_release():
    assert _sha256(RAW_FILE) == RAW_SHA256


def test_cohort_is_real_and_filtered(cohort):
    assert len(cohort) == 99_340
    assert cohort["patient_nbr"].nunique() == 69_987
    assert not cohort["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).any()
    assert cohort["encounter_id"].is_unique


@pytest.mark.parametrize("task", ["readmit30", "escalation"])
def test_grouped_split_is_patient_disjoint(cohort, task):
    split = grouped_split(cohort, task, seed=1000)
    pids = {k: set(cohort["patient_nbr"].to_numpy()[v]) for k, v in split.items()}
    assert not pids["train"] & pids["val"]
    assert not pids["train"] & pids["test"]
    assert not pids["val"] & pids["test"]
    labels = task_labels(cohort, task)
    for rows in split.values():
        assert set(np.unique(labels[rows])) <= {0, 1}


@pytest.mark.parametrize("task", ["readmit30", "escalation"])
def test_temporal_split_is_forward_in_time(cohort, task):
    split = temporal_split(cohort, task)
    ids = cohort["encounter_id"].to_numpy()
    assert ids[split["train"]].max() < ids[split["val"]].min()
    assert ids[split["val"]].max() < ids[split["test"]].min()


def test_escalation_temporal_labels_do_not_cross_periods(cohort):
    split = temporal_split(cohort, "escalation")
    nxt = cohort.groupby("patient_nbr")["encounter_id"].shift(-1).to_numpy()
    ids = cohort["encounter_id"].to_numpy()
    val_start = ids[split["val"]].min()
    assert (nxt[split["train"]] < val_start).all()


def test_history_contains_only_earlier_stays_of_same_patient(cohort):
    hist = history_index(cohort, 10)
    pid = cohort["patient_nbr"].to_numpy()
    enc = cohort["encounter_id"].to_numpy()
    rows, cols = np.nonzero(hist >= 0)
    prev = hist[rows, cols]
    assert (pid[prev] == pid[rows]).all()
    assert (enc[prev] < enc[rows]).all()
    first = cohort["order"].to_numpy() == 0
    assert (hist[first] == -1).all()


def test_index_outcome_is_never_visible(cohort):
    from src.knowledge_graph import ClinicalKnowledgeGraph
    from src.features import SequenceEncoder
    from src.neural import SequenceData

    split = grouped_split(cohort, "readmit30", seed=1000)
    kg = ClinicalKnowledgeGraph().fit(cohort, split["train"])
    seq = SequenceEncoder().fit(cohort, split["train"])
    arrays = seq.transform(cohort, kg.encode_diagnoses(cohort))
    data = SequenceData(arrays, history_index(cohort, 10),
                        task_labels(cohort, "readmit30").astype(np.float32))
    batch = data.batch(split["test"][:512])
    assert (batch["outcome"][:, -1] == UNKNOWN_OUTCOME).all()
    assert batch["mask"][:, -1].all()


def test_knowledge_graph_uses_training_rows_only(cohort):
    from src.knowledge_graph import ClinicalKnowledgeGraph
    split = grouped_split(cohort, "readmit30", seed=1000)
    small = split["train"][:2000]
    kg = ClinicalKnowledgeGraph().fit(cohort, small)
    n_chapter = sum(n.startswith("L1:") for n in kg.nodes)
    assert kg.node_counts["L1:C07"] <= len(small)
    assert n_chapter == 19


def test_icd9_hierarchy():
    assert ancestors("250.83") == ["L4:250.83", "L3:250.8", "L2:250", "L1:C03", "ROOT"]
    assert chapter_of("V58") == "CV"
    assert chapter_of("E888") == "CE"
    assert chapter_of("428") == "C07"


def test_delong_matches_auroc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    a = y + rng.normal(size=2000)
    b = rng.normal(size=2000)
    res = delong_test(y, a, b)
    assert abs(res["auroc_a"] - roc_auc_score(y, a)) < 1e-9
    assert res["p_value"] < 1e-6


def test_ece_of_calibrated_predictions_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=50_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
