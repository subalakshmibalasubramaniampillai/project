"""
Feature construction.

Two views of the same encounters are produced:

* ``SequenceEncoder`` -> integer / float arrays per encounter, consumed
  by the neural sequence models (history is assembled at batch time).
* ``tabular_features`` -> one flat row per encounter for classical
  learners, including hand-crafted history aggregates so that these
  baselines see the same longitudinal information as the neural models.

Every statistic (vocabularies, scaling constants, frequent-category
lists) is fitted on training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_COLUMNS, DIAG_COLUMNS, NUMERIC_COLUMNS
from .icd9 import (
    CHAPTER_NAMES, DRUG_CLASSES, DRUG_COLUMNS, DRUG_STATES, chapter_of,
    normalise_code,
)

UNKNOWN_OUTCOME = 3  # outcome token for the index encounter (not yet known)


def _cat_values(cohort: pd.DataFrame, col: str) -> pd.Series:
    return cohort[col].astype(str).where(cohort[col].notna(), "<MISSING>")


class SequenceEncoder:
    """Encounter-level arrays for the neural models."""

    def __init__(self, min_count: int = 20):
        self.min_count = min_count

    def fit(self, cohort: pd.DataFrame, train_rows: np.ndarray):
        self.cat_vocab = {}
        for col in CATEGORICAL_COLUMNS:
            counts = _cat_values(cohort.iloc[train_rows], col).value_counts()
            kept = sorted(counts[counts >= self.min_count].index)
            # 0 = padding, 1 = other/rare
            self.cat_vocab[col] = {v: i + 2 for i, v in enumerate(kept)}
        logged = np.log1p(cohort.iloc[train_rows][NUMERIC_COLUMNS]
                          .to_numpy(dtype=float))
        self.num_mean = logged.mean(0)
        self.num_std = logged.std(0) + 1e-6
        raw = cohort.iloc[train_rows][NUMERIC_COLUMNS].to_numpy(dtype=float)
        self.num_edges = [np.unique(np.quantile(raw[:, j], np.linspace(0, 1, 17)[1:-1]))
                          for j in range(raw.shape[1])]
        classes = sorted({c for cls in DRUG_CLASSES.values() for c in cls})
        self.drug_classes = classes
        self.drug_class_matrix = np.zeros((len(DRUG_COLUMNS), len(classes)),
                                          dtype=np.float32)
        for i, drug in enumerate(DRUG_COLUMNS):
            for cls in DRUG_CLASSES[drug]:
                self.drug_class_matrix[i, classes.index(cls)] = 1.0
        self.drug_class_matrix /= self.drug_class_matrix.sum(1, keepdims=True)
        return self

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(v) + 2 for v in self.cat_vocab.values()]

    @property
    def num_bin_counts(self) -> list[int]:
        # +1 for values above the last edge, +1 for padding (index 0)
        return [len(e) + 2 for e in self.num_edges]

    def transform(self, cohort: pd.DataFrame, diag_codes: np.ndarray) -> dict:
        state_index = {s: i for i, s in enumerate(DRUG_STATES)}
        drugs = np.stack([cohort[d].map(state_index).fillna(0).to_numpy()
                          for d in DRUG_COLUMNS], axis=1).astype(np.int64)
        cats = np.stack([
            _cat_values(cohort, col).map(vocab).fillna(1).to_numpy()
            for col, vocab in self.cat_vocab.items()
        ], axis=1).astype(np.int64)
        raw = cohort[NUMERIC_COLUMNS].to_numpy(dtype=float)
        nums = (np.log1p(raw) - self.num_mean) / self.num_std
        bins = np.stack([np.searchsorted(e, raw[:, j], side="right") + 1
                         for j, e in enumerate(self.num_edges)], 1)
        return {
            "diag": diag_codes.astype(np.int64),
            "drugs": drugs,
            "cats": cats,
            "nums": nums.astype(np.float32),
            "num_bins": bins.astype(np.int64),
            "outcome": cohort["outcome_level"].to_numpy().astype(np.int64),
        }


# ──────────────────────────────────────────────
# Tabular view for classical models
# ──────────────────────────────────────────────

def _history_aggregates(cohort: pd.DataFrame) -> pd.DataFrame:
    """Aggregates over each patient's strictly earlier encounters."""
    g = cohort.groupby("patient_nbr")
    out = pd.DataFrame(index=cohort.index)
    out["hist_n_prior"] = cohort["order"]

    def prior_sum(series: pd.Series) -> pd.Series:
        return series.groupby(cohort["patient_nbr"]).cumsum() - series

    readmit30 = (cohort["outcome_level"] == 2).astype(float)
    readmit_any = (cohort["outcome_level"] >= 1).astype(float)
    out["hist_prior_readmit30"] = prior_sum(readmit30)
    out["hist_prior_readmit_any"] = prior_sum(readmit_any)
    has_prior = cohort["order"] > 0
    for col in ["time_in_hospital", "num_medications", "number_diagnoses",
                "num_lab_procedures", "n_active_drugs"]:
        mean = prior_sum(cohort[col].astype(float)) / cohort["order"].clip(lower=1)
        out[f"hist_mean_{col}"] = mean.where(has_prior, 0.0)
        prev = g[col].shift(1)
        out[f"delta_{col}"] = (cohort[col] - prev).fillna(0.0)
    out["hist_prior_insulin"] = prior_sum(cohort["on_insulin"].astype(float))
    prev_outcome = g["outcome_level"].shift(1)
    for level, name in [(0, "no"), (1, "gt30"), (2, "lt30")]:
        out[f"hist_last_outcome_{name}"] = (prev_outcome == level).astype(float)
    return out


class TabularEncoder:
    """One-hot / multi-hot design matrix fitted on training rows."""

    def __init__(self, min_count: int = 20, min_diag_count: int = 50):
        self.min_count = min_count
        self.min_diag_count = min_diag_count

    def fit(self, cohort: pd.DataFrame, train_rows: np.ndarray):
        train = cohort.iloc[train_rows]
        self.cat_levels = {}
        for col in CATEGORICAL_COLUMNS:
            counts = _cat_values(train, col).value_counts()
            self.cat_levels[col] = sorted(counts[counts >= self.min_count].index)
        cats = pd.concat([train[c] for c in DIAG_COLUMNS]).map(normalise_code)
        cats = cats.dropna().map(lambda c: c.split(".")[0])
        counts = cats.value_counts()
        self.diag_categories = sorted(counts[counts >= self.min_diag_count].index)
        return self

    def transform(self, cohort: pd.DataFrame) -> pd.DataFrame:
        blocks = [cohort[NUMERIC_COLUMNS].astype(float)]
        for col, levels in self.cat_levels.items():
            values = _cat_values(cohort, col)
            block = pd.DataFrame(
                {f"{col}={lvl}": (values == lvl).astype(np.float32)
                 for lvl in levels}, index=cohort.index)
            blocks.append(block)
        for drug in DRUG_COLUMNS:
            for state in ("Steady", "Up", "Down"):
                blocks.append(pd.Series((cohort[drug] == state).astype(np.float32),
                                        name=f"{drug}={state}"))

        codes = [cohort[c].map(normalise_code) for c in DIAG_COLUMNS]
        category_hot = {f"dx={c}": np.zeros(len(cohort), dtype=np.float32)
                        for c in self.diag_categories}
        chapter_ids = sorted(CHAPTER_NAMES)
        chapter_hot = {f"chapter={c}": np.zeros(len(cohort), np.float32)
                       for c in chapter_ids}
        primary_chapter = {f"primary_chapter={c}":
                           np.zeros(len(cohort), np.float32)
                           for c in chapter_ids}
        for pos, series in enumerate(codes):
            for i, code in enumerate(series.to_numpy()):
                if not isinstance(code, str):
                    continue
                cat = code.split(".")[0]
                key = f"dx={cat}"
                if key in category_hot:
                    category_hot[key][i] = 1.0
                chap = chapter_of(code)
                if f"chapter={chap}" not in chapter_hot:
                    continue
                chapter_hot[f"chapter={chap}"][i] = 1.0
                if pos == 0:
                    primary_chapter[f"primary_chapter={chap}"][i] = 1.0
        blocks.append(pd.DataFrame(category_hot, index=cohort.index))
        blocks.append(pd.DataFrame(chapter_hot, index=cohort.index))
        blocks.append(pd.DataFrame(primary_chapter, index=cohort.index))
        blocks.append(_history_aggregates(cohort))
        frame = pd.concat(blocks, axis=1)
        if not hasattr(self, "columns"):
            self.columns = list(frame.columns)
        return frame.reindex(columns=self.columns, fill_value=0.0).astype(np.float32)
