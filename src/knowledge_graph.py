"""
Hybrid clinical knowledge graph.

Two kinds of knowledge are fused into one graph over diagnosis concepts:

1. Ontology edges (``is_a``): the public ICD-9-CM hierarchy
   code -> subcategory -> category -> chapter -> root, and the
   anti-diabetic drug -> pharmacological class relation.
2. Empirical co-morbidity edges (``co_occurs``): positive pointwise
   mutual information (PPMI) between three-character ICD-9 categories
   that are recorded together in the same encounter.  These are counted
   on *training encounters only*, so the test partition never shapes the
   graph.

A node is kept only if it is observed at least ``min_count`` times in the
training encounters (chapters and the root are always kept).  A rare or
unseen code is therefore represented through its nearest retained
ancestors, which lets the model generalise to codes it never saw.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .data import DIAG_COLUMNS
from .icd9 import (
    CHAPTER_NAMES, DRUG_CLASSES, ROOT, ancestors, normalise_code,
)

MAX_ANCESTORS = 5


class ClinicalKnowledgeGraph:
    """Vocabulary, ancestor table and co-morbidity adjacency."""

    def __init__(self, min_count: int = 5, min_pair_count: int = 20,
                 top_k: int = 10):
        self.min_count = min_count
        self.min_pair_count = min_pair_count
        self.top_k = top_k

    # ── fitting ───────────────────────────────
    def fit(self, cohort: pd.DataFrame, train_rows: np.ndarray):
        diags = cohort[DIAG_COLUMNS].to_numpy()
        train_diags = diags[train_rows]

        node_counts: Counter = Counter()
        for row in train_diags:
            seen = set()
            for raw in row:
                code = normalise_code(raw)
                if code is None:
                    continue
                seen.update(ancestors(code))
            node_counts.update(seen)

        keep = [ROOT] + [f"L1:{c}" for c in sorted(CHAPTER_NAMES)]
        for node, count in sorted(node_counts.items()):
            if node in keep:
                continue
            if count >= self.min_count:
                keep.append(node)
        self.nodes = keep
        self.node_index = {n: i for i, n in enumerate(self.nodes)}
        self.node_counts = {n: int(node_counts.get(n, 0)) for n in self.nodes}

        # code table: index 0 is the padding / missing-diagnosis code
        all_codes = sorted({c for c in (normalise_code(v)
                                        for v in np.unique(diags.astype(str)))
                            if c is not None and c != "nan"})
        self.codes = ["<PAD>"] + all_codes
        self.code_index = {c: i for i, c in enumerate(self.codes)}
        anc = np.full((len(self.codes), MAX_ANCESTORS), -1, dtype=np.int64)
        flat = np.zeros(len(self.codes), dtype=np.int64)
        own = np.zeros(len(self.codes), dtype=bool)
        self.flat_vocab = ["<PAD>", "<UNK>"]
        for ci, code in enumerate(self.codes):
            if ci == 0:
                continue
            chain = [self.node_index[a] for a in ancestors(code)
                     if a in self.node_index]
            anc[ci, :len(chain)] = chain[:MAX_ANCESTORS]
            leaf = ancestors(code)[0]
            if leaf in self.node_index:
                own[ci] = True
                flat[ci] = len(self.flat_vocab)
                self.flat_vocab.append(code)
            else:
                flat[ci] = 1
        self.code_ancestors = anc
        self.code_flat = flat
        self.code_own_node = own

        self._fit_cooccurrence(train_diags)
        return self

    def _fit_cooccurrence(self, train_diags: np.ndarray):
        single: Counter = Counter()
        pairs: Counter = Counter()
        n_enc = 0
        for row in train_diags:
            cats = set()
            for raw in row:
                code = normalise_code(raw)
                if code is None:
                    continue
                node = f"L2:{code.split('.')[0]}"
                if node in self.node_index:
                    cats.add(node)
            if not cats:
                continue
            n_enc += 1
            single.update(cats)
            pairs.update(combinations(sorted(cats), 2))

        scored = []
        for (a, b), n_ab in pairs.items():
            if n_ab < self.min_pair_count:
                continue
            pmi = np.log(n_ab * n_enc / (single[a] * single[b]))
            if pmi > 0:
                scored.append((a, b, float(pmi), int(n_ab)))

        # keep the top-k strongest partners per node
        partners: dict[str, list] = {}
        for a, b, pmi, n_ab in scored:
            partners.setdefault(a, []).append((pmi, b, n_ab))
            partners.setdefault(b, []).append((pmi, a, n_ab))
        edges = {}
        for node, lst in partners.items():
            for pmi, other, n_ab in sorted(lst, reverse=True)[:self.top_k]:
                key = tuple(sorted((node, other)))
                edges[key] = (pmi, n_ab)
        self.cooc_edges = [(a, b, pmi, n) for (a, b), (pmi, n) in edges.items()]

        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        for a, b, pmi, _ in self.cooc_edges:
            i, j = self.node_index[a], self.node_index[b]
            adj[i, j] = adj[j, i] = pmi
        adj += np.eye(n, dtype=np.float32)
        deg = adj.sum(1)
        inv = 1.0 / np.sqrt(deg)
        self.cooc_adjacency = (adj * inv[:, None]) * inv[None, :]

    # ── lookups ───────────────────────────────
    def encode_diagnoses(self, cohort: pd.DataFrame) -> np.ndarray:
        """Integer code ids for diag_1..diag_3 (0 when missing)."""
        out = np.zeros((len(cohort), len(DIAG_COLUMNS)), dtype=np.int64)
        for j, col in enumerate(DIAG_COLUMNS):
            values = cohort[col].map(normalise_code)
            out[:, j] = values.map(
                lambda c: self.code_index.get(c, 0) if isinstance(c, str) else 0
            ).to_numpy()
        return out

    # ── export ────────────────────────────────
    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        for node in self.nodes:
            level = "root" if node == ROOT else {
                "L1": "chapter", "L2": "category", "L3": "subcategory",
                "L4": "code",
            }[node[:2]]
            label = node if node == ROOT else node[3:]
            if level == "chapter":
                label = f"{label} {CHAPTER_NAMES.get(label, '')}"
            graph.add_node(node, kind=level, label=label,
                           count=self.node_counts.get(node, 0))
        for node in self.nodes:
            if node == ROOT:
                continue
            if node.startswith("L1:"):
                parent = ROOT
            else:
                chain = ancestors(node[3:])
                pos = chain.index(node) if node in chain else 0
                parent = next((a for a in chain[pos + 1:]
                               if a in self.node_index), ROOT)
            graph.add_edge(node, parent, relation="is_a",
                           source="ICD-9-CM hierarchy")
        for a, b, pmi, n_ab in self.cooc_edges:
            graph.add_edge(a, b, relation="co_occurs", ppmi=round(pmi, 4),
                           count=n_ab, source="training encounters")
        for drug, classes in DRUG_CLASSES.items():
            graph.add_node(f"DRUG:{drug}", kind="drug", label=drug, count=0)
            for cls in classes:
                graph.add_node(f"CLASS:{cls}", kind="drug_class", label=cls,
                               count=0)
                graph.add_edge(f"DRUG:{drug}", f"CLASS:{cls}",
                               relation="is_a", source="pharmacology")
        return graph

    def summary(self) -> dict:
        levels = Counter(n[:2] for n in self.nodes if n != ROOT)
        top = sorted(self.cooc_edges, key=lambda e: -e[2])
        top = [e for e in top if e[3] >= 50][:40]
        return {
            "n_nodes": len(self.nodes),
            "n_codes": len(self.codes) - 1,
            "nodes_by_level": {
                "chapter": levels.get("L1", 0),
                "category": levels.get("L2", 0),
                "subcategory": levels.get("L3", 0),
                "code": levels.get("L4", 0),
            },
            "n_cooccurrence_edges": len(self.cooc_edges),
            "top_cooccurrence": [
                {"a": a[3:], "b": b[3:], "ppmi": round(p, 3), "count": n}
                for a, b, p, n in top
            ],
            "chapters": [
                {"id": c, "name": CHAPTER_NAMES[c],
                 "count": self.node_counts.get(f"L1:{c}", 0)}
                for c in sorted(CHAPTER_NAMES)
            ],
        }

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.to_networkx(), directory / "knowledge_graph.graphml")
        (directory / "knowledge_graph_summary.json").write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8")
