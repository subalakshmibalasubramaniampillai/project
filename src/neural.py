"""
Neural sequence models over a patient's encounter history.

Proposed model
    TKGN - Temporal Knowledge-Gated Network.  Three ideas are combined:

    (a) Knowledge-grounded code representation.  A diagnosis code is
        represented by attention over its retained ancestors in the
        ICD-9-CM hierarchy, and the category-level embeddings are first
        refined by one propagation step over an empirical co-morbidity
        graph (PPMI edges learned from training encounters).  The
        ontology part follows the idea of GRAM (Choi et al., KDD 2017);
        adding the data-driven co-morbidity propagation before the
        ancestor attention is specific to this work.
    (b) History-reliability gating.  Real-world inpatient records are
        short and irregular: most patients have no earlier stay and a
        few have many.  Instead of always adding a history summary, TKGN
        computes a feature-wise gate from the current encounter, the
        history summary, the change since the last stay and an embedding
        of the history length, so the network decides per patient and
        per dimension how much the past should move the prediction.
    (c) Progression (delta) encoding.  The difference between the index
        encounter and the most recent previous one is modelled as its
        own, separately gated signal, which captures escalation of
        treatment, utilisation or diagnosis burden between stays.

Baselines implemented with the same encounter encoder
    * GRU over the encounter sequence (Doctor-AI style recurrent model).
    * RETAIN (Choi et al., NeurIPS 2016): reverse-time two-level attention.
    * Transformer encoder over encounters (BEHRT-style, small).

All models use only the index encounter and strictly earlier encounters.
The outcome token of the index encounter is always "unknown".
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .features import UNKNOWN_OUTCOME


# ──────────────────────────────────────────────
# Diagnosis code encoders
# ──────────────────────────────────────────────

class KnowledgeCodeEncoder(nn.Module):
    """
    Ontology-attention + co-morbidity propagation code embeddings.

    A code keeps its own embedding (when it is frequent enough to have a
    node) and receives an additive, attention-weighted mixture of its
    proper ancestors.  The co-morbidity propagation is zero-initialised,
    so the graph only changes the representation when training finds it
    useful; frequent codes stay specific while rare codes inherit
    information from their ancestors and co-morbid neighbours.
    """

    def __init__(self, n_nodes: int, code_ancestors: np.ndarray,
                 cooc_adjacency: np.ndarray | None, dim: int,
                 own_node: np.ndarray | None = None):
        super().__init__()
        self.node_emb = nn.Embedding(n_nodes, dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)
        anc = torch.as_tensor(code_ancestors)
        if own_node is None:
            own_node = np.zeros(len(code_ancestors), dtype=bool)
        own = torch.as_tensor(own_node, dtype=torch.bool)
        # the first ancestor is the code's own node when it was retained
        self.register_buffer("leaf", torch.where(own, anc[:, 0],
                                                 torch.full_like(anc[:, 0], -1)))
        proper = anc.clone()
        proper[own, 0] = -1
        self.register_buffer("anc", proper)
        if cooc_adjacency is not None:
            self.register_buffer("adj", torch.as_tensor(cooc_adjacency))
            self.cooc = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.cooc.weight)
        else:
            self.adj = None
        self.att = nn.Sequential(nn.Linear(2 * dim, dim), nn.Tanh(),
                                 nn.Linear(dim, 1))

    def node_table(self) -> torch.Tensor:
        table = self.node_emb.weight
        if self.adj is not None:
            table = table + torch.tanh(self.adj @ self.cooc(table))
        return table

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (code table [C, d], ancestor attention [C, A])."""
        table = self.node_table()
        has_leaf = (self.leaf >= 0).unsqueeze(-1)
        leaf = torch.where(has_leaf, table[self.leaf.clamp(min=0)],
                           torch.zeros_like(table[:1]).expand(len(self.leaf), -1))
        valid = self.anc >= 0
        emb = table[self.anc.clamp(min=0)]                  # [C, A, d]
        query = leaf.unsqueeze(1).expand_as(emb)
        score = self.att(torch.cat([query, emb], -1)).squeeze(-1)
        score = score.masked_fill(~valid, -1e9)
        alpha = torch.softmax(score, dim=-1) * valid
        codes = leaf + (alpha.unsqueeze(-1) * emb).sum(1)
        codes = torch.cat([torch.zeros_like(codes[:1]), codes[1:]], 0)
        return codes, alpha


class FlatCodeEncoder(nn.Module):
    """One embedding per frequent code, a shared one for rare codes."""

    def __init__(self, code_flat: np.ndarray, n_flat: int, dim: int):
        super().__init__()
        self.register_buffer("flat", torch.as_tensor(code_flat))
        self.emb = nn.Embedding(n_flat, dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self):
        return self.emb(self.flat), None


# ──────────────────────────────────────────────
# Encounter encoder (shared by all neural models)
# ──────────────────────────────────────────────

class EncounterEncoder(nn.Module):
    def __init__(self, code_encoder: nn.Module, drug_class_matrix: np.ndarray,
                 cat_cardinalities: list[int], num_bin_counts: list[int],
                 dim: int, dropout: float, field_dim: int = 16,
                 diag_pool: str = "attention"):
        super().__init__()
        self.code_encoder = code_encoder
        self.diag_pool = diag_pool
        if diag_pool == "concat":
            self.diag_proj = nn.Linear(3 * dim, dim)
        n_drugs, n_classes = drug_class_matrix.shape
        self.diag_pos = nn.Parameter(torch.zeros(3, dim))
        self.diag_att = nn.Linear(dim, 1)
        self.drug_emb = nn.Embedding(n_drugs, dim)
        self.class_emb = nn.Linear(n_classes, dim, bias=False)
        self.register_buffer("drug_classes", torch.as_tensor(drug_class_matrix))
        self.state_emb = nn.Embedding(4, dim)
        self.no_drug = nn.Parameter(torch.zeros(dim))
        self.cat_embs = nn.ModuleList([nn.Embedding(n, field_dim, padding_idx=0)
                                       for n in cat_cardinalities])
        self.cat_proj = nn.Linear(len(cat_cardinalities) * field_dim, dim)
        # numeric fields: quantile-bin embeddings plus the scaled value
        self.bin_embs = nn.ModuleList([nn.Embedding(n, field_dim, padding_idx=0)
                                       for n in num_bin_counts])
        n_numeric = len(num_bin_counts)
        self.num_proj = nn.Linear(n_numeric * (field_dim + 1), dim)
        self.outcome_emb = nn.Embedding(4, dim)
        self.mix = nn.Sequential(
            nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, diag, drugs, cats, nums, num_bins, outcome):
        codes, _ = self.code_encoder()
        dx = codes[diag] + self.diag_pos                     # [B,T,3,d]
        dmask = diag > 0
        if self.diag_pool == "concat":
            dx = self.diag_proj((dx * dmask.unsqueeze(-1)).flatten(-2))
        else:
            dscore = self.diag_att(dx).squeeze(-1).masked_fill(~dmask, -1e9)
            dalpha = torch.softmax(dscore, -1) * dmask
            dx = (dalpha.unsqueeze(-1) * dx).sum(-2)

        drug_vec = (self.drug_emb.weight
                    + self.class_emb(self.drug_classes))   # [D, d]
        tokens = drug_vec + self.state_emb(drugs)          # [B,T,D,d]
        active = (drugs > 0).float().unsqueeze(-1)
        n_active = active.sum(-2)
        med = (tokens * active).sum(-2) / n_active.clamp(min=1)
        med = torch.where(n_active > 0, med, self.no_drug.expand_as(med))

        cat = self.cat_proj(torch.cat(
            [emb(cats[..., i]) for i, emb in enumerate(self.cat_embs)], -1))
        num = self.num_proj(torch.cat(
            [emb(num_bins[..., i]) for i, emb in enumerate(self.bin_embs)]
            + [nums], -1))
        out = self.outcome_emb(outcome)
        return self.mix(torch.cat([dx, med, cat, num, out], -1))


# ──────────────────────────────────────────────
# Proposed model
# ──────────────────────────────────────────────

class TKGN(nn.Module):
    """Temporal Knowledge-Gated Network."""

    def __init__(self, encoder: EncounterEncoder, dim: int, max_history: int,
                 dropout: float, use_history=True, use_gate=True,
                 use_delta=True):
        super().__init__()
        self.encoder = encoder
        self.use_history = use_history
        self.use_gate = use_gate
        self.use_delta = use_delta
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.recency = nn.Embedding(max_history + 1, dim)
        self.length_emb = nn.Embedding(max_history + 1, dim)
        self.null_history = nn.Parameter(torch.zeros(dim))
        self.hist_proj = nn.Linear(2 * dim, dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.gate_hist = nn.Linear(4 * dim, dim)
        self.gate_delta = nn.Linear(4 * dim, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(dim, 1))
        # auxiliary head: 3-level readmission outcome of the index stay
        self.aux_head = nn.Linear(dim, 3)
        self.max_history = max_history

    def _gru_summary(self, hist, mask):
        """Last GRU state over the valid (left-padded) history."""
        B, H, d = hist.shape
        n = mask.sum(1)
        shift = (H - n).unsqueeze(1)
        idx = (torch.arange(H, device=hist.device).unsqueeze(0) + shift) % H
        right = torch.gather(hist, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        packed = nn.utils.rnn.pack_padded_sequence(
            right, n.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.gru(packed)
        return last[0]

    def forward(self, batch):
        enc = self.encoder(batch["diag"], batch["drugs"], batch["cats"],
                           batch["nums"], batch["num_bins"], batch["outcome"])
        current = enc[:, -1]
        extras = {}
        if not self.use_history:
            extras["aux_logits"] = self.aux_head(current)
            return self.head(current).squeeze(-1), extras

        hist = enc[:, :-1]
        mask = batch["mask"][:, :-1]
        n = mask.sum(1)
        has = (n > 0).unsqueeze(-1)
        H = hist.shape[1]

        gru_last = self._gru_summary(hist, mask)
        recency = self.recency(torch.arange(H, 0, -1, device=hist.device))
        keys = self.key(hist + recency)
        score = (keys @ self.query(current).unsqueeze(-1)).squeeze(-1)
        score = score / math.sqrt(hist.shape[-1])
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, -1) * mask
        attended = (alpha.unsqueeze(-1) * hist).sum(1)
        summary = self.hist_proj(torch.cat([gru_last, attended], -1))
        summary = torch.where(has, summary, self.null_history.expand_as(summary))

        delta = torch.where(has, current - hist[:, -1], torch.zeros_like(current))
        length = self.length_emb(n.clamp(max=self.max_history))
        context = torch.cat([current, summary, delta, length], -1)
        if self.use_gate:
            g_hist = torch.sigmoid(self.gate_hist(context))
            g_delta = torch.sigmoid(self.gate_delta(context))
        else:
            g_hist = torch.ones_like(current)
            g_delta = torch.ones_like(current)
        fused = current + g_hist * summary
        if self.use_delta:
            fused = fused + g_delta * self.delta_proj(delta)
        extras = {"gate_history": g_hist.mean(-1), "gate_delta": g_delta.mean(-1),
                  "attention": alpha, "aux_logits": self.aux_head(fused)}
        return self.head(fused).squeeze(-1), extras


# ──────────────────────────────────────────────
# Sequence baselines
# ──────────────────────────────────────────────

class GRUSequence(nn.Module):
    def __init__(self, encoder, dim, max_history, dropout):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, batch):
        enc = self.encoder(batch["diag"], batch["drugs"], batch["cats"],
                           batch["nums"], batch["num_bins"], batch["outcome"])
        mask = batch["mask"]
        n = mask.sum(1)
        T, d = enc.shape[1], enc.shape[2]
        idx = (torch.arange(T, device=enc.device).unsqueeze(0)
               + (T - n).unsqueeze(1)) % T
        right = torch.gather(enc, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        packed = nn.utils.rnn.pack_padded_sequence(
            right, n.cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.gru(packed)
        return self.head(last[0]).squeeze(-1), {}


class RETAIN(nn.Module):
    """RETAIN: reverse-time attention over visits (Choi et al., 2016)."""

    def __init__(self, encoder, dim, max_history, dropout):
        super().__init__()
        self.encoder = encoder
        self.rnn_alpha = nn.GRU(dim, dim, batch_first=True)
        self.rnn_beta = nn.GRU(dim, dim, batch_first=True)
        self.w_alpha = nn.Linear(dim, 1)
        self.w_beta = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, batch):
        v = self.encoder(batch["diag"], batch["drugs"], batch["cats"],
                         batch["nums"], batch["num_bins"], batch["outcome"])
        mask = batch["mask"]
        rev = torch.flip(v, [1])              # most recent first
        rmask = torch.flip(mask, [1])
        g, _ = self.rnn_alpha(rev)
        h, _ = self.rnn_beta(rev)
        e = self.w_alpha(g).squeeze(-1).masked_fill(~rmask, -1e9)
        alpha = torch.softmax(e, -1) * rmask
        beta = torch.tanh(self.w_beta(h))
        context = (alpha.unsqueeze(-1) * beta * rev).sum(1)
        return self.out(context).squeeze(-1), {"attention": alpha}


class TransformerSequence(nn.Module):
    def __init__(self, encoder, dim, max_history, dropout, heads=4, layers=2):
        super().__init__()
        self.encoder = encoder
        self.position = nn.Embedding(max_history + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers,
                                                 enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, batch):
        v = self.encoder(batch["diag"], batch["drugs"], batch["cats"],
                         batch["nums"], batch["num_bins"], batch["outcome"])
        T = v.shape[1]
        v = v + self.position(torch.arange(T, device=v.device))
        h = self.transformer(v, src_key_padding_mask=~batch["mask"])
        return self.head(h[:, -1]).squeeze(-1), {}


# ──────────────────────────────────────────────
# Model zoo
# ──────────────────────────────────────────────

NEURAL_VARIANTS = {
    # name: (architecture, knowledge graph, co-occurrence, history,
    #        gate, delta, prior outcomes)
    "tkgn": ("tkgn", True, True, True, True, True, True),
    "tkgn_no_kg": ("tkgn", False, False, True, True, True, True),
    "tkgn_no_cooccurrence": ("tkgn", True, False, True, True, True, True),
    "tkgn_no_history": ("tkgn", True, True, False, False, False, True),
    "tkgn_no_gate": ("tkgn", True, True, True, False, True, True),
    "tkgn_no_delta": ("tkgn", True, True, True, True, False, True),
    "tkgn_no_prior_outcomes": ("tkgn", True, True, True, True, True, False),
    "gru": ("gru", False, False, True, False, False, True),
    "retain": ("retain", False, False, True, False, False, True),
    "transformer": ("transformer", False, False, True, False, False, True),
}


def build_model(name, kg, seq_encoder, dim=64, max_history=10, dropout=0.2,
                diag_pool="attention"):
    arch, use_kg, use_cooc, use_hist, use_gate, use_delta, _ = NEURAL_VARIANTS[name]
    if use_kg:
        code_encoder = KnowledgeCodeEncoder(
            len(kg.nodes), kg.code_ancestors,
            kg.cooc_adjacency if use_cooc else None, dim,
            own_node=kg.code_own_node)
    else:
        code_encoder = FlatCodeEncoder(kg.code_flat, len(kg.flat_vocab), dim)
    encoder = EncounterEncoder(code_encoder, seq_encoder.drug_class_matrix,
                               seq_encoder.cat_cardinalities,
                               seq_encoder.num_bin_counts, dim, dropout,
                               diag_pool=diag_pool)
    if arch == "tkgn":
        return TKGN(encoder, dim, max_history, dropout, use_history=use_hist,
                    use_gate=use_gate, use_delta=use_delta)
    cls = {"gru": GRUSequence, "retain": RETAIN,
           "transformer": TransformerSequence}[arch]
    return cls(encoder, dim, max_history, dropout)


def uses_prior_outcomes(name: str) -> bool:
    return NEURAL_VARIANTS[name][6]


# ──────────────────────────────────────────────
# Batching and training
# ──────────────────────────────────────────────

class SequenceData:
    """Holds encounter arrays and gathers history windows per batch."""

    def __init__(self, arrays: dict, history: np.ndarray, labels: np.ndarray,
                 prior_outcomes: bool = True, offset: np.ndarray | None = None):
        n = len(labels)
        self.offset = (None if offset is None
                       else torch.as_tensor(offset, dtype=torch.float32))
        self.aux_labels = torch.as_tensor(arrays["outcome"].copy())
        self.arrays = {}
        for key, value in arrays.items():
            pad = np.zeros((1,) + value.shape[1:], dtype=value.dtype)
            self.arrays[key] = torch.as_tensor(np.concatenate([value, pad]))
        if not prior_outcomes:
            self.arrays["outcome"][:] = UNKNOWN_OUTCOME
        self.pad_row = n
        self.history = torch.as_tensor(history)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def batch(self, rows: np.ndarray) -> dict:
        rows_t = torch.as_tensor(rows)
        window = torch.cat([self.history[rows_t], rows_t.unsqueeze(1)], 1)
        mask = window >= 0
        gather = torch.where(mask, window, torch.full_like(window, self.pad_row))
        out = {key: value[gather] for key, value in self.arrays.items()}
        outcome = out["outcome"].clone()
        outcome[:, -1] = UNKNOWN_OUTCOME   # index outcome is never visible
        outcome[~mask] = 0
        out["outcome"] = outcome
        out["mask"] = mask
        out["y"] = self.labels[rows_t]
        out["aux_y"] = self.aux_labels[rows_t]
        if self.offset is not None:
            out["offset"] = self.offset[rows_t]
        return out


def predict(model, data: SequenceData, rows: np.ndarray, batch_size=2048,
            with_extras=False):
    model.eval()
    probs, gates = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = data.batch(rows[start:start + batch_size])
            logits, extras = model(batch)
            if "offset" in batch:
                logits = logits + batch["offset"]
            probs.append(torch.sigmoid(logits).numpy())
            if with_extras and "gate_history" in extras:
                gates.append(np.stack([extras["gate_history"].numpy(),
                                       extras["gate_delta"].numpy()], 1))
    probs = np.concatenate(probs)
    if with_extras:
        return probs, (np.concatenate(gates) if gates else None)
    return probs


def train_model(model, data: SequenceData, train_rows, val_rows, seed,
                max_epochs=30, patience=4, batch_size=512, lr=1e-3,
                weight_decay=1e-4, aux_weight=0.0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    y_val = data.labels[val_rows].numpy()
    best, best_state, wait, history = -1.0, None, 0, []
    started = time.time()
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_rows)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = data.batch(order[start:start + batch_size])
            logits, extras = model(batch)
            if "offset" in batch:
                logits = logits + batch["offset"]
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            if aux_weight > 0 and "aux_logits" in extras:
                loss = loss + aux_weight * F.cross_entropy(
                    extras["aux_logits"], batch["aux_y"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch["y"])
        val_auc = roc_auc_score(y_val, predict(model, data, val_rows))
        history.append({"epoch": epoch + 1, "train_loss": total / len(order),
                        "val_auroc": float(val_auc)})
        if verbose:
            print(f"    epoch {epoch + 1:2d} loss={total / len(order):.4f} "
                  f"val_auroc={val_auc:.4f} ({time.time() - started:.0f}s)")
        if val_auc > best + 1e-4:
            best, wait = val_auc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_auroc": float(best), "epochs": history,
            "train_seconds": time.time() - started}
