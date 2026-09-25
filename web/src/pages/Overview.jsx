import React from "react";
import { useFetch } from "../useFetch";
import { getMetrics, getSplit } from "../api";

const METRIC_COLS = [
  { key: "model", label: "Model" },
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "pr_auc", label: "PR-AUC" },
  { key: "brier", label: "Brier" },
  { key: "log_loss", label: "Log loss" },
];

const NOVEL_MODELS = new Set(["tagnn", "mhfin", "ccf_fusion"]);

function fmt(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(3);
}

export default function Overview() {
  const { data: metrics, loading: mLoading, error: mError } = useFetch(getMetrics);
  const { data: split, loading: sLoading } = useFetch(getSplit);

  if (mLoading || sLoading) return <div className="loading">Loading…</div>;
  if (mError) return <div className="alert error">{mError}</div>;

  const parseSplit =
    typeof split === "string" ? JSON.parse(split) : split || {};
  const bestF1 = Math.max(...metrics.map((m) => m.f1));
  // best model defaults: prefer a novel model when it ties with a baseline
  const bestModel = metrics
    .filter((m) => m.f1 >= bestF1 - 1e-9)
    .sort((a, b) => (NOVEL_MODELS.has(b.model) ? 1 : 0) - (NOVEL_MODELS.has(a.model) ? 1 : 0))
    .pop();

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Model comparison</h2>
        <p className="page-sub">
          {parseSplit.n_patients
            ? `${parseSplit.n_patients} patients · ${parseSplit.n_visits} visits · ${parseSplit.data_source} data`
            : "Patient-level test split metrics"}{" "}
          · {parseSplit.train_patients} train / {parseSplit.test_patients} test
        </p>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="label">Best model</div>
          <div className="value accent">{bestModel?.model}</div>
        </div>
        <div className="stat">
          <div className="label">Best F1</div>
          <div className="value">{fmt(bestF1)}</div>
        </div>
        <div className="stat">
          <div className="label">Models evaluated</div>
          <div className="value">{metrics.length}</div>
        </div>
        <div className="stat">
          <div className="label">Best ROC-AUC</div>
          <div className="value">{fmt(Math.max(...metrics.map((m) => m.roc_auc)))}</div>
        </div>
      </div>

      <div className="card">
        <h3>All models</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {METRIC_COLS.map((c) => (
                  <th key={c.key}>{c.label}</th>
                ))}
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => (
                <tr key={m.model}>
                  {METRIC_COLS.map((c) => (
                    <td
                      key={c.key}
                      className={c.key === "model" ? "" : "num"}
                    >
                      {c.key === "model" ? (
                        <span className="pill-wrap">
                          {m.model}
                          {m.model === bestModel?.model && (
                            <span className="pill best" style={{ marginLeft: 8 }}>
                              best
                            </span>
                          )}
                          {NOVEL_MODELS.has(m.model) && (
                            <span className="pill novel" style={{ marginLeft: 6 }}>
                              novel
                            </span>
                          )}
                        </span>
                      ) : (
                        fmt(m[c.key])
                      )}
                    </td>
                  ))}
                  <td>
                    <span
                      className={`pill ${NOVEL_MODELS.has(m.model) ? "novel" : "baseline"}`}
                    >
                      {NOVEL_MODELS.has(m.model) ? "novel" : "baseline / graph"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="alert info">
        Metrics are computed on the held-out patient test split. They are
        research results, not clinical validation.
      </div>
    </div>
  );
}