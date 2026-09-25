import React from "react";
import { useFetch } from "../useFetch";
import { getFusion } from "../api";

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(4);
}

export default function Fusion() {
  const { data, loading, error } = useFetch(getFusion);

  if (loading) return <div className="loading">Loading…</div>;
  if (error) return <div className="alert error">{error}</div>;

  const rows = Array.isArray(data) ? data : [];
  const first = rows[0] || {};

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Confidence-Calibrated Fusion</h2>
        <p className="page-sub">
          Stacks predictions from multiple base models and fits a logistic
          calibration layer on held-out folds to learn per-model reliability.
        </p>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="label">F1</div>
          <div className="value accent">{fmt(first.f1)}</div>
        </div>
        <div className="stat">
          <div className="label">ROC-AUC</div>
          <div className="value">{fmt(first.roc_auc)}</div>
        </div>
        <div className="stat">
          <div className="label">Accuracy</div>
          <div className="value">{fmt(first.accuracy)}</div>
        </div>
      </div>

      <div className="card">
        <h3>Fusion model metrics</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(first)
                .filter(([k]) => k !== "model_weights")
                .map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num">
                      {typeof v === "number" ? fmt(v) : String(v)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {first.model_weights && (
        <div className="card">
          <h3>Learned per-model calibration weights</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Base model column</th>
                  <th>Coefficient</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(first.model_weights).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num">{Number(v).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="alert info" style={{ marginTop: 12 }}>
            The calibration layer learns which base models to trust most.
            Weights are internal model parameters, not clinical evidence.
          </div>
        </div>
      )}
    </div>
  );
}