import React from "react";
import { useFetch } from "../useFetch";
import { getAblation } from "../api";

const STAGE_ORDER = [
  "baseline",
  "gnn",
  "gnn+kg",
  "gnn+kg+longitudinal",
  "novel-tagnn",
  "novel-mhfin",
  "novel-ccf",
];

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(3);
}

export default function Ablation() {
  const { data, loading, error } = useFetch(getAblation);

  if (loading) return <div className="loading">Loading…</div>;
  if (error) return <div className="alert error">{error}</div>;

  const rows = (Array.isArray(data) ? data : []).slice().sort((a, b) => {
    const ai = STAGE_ORDER.indexOf(a.ablation_stage);
    const bi = STAGE_ORDER.indexOf(b.ablation_stage);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Ablation</h2>
        <p className="page-sub">
          Stages show how performance changes as components are added:
          baseline → graph → knowledge graph → longitudinal → novel models.
        </p>
      </div>

      <div className="card">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Model</th>
                <th>F1</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model}>
                  <td>
                    <span
                      className={`pill ${
                        (r.ablation_stage || "").startsWith("novel")
                          ? "novel"
                          : "baseline"
                      }`}
                    >
                      {r.ablation_stage}
                    </span>
                  </td>
                  <td>{r.model}</td>
                  <td className="num">{fmt(r.f1)}</td>
                  <td className="num">{fmt(r.roc_auc)}</td>
                  <td className="num">{fmt(r.pr_auc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="alert info">
        Ablation isolates the effect of each component family. Novel stages
        are highlighted in orange.
      </div>
    </div>
  );
}