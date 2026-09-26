import React, { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { useFetch } from "../useFetch";
import { getAblation } from "../api";
import { ErrorBox, Loading, Selector, TASKS, fmt } from "../common";

export default function Ablation() {
  const { data, loading, error } = useFetch(getAblation);
  const [task, setTask] = useState("readmit30");
  if (loading) return <Loading what="ablation" />;
  if (error) return <ErrorBox error={error} />;

  const rows = data.filter((r) => r.task === task);
  const full = rows.find((r) => r.model === "tkgn");
  const removed = rows.filter((r) => r.model !== "tkgn")
    .sort((a, b) => a.delta_auroc_vs_full - b.delta_auroc_vs_full);

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Component ablation</h2>
        <p className="page-sub">
          Each TKGN component is removed in turn and the model is retrained
          from scratch on the same patient-grouped splits. Negative ΔAUROC
          means the component helps.
        </p>
      </div>
      <Selector label="Task" value={task} onChange={setTask} options={TASKS} />
      {full && (
        <div className="stat-row">
          <div className="stat"><div className="label">Full TKGN AUROC</div>
            <div className="value accent">{fmt(full.auroc)}</div></div>
          <div className="stat"><div className="label">Runs per variant</div>
            <div className="value">{full.n_runs}</div></div>
        </div>
      )}
      <div className="card">
        <h3>ΔAUROC when a component is removed</h3>
        <div className="chart-box">
          <ResponsiveContainer>
            <BarChart data={removed} layout="vertical"
              margin={{ top: 10, right: 30, bottom: 10, left: 170 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
              <XAxis type="number" tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <YAxis type="category" dataKey="label" width={170}
                tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <ReferenceLine x={0} stroke="#e6ebf5" />
              <Tooltip contentStyle={{ background: "#171f33", border: "1px solid #2a3550" }}
                formatter={(v) => fmt(v, 4)} />
              <Bar dataKey="delta_auroc_vs_full">
                {removed.map((r) => (
                  <Cell key={r.model}
                    fill={r.delta_auroc_vs_full < 0 ? "#38d9a9" : "#e5484d"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="card">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Variant</th><th>Runs</th><th>AUROC</th><th>SD</th>
              <th>AUPRC</th><th>ΔAUROC vs full</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model} className={r.model === "tkgn" ? "proposed" : ""}>
                  <td>{r.model === "tkgn" ? "TKGN (full)" : r.label}</td>
                  <td className="num">{r.n_runs}</td>
                  <td className="num">{fmt(r.auroc, 4)}</td>
                  <td className="num">{fmt(r.auroc_sd, 4)}</td>
                  <td className="num">{fmt(r.auprc, 4)}</td>
                  <td className="num">{r.model === "tkgn" ? "—" : fmt(r.delta_auroc_vs_full, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
