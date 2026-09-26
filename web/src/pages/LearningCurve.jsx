import React, { useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { useFetch } from "../useFetch";
import { getLearningCurve } from "../api";
import { COLORS, ErrorBox, LABELS, Loading, Selector, TASKS, fmt } from "../common";

const VIEWS = {
  auroc: "All test encounters",
  auroc_with_history: "Patients with ≥1 earlier stay",
  auroc_no_history: "First recorded stay",
};

export default function LearningCurve() {
  const { data, loading, error } = useFetch(getLearningCurve);
  const [task, setTask] = useState("readmit30");
  const [view, setView] = useState("auroc");
  if (loading) return <Loading what="learning curve" />;
  if (error) return <ErrorBox error={error} />;

  const rows = data.filter((r) => r.task === task);
  const models = [...new Set(rows.map((r) => r.model))];
  const fractions = [...new Set(rows.map((r) => r.fraction))].sort((a, b) => a - b);
  const chart = fractions.map((f) => {
    const point = { fraction: `${Math.round(f * 100)}%` };
    rows.filter((r) => r.fraction === f).forEach((r) => {
      point[r.model] = r[view];
      point.n_train = r.n_train;
    });
    return point;
  });

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Data efficiency</h2>
        <p className="page-sub">
          Models retrained on random subsets of training patients (same test
          patients). One repeat per training size, so small differences should
          be read as trends. The knowledge-graph comparison under data
          scarcity is shown on the Ablation page.
        </p>
      </div>
      <div className="toolbar">
        <Selector label="Task" value={task} onChange={setTask} options={TASKS} />
        <Selector label="Test subset" value={view} onChange={setView} options={VIEWS} />
      </div>
      <div className="card">
        <h3>Test AUROC vs. share of training patients</h3>
        <div className="chart-box">
          <ResponsiveContainer>
            <LineChart data={chart} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
              <XAxis dataKey="fraction" tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#171f33", border: "1px solid #2a3550" }}
                formatter={(v) => fmt(v, 4)} />
              <Legend />
              {models.map((m) => (
                <Line key={m} dataKey={m} name={LABELS[m]} stroke={COLORS[m]}
                  strokeWidth={2} dot />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="card">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Training share</th><th>Train encounters</th>
              {models.map((m) => <th key={m}>{LABELS[m]}</th>)}</tr></thead>
            <tbody>
              {chart.map((p) => (
                <tr key={p.fraction}>
                  <td>{p.fraction}</td><td className="num">{p.n_train?.toLocaleString()}</td>
                  {models.map((m) => <td key={m} className="num">{fmt(p[m], 4)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
