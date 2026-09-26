import React, { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useFetch } from "../useFetch";
import { getExplanations, getGates } from "../api";
import { ErrorBox, Loading, Selector, TASKS, fmt } from "../common";

const tooltipStyle = { background: "#171f33", border: "1px solid #2a3550" };

export default function Explainability() {
  const ex = useFetch(getExplanations);
  const gates = useFetch(getGates);
  const [task, setTask] = useState("readmit30");
  if (ex.loading || gates.loading) return <Loading what="explanations" />;
  if (ex.error || gates.error) return <ErrorBox error={ex.error || gates.error} />;

  const e = ex.data[task];
  const g = gates.data.filter((r) => r.task === task);
  if (!e) return <div className="empty">No explanation for this task yet.</div>;
  const perm = e.permutation_importance.groups;
  const shap = e.treeshap.slice(0, 15).map((d) => ({ ...d, short: d.feature.slice(0, 36) }));

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Explainability</h2>
        <p className="page-sub">
          Global explanations on held-out test encounters: permutation
          importance of input groups for TKGN, exact TreeSHAP values for the
          gradient-boosting anchor of TKGN-B, and the learned history gates.
        </p>
      </div>
      <Selector label="Task" value={task} onChange={setTask} options={TASKS} />

      <div className="grid grid-2">
        <div className="card">
          <h3>TKGN: AUROC drop when a group is shuffled</h3>
          <p className="muted small">Base AUROC {fmt(e.permutation_importance.base_auroc)} on{" "}
            {e.permutation_importance.n.toLocaleString()} test encounters.</p>
          <div className="chart-box" style={{ height: 380 }}>
            <ResponsiveContainer>
              <BarChart data={perm} layout="vertical" margin={{ left: 190, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                <XAxis type="number" tick={{ fill: "#9aa7c0", fontSize: 11 }} />
                <YAxis type="category" dataKey="group" width={190}
                  tick={{ fill: "#9aa7c0", fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v, 4)} />
                <Bar dataKey="auroc_drop" fill="#f4a340" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card">
          <h3>GBDT anchor: mean |SHAP| (log-odds)</h3>
          <div className="chart-box" style={{ height: 380 }}>
            <ResponsiveContainer>
              <BarChart data={shap} layout="vertical" margin={{ left: 200, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                <XAxis type="number" tick={{ fill: "#9aa7c0", fontSize: 11 }} />
                <YAxis type="category" dataKey="short" width={200}
                  tick={{ fill: "#9aa7c0", fontSize: 10 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v, 4)} />
                <Bar dataKey="mean_abs_shap" fill="#4f8ef7" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>History-reliability gates by number of earlier stays</h3>
        <p className="muted small">
          Mean gate activation (0 = history ignored, 1 = fully used). The
          history gate is expected to open as more past encounters exist.
        </p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Earlier stays</th><th>Test encounters</th>
              <th>History gate</th><th>Change (delta) gate</th></tr></thead>
            <tbody>
              {g.map((r) => (
                <tr key={r.prior_encounters}>
                  <td>{r.prior_encounters}</td>
                  <td className="num">{r.n.toLocaleString()}</td>
                  <td className="num">{fmt(r.gate_history)}</td>
                  <td className="num">{fmt(r.gate_delta)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Example patients: attention over earlier stays</h3>
        <div className="grid grid-2">
          {e.examples.map((p) => (
            <div key={p.encounter_id} className="encounter-card">
              <h4>
                Patient {p.patient_nbr}
                <span className="muted small">risk {fmt(p.probability)}</span>
              </h4>
              <p className="muted small" style={{ marginTop: 0 }}>
                history gate {fmt(p.gate_history)} · delta gate {fmt(p.gate_delta)}
              </p>
              <table>
                <thead><tr><th>Earlier stay</th><th>Primary dx</th><th>Readmitted</th><th>Attention</th></tr></thead>
                <tbody>
                  {p.history.map((h) => (
                    <tr key={h.encounter_id}>
                      <td>{h.encounter_id}</td><td>{h.primary_diagnosis}</td>
                      <td>{h.readmitted}</td><td className="num">{fmt(h.attention)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
