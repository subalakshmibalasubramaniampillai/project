import React, { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Cell,
} from "recharts";
import { useFetch } from "../useFetch";
import { getOverview, getSummary } from "../api";
import {
  COLORS, ErrorBox, LABELS, Loading, ModelName, PROPOSED, SPLITS, Selector,
  TASKS, fmt, pct,
} from "../common";

const MAIN = [
  "logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp",
  "gru", "retain", "transformer", "tkgn", "tkgn_b",
];

const COLS = [
  ["auroc", "AUROC"], ["auprc", "AUPRC"], ["brier", "Brier"], ["ece", "ECE"],
  ["calibration_slope", "Cal. slope"], ["sensitivity", "Sens."],
  ["specificity", "Spec."], ["f1", "F1"], ["balanced_accuracy", "Bal. acc."],
];

export default function Overview() {
  const overview = useFetch(getOverview);
  const summary = useFetch(getSummary);
  const [task, setTask] = useState("readmit30");
  const [split, setSplit] = useState("grouped");

  if (overview.loading || summary.loading) return <Loading />;
  if (overview.error || summary.error)
    return <ErrorBox error={overview.error || summary.error} />;

  const d = overview.data.dataset;
  const rows = summary.data
    .filter((r) => r.task === task && r.split === split && MAIN.includes(r.model))
    .sort((a, b) => MAIN.indexOf(a.model) - MAIN.indexOf(b.model));
  const best = rows.reduce((acc, r) => (!acc || r.auroc > acc.auroc ? r : acc), null);
  const splitInfo = overview.data.splits[`${task}_${split}`];

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Real-world evaluation</h2>
        <p className="page-sub">
          {d.source}. {d.cohort_encounters.toLocaleString()} inpatient encounters
          of {d.patients.toLocaleString()} patients after excluding deaths and
          hospice discharges. No synthetic or simulated records are used.
        </p>
      </div>

      <div className="stat-row">
        <div className="stat"><div className="label">Encounters</div>
          <div className="value">{d.cohort_encounters.toLocaleString()}</div></div>
        <div className="stat"><div className="label">Patients</div>
          <div className="value">{d.patients.toLocaleString()}</div></div>
        <div className="stat"><div className="label">Patients with repeat stays</div>
          <div className="value">{d.patients_with_repeat_encounters.toLocaleString()}</div></div>
        <div className="stat"><div className="label">30-day readmission</div>
          <div className="value">{pct(d.readmit30_rate)}</div></div>
        <div className="stat"><div className="label">Escalation (next stay)</div>
          <div className="value">{pct(d.escalation_rate)}</div></div>
      </div>

      <div className="toolbar">
        <Selector label="Prediction task" value={task} onChange={setTask} options={TASKS} />
        <Selector label="Validation design" value={split} onChange={setSplit} options={SPLITS} />
      </div>
      {splitInfo && (
        <p className="muted small" style={{ margin: 0 }}>
          Train {splitInfo.n.train.toLocaleString()} · validation{" "}
          {splitInfo.n.val.toLocaleString()} · test {splitInfo.n.test.toLocaleString()}{" "}
          encounters; test prevalence {pct(splitInfo.prevalence.test)}.
          Operating thresholds were chosen on validation (max F1) and frozen.
        </p>
      )}

      <div className="card">
        <h3>Test AUROC (mean ± SD over runs)</h3>
        <div className="chart-box">
          <ResponsiveContainer>
            <BarChart data={rows.map((r) => ({ ...r, name: LABELS[r.model] }))}
              margin={{ top: 10, right: 10, bottom: 50, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
              <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0}
                tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <YAxis domain={[0.55, "auto"]} tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#171f33", border: "1px solid #2a3550" }}
                formatter={(v) => fmt(v, 4)} />
              <Bar dataKey="auroc">
                {rows.map((r) => <Cell key={r.model} fill={COLORS[r.model]} />)}
                <ErrorBar dataKey="auroc_sd" width={4} stroke="#e6ebf5" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3>All models — {TASKS[task]}, {SPLITS[split]}</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Runs</th>
                {COLS.map(([, label]) => <th key={label}>{label}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model} className={PROPOSED.has(r.model) ? "proposed" : ""}>
                  <td>
                    <ModelName model={r.model} />
                    {best && r.model === best.model && (
                      <span className="pill best" style={{ marginLeft: 6 }}>best AUROC</span>
                    )}
                  </td>
                  <td className="num">{r.n_runs}</td>
                  {COLS.map(([key]) => (
                    <td key={key} className="num">
                      {fmt(r[key])}
                      {r.n_runs > 1 && (
                        <span className="muted small"> ±{fmt(r[`${key}_sd`])}</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="alert info">
        Metrics come from held-out test encounters of real patients never seen
        during training (grouped design) or from the most recent 20% of
        admissions (temporal design). They are research results, not clinical
        validation.
      </div>
    </div>
  );
}
