import React, { useState } from "react";
import { useFetch } from "../useFetch";
import { getSubgroups } from "../api";
import { ErrorBox, LABELS, Loading, Selector, TASKS, fmt, pct } from "../common";

const ATTRIBUTES = {
  race: "Race",
  gender: "Gender",
  age: "Age band",
  prior_encounters: "Number of earlier stays",
  hba1c_measured: "HbA1c measured during stay",
};

const MODELS = { tkgn_b: "TKGN-B", tkgn: "TKGN", lightgbm: "LightGBM",
  gru: "GRU", logistic_regression: "Logistic regression" };

export default function Fairness() {
  const { data, loading, error } = useFetch(getSubgroups);
  const [task, setTask] = useState("readmit30");
  const [attribute, setAttribute] = useState("race");
  if (loading) return <Loading what="subgroup results" />;
  if (error) return <ErrorBox error={error} />;

  const rows = data.filter((r) => r.task === task && r.attribute === attribute);
  const groups = [...new Set(rows.map((r) => r.group))];
  const models = Object.keys(MODELS).filter((m) => rows.some((r) => r.model === m));
  const cell = (g, m) => rows.find((r) => r.group === g && r.model === m);
  const ref = rows.filter((r) => r.model === "tkgn_b");
  const gap = ref.length > 1
    ? Math.max(...ref.map((r) => r.auroc)) - Math.min(...ref.map((r) => r.auroc))
    : null;

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Subgroup performance and fairness</h2>
        <p className="page-sub">
          Test-set discrimination, calibration error and sensitivity (at the
          validation-selected threshold) within demographic and clinical
          subgroups. Groups with fewer than 100 encounters are not shown.
        </p>
      </div>
      <div className="toolbar">
        <Selector label="Task" value={task} onChange={setTask} options={TASKS} />
        <Selector label="Attribute" value={attribute} onChange={setAttribute} options={ATTRIBUTES} />
      </div>
      {gap !== null && (
        <div className="alert info">
          Largest AUROC gap between {ATTRIBUTES[attribute].toLowerCase()} groups
          for TKGN-B: {fmt(gap, 3)}.
        </div>
      )}
      <div className="card">
        <h3>AUROC</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Group</th><th>n</th><th>Prevalence</th>
              {models.map((m) => <th key={m}>{MODELS[m]}</th>)}</tr></thead>
            <tbody>
              {groups.map((g) => {
                const any = cell(g, models[0]);
                return (
                  <tr key={g}>
                    <td>{g}</td>
                    <td className="num">{any?.n.toLocaleString()}</td>
                    <td className="num">{pct(any?.prevalence)}</td>
                    {models.map((m) => <td key={m} className="num">{fmt(cell(g, m)?.auroc)}</td>)}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <div className="card">
        <h3>Calibration and error balance (TKGN-B vs LightGBM)</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Group</th><th>Observed</th>
              <th>TKGN-B mean risk</th><th>TKGN-B ECE</th><th>TKGN-B sensitivity</th>
              <th>TKGN-B specificity</th><th>LightGBM ECE</th><th>LightGBM sensitivity</th></tr></thead>
            <tbody>
              {groups.map((g) => {
                const t = cell(g, "tkgn_b");
                const l = cell(g, "lightgbm");
                return (
                  <tr key={g}>
                    <td>{g}</td>
                    <td className="num">{pct(t?.prevalence)}</td>
                    <td className="num">{pct(t?.mean_predicted)}</td>
                    <td className="num">{fmt(t?.ece)}</td>
                    <td className="num">{fmt(t?.sensitivity)}</td>
                    <td className="num">{fmt(t?.specificity)}</td>
                    <td className="num">{fmt(l?.ece)}</td>
                    <td className="num">{fmt(l?.sensitivity)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <p className="muted small">
        Model names: {Object.values(MODELS).join(", ")}. {LABELS.tkgn_b} is the
        GBDT-anchored variant of TKGN.
      </p>
    </div>
  );
}
