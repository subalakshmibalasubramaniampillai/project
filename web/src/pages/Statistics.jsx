import React, { useState } from "react";
import { useFetch } from "../useFetch";
import { getRepeatTests, getSignificance } from "../api";
import {
  ErrorBox, LABELS, Loading, ModelName, SPLITS, Selector, TASKS, fmt, pValue,
} from "../common";

export default function Statistics() {
  const sig = useFetch(getSignificance);
  const rep = useFetch(getRepeatTests);
  const [task, setTask] = useState("readmit30");
  const [split, setSplit] = useState("grouped");
  const [reference, setReference] = useState("tkgn_b");

  if (sig.loading || rep.loading) return <Loading what="statistics" />;
  if (sig.error || rep.error) return <ErrorBox error={sig.error || rep.error} />;

  const entry = sig.data[`${task}_${split}`];
  if (!entry) return <div className="empty">No runs for this setting yet.</div>;
  const comparisons = entry[`vs_${reference}`] || {};
  const repeats = rep.data.filter(
    (r) => r.task === task && r.split === split && r.reference === reference,
  );

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Statistical comparison</h2>
        <p className="page-sub">
          First test run of each design. 95% confidence intervals use a
          patient-clustered bootstrap (1,000 resamples) so that repeat stays of
          one person are resampled together. Paired AUROC differences are
          tested with DeLong's method and with the clustered bootstrap.
        </p>
      </div>
      <div className="toolbar">
        <Selector label="Task" value={task} onChange={setTask} options={TASKS} />
        <Selector label="Design" value={split} onChange={setSplit} options={SPLITS} />
        <Selector label="Reference model" value={reference} onChange={setReference}
          options={{ tkgn_b: "TKGN-B", tkgn: "TKGN" }} />
      </div>

      <div className="card">
        <h3>{LABELS[reference]} versus each model</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Compared model</th><th>ΔAUROC</th><th>95% CI (bootstrap)</th>
                <th>p (bootstrap)</th><th>DeLong z</th><th>p (DeLong)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparisons).map(([model, c]) => (
                <tr key={model}>
                  <td><ModelName model={model} /></td>
                  <td className="num">{fmt(c.bootstrap.delta_auroc, 4)}</td>
                  <td className="num">[{fmt(c.bootstrap.ci[0], 4)}, {fmt(c.bootstrap.ci[1], 4)}]</td>
                  <td className="num">{pValue(c.bootstrap.p_value)}</td>
                  <td className="num">{fmt(c.delong.z, 2)}</td>
                  <td className="num">{pValue(c.delong.p_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          Positive ΔAUROC means {LABELS[reference]} ranks encounters better.
          DeLong assumes independent encounters; the clustered bootstrap does
          not, and is the more conservative of the two.
        </p>
      </div>

      <div className="card">
        <h3>Consistency across repeated runs</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Compared model</th><th>Runs</th><th>Mean ΔAUROC</th>
                <th>Runs won</th><th>p (paired t)</th></tr>
            </thead>
            <tbody>
              {repeats.map((r) => (
                <tr key={r.model}>
                  <td><ModelName model={r.model} /></td>
                  <td className="num">{r.n_repeats}</td>
                  <td className="num">{fmt(r.mean_delta_auroc, 4)}</td>
                  <td className="num">{r.wins}/{r.n_repeats}</td>
                  <td className="num">{pValue(r.t_test_p)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Bootstrap 95% CIs per model</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>AUROC CI</th><th>AUPRC CI</th><th>Brier CI</th></tr></thead>
            <tbody>
              {Object.entries(entry.ci)
                .filter(([m]) => LABELS[m])
                .map(([model, ci]) => (
                  <tr key={model}>
                    <td><ModelName model={model} /></td>
                    <td className="num">[{fmt(ci.auroc[0])}, {fmt(ci.auroc[1])}]</td>
                    <td className="num">[{fmt(ci.auprc[0])}, {fmt(ci.auprc[1])}]</td>
                    <td className="num">[{fmt(ci.brier[0], 4)}, {fmt(ci.brier[1], 4)}]</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
