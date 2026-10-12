import React, { useState } from "react";
import { BarList, LineChart } from "../components/charts";
import { ErrorMsg, Legend, Loading, Section, Segmented } from "../components/ui";
import {
  DESIGNS, FAMILIES, INPUT_GROUP_NAMES, MODELS, MODEL_ORDER, OUTCOMES,
  ROLE_COLOR, featureName, num, pText, pct,
} from "../format";
import { useApi } from "../useFetch";

function comparison(sig, model) {
  if (model === "tkgn_b" || !sig) return null;
  const c = sig[model];
  if (!c) return null;
  const { delta_auroc: delta, p_value: p } = c.bootstrap;
  if (p < 0.05 && delta > 0) return { text: `TKGN-B higher by ${num(delta)}`, cls: "verdict-better", p };
  if (p < 0.05 && delta < 0) return { text: `Higher than TKGN-B by ${num(-delta)}`, cls: "verdict-worse", p };
  return { text: "No clear difference", cls: "", p };
}

export default function Results() {
  const { data, error, loading } = useApi({
    summary: "/summary",
    significance: "/significance",
    repeats: "/repeat-tests",
    explanations: "/explanations",
    learning: "/learning-curve",
  });
  const [task, setTask] = useState("readmit30");
  const [split, setSplit] = useState("grouped");
  const [more, setMore] = useState(false);
  if (loading) return <Loading />;
  if (error) return <ErrorMsg error={error} />;

  const sig = data.significance[`${task}_${split}`]?.vs_tkgn_b;
  const rows = data.summary.filter((r) => r.task === task && r.split === split && MODELS[r.model]);
  const byModel = Object.fromEntries(rows.map((r) => [r.model, r]));
  const best = rows.reduce((a, r) => (r.auroc > a.auroc ? r : a), rows[0]);
  const wins = (m) => data.repeats.find((r) => r.task === task && r.split === split
    && r.reference === "tkgn_b" && r.model === m);
  const runs = split === "grouped" ? "five repeats" : "three seeds";

  const ex = data.explanations[task];
  const groups = ex.permutation_importance.groups
    .filter((g) => g.auroc_drop > 0.0015)
    .map((g) => ({ label: INPUT_GROUP_NAMES[g.group] || g.group, value: g.auroc_drop,
      tip: `AUROC falls by ${num(g.auroc_drop)} when this is scrambled` }));
  const shap = ex.treeshap.slice(0, 8).map((f) => ({
    label: featureName(f.feature), value: f.mean_abs_shap,
    tip: `Average effect ${num(f.mean_abs_shap)} on the log-odds of the outcome`,
  }));

  const lc = (m) => data.learning
    .filter((r) => r.task === task && r.model === m)
    .sort((a, b) => a.fraction - b.fraction)
    .map((r) => ({ x: r.fraction * 100, y: r.auroc,
      tip: `${Math.round(r.fraction * 100)}% of training data → AUROC ${num(r.auroc)}` }));
  const lcSeries = [
    { key: "tkgn_b", label: "TKGN-B", color: ROLE_COLOR.ours, points: lc("tkgn_b") },
    { key: "lightgbm", label: "LightGBM", color: ROLE_COLOR.reference, points: lc("lightgbm") },
  ];
  const lcVals = lcSeries.flatMap((s) => s.points.map((p) => p.y));

  return (
    <>
      <div className="intro">
        <div className="kicker">Results</div>
        <h1>How the models compare</h1>
        <p className="lead">
          Ten models were trained and tested on the same stays under the same
          rules: settings were tuned on a validation set, and the test patients
          were used once, for the final scores below.
        </p>
      </div>

      <div className="controls">
        <Segmented label="Outcome" value={task} onChange={setTask}
          options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
        <Segmented label="Test design" value={split} onChange={setSplit}
          options={{ grouped: DESIGNS.grouped.short, temporal: DESIGNS.temporal.short }} />
      </div>
      <p className="muted small" style={{ marginTop: -8 }}>
        {OUTCOMES[task].question} {DESIGNS[split].long}.
      </p>

      <Section num="1" title="Model comparison">
        <p className="small muted">
          Mean over {runs} (± standard deviation). The last column compares
          each model with TKGN-B on the first test set using a bootstrap test
          that resamples whole patients.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">AUROC</th>
                <th className="num">AUPRC</th>
                <th className="num">Calibration error</th>
                {more && <>
                  <th className="num">Sensitivity</th>
                  <th className="num">Specificity</th>
                  <th className="num">Precision</th>
                  <th className="num">Brier score</th>
                </>}
                <th>Compared with TKGN-B</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(FAMILIES).map(([family, title]) => (
                <React.Fragment key={family}>
                  <tr className="group-row"><td colSpan={more ? 9 : 5}>{title}</td></tr>
                  {MODEL_ORDER.filter((m) => MODELS[m].family === family && byModel[m]).map((m) => {
                    const r = byModel[m];
                    const c = comparison(sig, m);
                    const w = wins(m);
                    return (
                      <tr key={m} className={m === "tkgn_b" ? "highlight" : ""}>
                        <td>
                          {MODELS[m].name}
                          {m === best.model && <span className="muted small"> · highest</span>}
                        </td>
                        <td className="num">{num(r.auroc)} <span className="muted">± {num(r.auroc_sd)}</span></td>
                        <td className="num">{num(r.auprc)}</td>
                        <td className="num">{num(r.ece)}</td>
                        {more && <>
                          <td className="num">{pct(r.sensitivity, 0)}</td>
                          <td className="num">{pct(r.specificity, 0)}</td>
                          <td className="num">{pct(r.ppv, 0)}</td>
                          <td className="num">{num(r.brier, 4)}</td>
                        </>}
                        <td>
                          {m === "tkgn_b" ? <span className="muted">Reference</span> : c && (
                            <>
                              <span className={c.cls}>{c.text}</span>
                              <span className="muted small"> ({pText(c.p)}
                                {w ? `; ahead in ${w.wins} of ${w.n_repeats} runs` : ""})</span>
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: 12 }}>
          <button type="button" className="btn link" onClick={() => setMore(!more)}>
            {more ? "Show fewer measures" : "Show sensitivity, specificity and other measures"}
          </button>
        </p>
        <div className="note">
          <p>
            AUPRC rewards finding the patients who do have the outcome, which
            matters when the outcome is uncommon (about 11% for readmission).
            Calibration error is the average gap between predicted and observed
            risk; lower is better. Sensitivity and specificity use a cut-off
            chosen on the validation set before testing.
          </p>
        </div>
      </Section>

      <Section num="2" title="What the predictions rely on">
        <div className="two-col">
          <figure>
            <h3>Inputs that matter most to TKGN</h3>
            <BarList rows={groups} ariaLabel="Importance of input groups" />
            <figcaption>
              Fall in AUROC when one group of inputs is scrambled across
              patients. A bigger fall means the model depends on it more.
            </figcaption>
          </figure>
          <figure>
            <h3>Strongest single factors</h3>
            <BarList rows={shap} format={(v) => v.toFixed(2)} color="var(--c-ref)"
              ariaLabel="Most influential features" />
            <figcaption>
              Average size of each factor&apos;s effect in the LightGBM part of
              TKGN-B (SHAP values on the log-odds scale).
            </figcaption>
          </figure>
        </div>
      </Section>

      <Section num="3" title="With less training data">
        <figure>
          <Legend items={lcSeries.map((s) => ({ label: s.label, color: s.color, line: true }))} />
          <LineChart series={lcSeries} logX xDomain={[4, 110]} xTicks={[5, 10, 25, 50, 100]}
            xFormat={(v) => `${v}%`} yDomain={[Math.floor(Math.min(...lcVals) * 100) / 100 - 0.005,
              Math.ceil(Math.max(...lcVals) * 100) / 100 + 0.002]}
            yTicks={[0.62, 0.64, 0.66, 0.68, 0.70].filter((t) =>
              t >= Math.min(...lcVals) - 0.01 && t <= Math.max(...lcVals) + 0.01)}
            yFormat={(v) => v.toFixed(2)} xLabel="Share of training patients used"
            ariaLabel="AUROC by amount of training data" />
          <figcaption>
            Both models retrained on part of the training patients and tested on
            the same test patients (one run per size, so read this as a trend).
            {task === "readmit30"
              ? " For readmission TKGN-B stays ahead of LightGBM at every size, most clearly with little data."
              : " For treatment escalation LightGBM is better with 5% of the data and TKGN-B from 10% upwards."}
          </figcaption>
        </figure>
      </Section>
    </>
  );
}
