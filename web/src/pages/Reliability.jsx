import React, { useState } from "react";
import { ForestChart, LineChart } from "../components/charts";
import { ErrorMsg, Legend, Loading, Section, Segmented } from "../components/ui";
import {
  MODELS, MODEL_ORDER, OUTCOMES, ROLE_COLOR, count, num, pct, roleOf,
} from "../format";
import { useApi } from "../useFetch";

const ATTRIBUTES = [
  ["gender", "Sex"],
  ["race", "Race"],
  ["age", "Age (years)"],
  ["prior_encounters", "Earlier stays"],
];

const AGE_ORDER = { "<50": 0, "50-69": 1, "70+": 2 };

function describeGroup(attribute, group) {
  if (attribute === "race") {
    if (group === "Unknown") return "patients whose race was not recorded";
    if (group === "Other") return "patients of other races";
    return `${groupLabel(attribute, group)} patients`;
  }
  if (attribute === "gender") return group === "Female" ? "women" : "men";
  if (attribute === "age") return `patients aged ${groupLabel(attribute, group).toLowerCase()}`;
  return { 0: "first stays", 1: "patients with one earlier stay" }[group]
    || "patients with two or more earlier stays";
}

function groupLabel(attribute, group) {
  if (attribute === "race" && group === "AfricanAmerican") return "African American";
  if (attribute === "race" && group === "Unknown") return "Not recorded";
  if (attribute === "prior_encounters") return group === "0" ? "None" : group === "1" ? "One" : "Two or more";
  if (attribute === "age") return { "<50": "Under 50", "50-69": "50 to 69", "70+": "70 and over" }[group];
  return group;
}

export default function Reliability() {
  const { data, error, loading } = useApi({
    calibration: "/calibration",
    summary: "/summary",
    subgroups: "/subgroups",
    shift: "/temporal-shift",
  });
  const [task, setTask] = useState("readmit30");
  if (loading) return <Loading />;
  if (error) return <ErrorMsg error={error} />;

  const curves = data.calibration[`${task}_grouped`];
  const calSeries = [
    { key: "tkgn_b", label: "TKGN-B", color: ROLE_COLOR.ours },
    { key: "lightgbm", label: "LightGBM", color: ROLE_COLOR.reference },
  ].map((s) => ({
    ...s,
    points: curves[s.key].map((b) => ({
      x: b.mean_predicted, y: b.observed,
      tip: `predicted ${pct(b.mean_predicted)}, observed ${pct(b.observed)}`,
    })),
  }));
  const top = Math.max(...calSeries.flatMap((s) => s.points.flatMap((p) => [p.x, p.y])));
  const calMax = Math.ceil(top * 10) / 10;
  const calTicks = Array.from({ length: Math.round(calMax / 0.1) + 1 }, (_, i) => +(i * 0.1).toFixed(1));
  const ece = (m) => data.summary.find((r) => r.task === task && r.split === "grouped" && r.model === m)?.ece;

  const sub = data.subgroups.filter((r) => r.task === task && r.n >= 300);
  const cell = (attr, group, model) => sub.find((r) => r.attribute === attr && r.group === group && r.model === model);

  const ours = sub.filter((r) => r.model === "tkgn_b");
  const worst = ours.reduce((a, r) =>
    (Math.abs(r.mean_predicted - r.prevalence) > Math.abs(a.mean_predicted - a.prevalence) ? r : a), ours[0]);
  const worstGap = worst.mean_predicted - worst.prevalence;

  const sh = data.shift;
  const tempRows = MODEL_ORDER
    .map((m) => data.summary.find((r) => r.task === task && r.split === "temporal" && r.model === m))
    .filter(Boolean)
    .sort((a, b) => b.auroc - a.auroc)
    .map((r) => ({
      key: r.model, label: MODELS[r.model].name, value: r.auroc,
      color: ROLE_COLOR[roleOf(r.model)], bold: r.model === "tkgn_b",
    }));
  const tMin = Math.min(...tempRows.map((r) => r.value));

  return (
    <>
      <div className="intro">
        <div className="kicker">Reliability</div>
        <h1>Can the risk estimates be trusted?</h1>
        <p className="lead">
          A useful risk model has to give honest percentages, work for every
          group of patients, and keep working when hospital data change over time.
        </p>
      </div>

      <div className="controls">
        <Segmented label="Outcome" value={task} onChange={setTask}
          options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
      </div>

      <Section num="1" title="Do predicted risks match what happened?">
        <div className="two-col">
          <figure>
            <Legend items={[
              ...calSeries.map((s) => ({ label: s.label, color: s.color, line: true })),
              { label: "Perfect agreement", color: "var(--rule-strong)", line: true },
            ]} />
            <LineChart series={calSeries} diagonal xDomain={[0, calMax]} yDomain={[0, calMax]}
              xTicks={calTicks} yTicks={calTicks} xFormat={(v) => `${Math.round(v * 100)}%`}
              yFormat={(v) => `${Math.round(v * 100)}%`} xLabel="Predicted risk"
              yLabel="Observed rate" height={320} ariaLabel="Calibration curve" />
            <figcaption>
              Test stays sorted into ten equal groups by predicted risk. Points on
              the grey diagonal mean the predicted and observed rates agree.
            </figcaption>
          </figure>
          <div>
            <p>
              When TKGN-B gives a group of patients a risk of 20%, close to 20%
              of them go on to have the outcome. The average gap between
              predicted and observed risk was {num(ece("tkgn_b"))} for TKGN-B and{" "}
              {num(ece("lightgbm"))} for LightGBM, both small.
            </p>
            <p>
              This matters as much as ranking accuracy: a model that ranks
              patients well but overstates their risk would lead to unnecessary
              follow-up.
            </p>
          </div>
        </div>
      </Section>

      <Section num="2" title="Does it work equally well for different patients?">
        <p className="small muted">
          First test set; groups with fewer than 300 test stays are left out
          because their scores are unstable.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th className="num">Test stays</th>
                <th className="num">Observed rate</th>
                <th className="num">Mean predicted (TKGN-B)</th>
                <th className="num">AUROC TKGN-B</th>
                <th className="num">AUROC LightGBM</th>
              </tr>
            </thead>
            <tbody>
              {ATTRIBUTES.map(([attr, title]) => {
                const groups = [...new Set(sub.filter((r) => r.attribute === attr).map((r) => r.group))]
                  .sort((a, b) => (attr === "age" ? AGE_ORDER[a] - AGE_ORDER[b] : a.localeCompare(b)));
                return (
                  <React.Fragment key={attr}>
                    <tr className="group-row"><td colSpan={6}>{title}</td></tr>
                    {groups.map((g) => {
                      const t = cell(attr, g, "tkgn_b");
                      const l = cell(attr, g, "lightgbm");
                      if (!t) return null;
                      return (
                        <tr key={g}>
                          <td>{groupLabel(attr, g)}</td>
                          <td className="num">{count(t.n)}</td>
                          <td className="num">{pct(t.prevalence)}</td>
                          <td className="num">{pct(t.mean_predicted)}</td>
                          <td className="num">{num(t.auroc)}</td>
                          <td className="num">{l ? num(l.auroc) : "–"}</td>
                        </tr>
                      );
                    })}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        <p style={{ marginTop: 14 }}>
          In every group the mean predicted risk is within{" "}
          {(Math.abs(worstGap) * 100).toFixed(1)} percentage points of the observed
          rate. The largest gap is a slight {worstGap > 0 ? "overestimate" : "underestimate"}{" "}
          for {describeGroup(worst.attribute, worst.group)}{" "}
          ({pct(worst.mean_predicted)} predicted, {pct(worst.prevalence)} observed).{" "}
          {task === "readmit30"
            ? "Accuracy is lowest for patients aged 70 and over and for patients with two or more earlier stays, whose risks are high and similar to each other, which makes ranking them harder."
            : "Accuracy is lowest for patients under 50 and for first stays."}
        </p>
      </Section>

      <Section num="3" title="Does it hold up over time?">
        <div className="two-col">
          <div>
            <p>
              The hospitals recorded data differently in later years. These
              changes are invisible when patients are split at random, but a
              model in use would face them.
            </p>
            <table>
              <thead>
                <tr><th>Measure</th><th className="num">Older 70%</th><th className="num">Newest 20%</th></tr>
              </thead>
              <tbody>
                <tr><td>Most diagnoses recorded on one stay</td>
                  <td className="num">{sh.train.number_diagnoses_max}</td>
                  <td className="num">{sh.test.number_diagnoses_max}</td></tr>
                <tr><td>Stays with an earlier stay</td>
                  <td className="num">{pct(sh.train.share_with_history)}</td>
                  <td className="num">{pct(sh.test.share_with_history)}</td></tr>
                <tr><td>Emergency visits in the prior year (mean)</td>
                  <td className="num">{num(sh.train.number_emergency_mean, 2)}</td>
                  <td className="num">{num(sh.test.number_emergency_mean, 2)}</td></tr>
              </tbody>
            </table>
          </div>
          <figure>
            <h3>AUROC on the newest admissions</h3>
            <ForestChart rows={tempRows} domain={[Math.min(0.59, tMin - 0.01), 0.71]}
              ticks={[0.6, 0.65, 0.7]} ariaLabel="AUROC on newest admissions" />
            <figcaption>
              {task === "readmit30"
                ? "For readmission a random forest coped best with the change; TKGN-B performed like LightGBM and XGBoost. Logistic regression and the simple neural network lost the most."
                : "For treatment escalation TKGN-B was the most accurate model on the newest admissions."}
            </figcaption>
          </figure>
        </div>
      </Section>
    </>
  );
}
