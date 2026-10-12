import React, { useState } from "react";
import { PAPER_URL } from "../api";
import { BarList, ForestChart, LineChart } from "../components/charts";
import { Chips, ErrorMsg, Legend, Loading } from "../components/ui";
import {
  DESIGNS, FAMILIES, MODELS, MODEL_ORDER, OUTCOMES, ROLE_COLOR, count, num, pText, pct, roleOf,
} from "../format";
import { SITE } from "../site";
import { useApi } from "../useFetch";

const PARTS = {
  tkgn_no_history: "Earlier stays",
  tkgn_no_prior_outcomes: "Outcomes of earlier stays",
  tkgn_no_aux: "Extra training signal (3-level outcome)",
  tkgn_no_gate: "Reliability gate",
  tkgn_no_delta: "Change since last stay",
  tkgn_no_kg: "Diagnosis hierarchy and co-morbidity graph",
  tkgn_no_cooccurrence: "Co-morbidity graph only",
};

function verdict(sig, model) {
  const c = sig?.[model];
  if (!c) return null;
  const { delta_auroc: delta, p_value: p } = c.bootstrap;
  if (p < 0.05 && delta > 0) return { text: `${SITE.brand} higher by ${num(delta)}`, cls: "good", p };
  if (p < 0.05 && delta < 0) return { text: `Higher than ${SITE.brand} by ${num(-delta)}`, cls: "bad", p };
  return { text: "No clear difference", cls: "", p };
}

function Block({ eyebrow, title, children, tint }) {
  return (
    <section className={`band${tint ? " tint" : ""}`}>
      <div className="wrap">
        <div className="band-head">
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
        </div>
        {children}
      </div>
    </section>
  );
}

export default function Evidence() {
  const { data, error, loading } = useApi({
    overview: "/overview", summary: "/summary", significance: "/significance",
    repeats: "/repeat-tests", calibration: "/calibration", subgroups: "/subgroups",
    shift: "/temporal-shift", ablation: "/ablation", bands: "/risk-bands",
    kg: "/kg-efficiency", gates: "/gates",
  });
  const [task, setTask] = useState("readmit30");
  const [split, setSplit] = useState("grouped");
  if (loading) return <div className="wrap page"><Loading /></div>;
  if (error) return <div className="wrap page"><ErrorMsg error={error} /></div>;

  const d = data.overview.dataset;
  const sp = data.overview.splits;
  const rows = data.summary.filter((r) => r.task === task && r.split === split && MODELS[r.model]);
  const byModel = Object.fromEntries(rows.map((r) => [r.model, r]));
  const sig = data.significance[`${task}_${split}`]?.vs_tkgn_b;
  const wins = (m) => data.repeats.find((r) => r.task === task && r.split === split
    && r.reference === "tkgn_b" && r.model === m);
  const forest = MODEL_ORDER.filter((m) => byModel[m]).map((m) => byModel[m])
    .sort((a, b) => b.auroc - a.auroc)
    .map((r) => ({
      key: r.model, label: MODELS[r.model].name, value: r.auroc,
      lo: r.auroc - r.auroc_sd, hi: r.auroc + r.auroc_sd,
      color: ROLE_COLOR[roleOf(r.model)], bold: r.model === "tkgn_b",
    }));
  const lo = Math.min(...forest.map((r) => r.lo));
  const hi = Math.max(...forest.map((r) => r.hi));
  const domain = [Math.floor(lo * 50) / 50 - 0.005, Math.ceil(hi * 50) / 50 + 0.002];
  const ticks = [];
  for (let t = Math.ceil(domain[0] * 50) / 50; t <= domain[1] + 1e-9; t += 0.02) ticks.push(+t.toFixed(2));

  const abl = (m, t = task) => data.ablation.find((r) => r.task === t && r.model === m);
  const ablRows = data.ablation.filter((r) => r.task === task && PARTS[r.model])
    .map((r) => ({ label: PARTS[r.model], value: -r.delta_auroc_vs_full }))
    .sort((a, b) => b.value - a.value);

  const curves = data.calibration[`${task}_grouped`];
  const calSeries = [
    { key: "tkgn_b", label: SITE.brand, color: ROLE_COLOR.ours },
    { key: "lightgbm", label: "LightGBM", color: ROLE_COLOR.reference },
  ].map((s) => ({ ...s, points: curves[s.key].map((b) => ({ x: b.mean_predicted, y: b.observed,
    tip: `predicted ${pct(b.mean_predicted)}, observed ${pct(b.observed)}` })) }));
  const calMax = Math.ceil(Math.max(...calSeries.flatMap((s) => s.points.flatMap((p) => [p.x, p.y]))) * 10) / 10;
  const calTicks = Array.from({ length: Math.round(calMax / 0.1) + 1 }, (_, i) => +(i * 0.1).toFixed(1));

  const sub = data.subgroups.filter((r) => r.task === task && r.n >= 300 && r.model === "tkgn_b"
    && ["gender", "race", "age", "prior_encounters"].includes(r.attribute));
  const subLabel = (r) => ({
    AfricanAmerican: "African American", Unknown: "Race not recorded", Female: "Women", Male: "Men",
    "<50": "Age under 50", "50-69": "Age 50 to 69", "70+": "Age 70 and over",
    0: "No earlier stay", 1: "One earlier stay", "2+": "Two or more earlier stays",
  }[r.group] || r.group);
  const band = data.bands[task];
  const kgSmall = data.kg.filter((r) => r.fraction < 1);
  const gate = (t, k) => data.gates.find((g) => g.task === t && g.prior_encounters === k)?.gate_history;

  return (
    <>
      <section className="hero">
        <div className="wrap" style={{ gridTemplateColumns: "1fr" }}>
          <div>
            <span className="eyebrow">Evidence for clinicians and reviewers</span>
            <h1 style={{ fontSize: "clamp(32px, 4vw, 48px)" }}>How {SITE.brand} was built and tested</h1>
            <p className="lead">
              Built and tested on {count(d.cohort_encounters)} real, de-identified
              hospital stays of {count(d.patients)} people with diabetes
              (Diabetes 130-US Hospitals database, 1999 to 2008). Nothing was
              simulated. Settings were chosen on validation data; test patients
              were used once.
            </p>
            <div className="hero-actions">
              <a className="btn" href={PAPER_URL}>Download the manuscript (PDF)</a>
              <a className="btn ghost" href={SITE.repository}>Code and results</a>
            </div>
          </div>
        </div>
      </section>

      <Block eyebrow="Novelty" title="What is new in this work">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Contribution</th><th>How it works</th><th>What the tests showed</th></tr></thead>
            <tbody>
              <tr>
                <td><strong>History-aware sequence model (TKGN)</strong></td>
                <td>Encodes each stay and reads up to ten earlier stays with a GRU and recency attention, including the outcome of each earlier stay and the change since the last stay.</td>
                <td>Removing the history cost {num(-abl("tkgn_no_history", "readmit30").delta_auroc_vs_full)} AUROC (readmission) and {num(-abl("tkgn_no_history", "escalation").delta_auroc_vs_full)} (escalation), the largest effect of any part.</td>
              </tr>
              <tr>
                <td><strong>History-reliability gate</strong></td>
                <td>Feature-wise gates conditioned on the current stay, the history summary, the change since the last stay and the number of earlier stays.</td>
                <td>For escalation the gate opened from {num(gate("escalation", "0"), 2)} (no history) to {num(gate("escalation", "5+"), 2)} (five or more stays); its effect on AUROC was within noise.</td>
              </tr>
              <tr>
                <td><strong>Boosting-anchored residual learning (TKGN-B)</strong></td>
                <td>The network is trained on top of an out-of-fold LightGBM logit, so it learns only what boosting misses.</td>
                <td>Best mean AUROC on both outcomes; beat GRU, RETAIN and Transformer in every repeat; small, non-significant edge over LightGBM and XGBoost.</td>
              </tr>
              <tr>
                <td><strong>Second outcome: treatment escalation</strong></td>
                <td>Insulin started, a dose increased, or more diabetes drugs at the next stay ({count(d.escalation_samples)} stays, {pct(d.escalation_rate)} positive).</td>
                <td>On the newest admissions {SITE.brand} was the best model (AUROC {num(data.summary.find((r) => r.task === "escalation" && r.split === "temporal" && r.model === "tkgn_b").auroc)}), ahead of every other model (significant on the first seed).</td>
              </tr>
              <tr>
                <td><strong>Leakage-controlled evaluation</strong></td>
                <td>Patient-disjoint splits (5 repeats), prospective temporal split, patient-clustered bootstrap, DeLong tests, calibration and subgroup audits.</td>
                <td>Exposed real data shift over time that random splits hide (see below).</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small muted" style={{ marginTop: 14 }}>
          Also tested and reported honestly: a hybrid ICD-9 knowledge graph
          (hierarchy plus co-morbidity links). It did not improve accuracy, even
          with only 5 to 25% of the training data (changes between{" "}
          {num(Math.min(...kgSmall.map((r) => r.delta_auroc)))} and{" "}
          {num(Math.max(...kgSmall.map((r) => r.delta_auroc)))}).
        </p>
      </Block>

      <Block eyebrow="Accuracy" title="Compared with nine other models" tint>
        <div className="controls">
          <Chips label="Outcome" value={task} onChange={setTask}
            options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
          <Chips label="Test design" value={split} onChange={setSplit}
            options={{ grouped: DESIGNS.grouped.short, temporal: DESIGNS.temporal.short }} />
        </div>
        <div className="grid-2">
          <figure className="card">
            <Legend items={[
              { label: SITE.brand, color: ROLE_COLOR.ours },
              { label: "LightGBM (strongest standard model)", color: ROLE_COLOR.reference },
              { label: "Other models", color: ROLE_COLOR.other },
            ]} />
            <ForestChart rows={forest} domain={domain} ticks={ticks} ariaLabel="AUROC by model" />
            <figcaption>
              Mean AUROC with one standard deviation, {DESIGNS[split].long.toLowerCase()}.
            </figcaption>
          </figure>
          <div className="card">
            <div className="table-wrap">
              <table>
                <thead><tr><th>Model</th><th className="num">AUROC</th><th>Compared with {SITE.brand}</th></tr></thead>
                <tbody>
                  {Object.entries(FAMILIES).map(([fam, title]) => (
                    <React.Fragment key={fam}>
                      <tr className="group-row"><td colSpan={3}>{title}</td></tr>
                      {MODEL_ORDER.filter((m) => MODELS[m].family === fam && byModel[m]).map((m) => {
                        const v = m === "tkgn_b" ? null : verdict(sig, m);
                        const w = wins(m);
                        return (
                          <tr key={m} className={m === "tkgn_b" ? "highlight" : ""}>
                            <td>{MODELS[m].name}</td>
                            <td className="num">{num(byModel[m].auroc)}</td>
                            <td className="small">
                              {m === "tkgn_b" ? "Reference" : v && <>
                                <span className={v.cls}>{v.text}</span>{" "}
                                <span className="muted">({pText(v.p)}{w ? `, ahead in ${w.wins}/${w.n_repeats}` : ""})</span>
                              </>}
                            </td>
                          </tr>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small muted" style={{ marginTop: 10 }}>
              Patient-clustered bootstrap on the first test set
              ({count(sp[`${task}_${split}`].n.test)} stays).
            </p>
          </div>
        </div>
      </Block>

      <Block eyebrow="Calibration" title="Do the percentages mean what they say?">
        <div className="grid-2">
          <figure className="card">
            <Legend items={[...calSeries.map((s) => ({ label: s.label, color: s.color, line: true })),
              { label: "Perfect agreement", color: "var(--rule-strong)", line: true }]} />
            <LineChart series={calSeries} diagonal xDomain={[0, calMax]} yDomain={[0, calMax]}
              xTicks={calTicks} yTicks={calTicks} xFormat={(v) => `${Math.round(v * 100)}%`}
              yFormat={(v) => `${Math.round(v * 100)}%`} xLabel="Predicted risk" yLabel="Observed rate"
              height={300} ariaLabel="Calibration" />
            <figcaption>Test stays in ten equal groups by predicted risk ({OUTCOMES[task].short.toLowerCase()}).</figcaption>
          </figure>
          <div className="card">
            <h3>Risk bands used in the risk check</h3>
            <table>
              <thead><tr><th>Band</th><th className="num">Predicted risk</th><th className="num">Share of patients</th><th className="num">Observed rate</th></tr></thead>
              <tbody>
                {band.bands.map((b) => (
                  <tr key={b.band}>
                    <td style={{ textTransform: "capitalize" }}>{b.band}</td>
                    <td className="num">{b.band === "higher" ? `≥ ${pct(b.from)}` : `${pct(b.from)} to ${pct(b.to)}`}</td>
                    <td className="num">{pct(b.share, 0)}</td>
                    <td className="num"><strong>{pct(b.observed_rate)}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small" style={{ marginTop: 12 }}>
              Expected calibration error was{" "}
              {num(data.summary.find((r) => r.task === task && r.split === "grouped" && r.model === "tkgn_b").ece)}{" "}
              for {SITE.brand}. Overall rate in the test set: {pct(band.overall_rate)}.
            </p>
          </div>
        </div>
      </Block>

      <Block eyebrow="Fairness" title="Performance across patient groups" tint>
        <div className="card">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Group</th><th className="num">Test stays</th><th className="num">Observed rate</th><th className="num">Mean predicted</th><th className="num">AUROC</th></tr></thead>
              <tbody>
                {sub.map((r) => (
                  <tr key={`${r.attribute}${r.group}`}>
                    <td>{subLabel(r)}</td>
                    <td className="num">{count(r.n)}</td>
                    <td className="num">{pct(r.prevalence)}</td>
                    <td className="num">{pct(r.mean_predicted)}</td>
                    <td className="num">{num(r.auroc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted" style={{ marginTop: 10 }}>
            {OUTCOMES[task].short}, first test set; groups with fewer than 300 stays are not shown.
          </p>
        </div>
      </Block>

      <Block eyebrow="Over time" title="Tested on the newest admissions">
        <div className="grid-2">
          <div className="card">
            <p>
              Hospitals recorded data differently in later years. Training on the
              oldest 70% of admissions and testing on the newest 20% exposes this.
            </p>
            <table>
              <thead><tr><th>Measure</th><th className="num">Older</th><th className="num">Newest</th></tr></thead>
              <tbody>
                <tr><td>Most diagnoses on one stay</td><td className="num">{data.shift.train.number_diagnoses_max}</td><td className="num">{data.shift.test.number_diagnoses_max}</td></tr>
                <tr><td>Stays with an earlier stay</td><td className="num">{pct(data.shift.train.share_with_history)}</td><td className="num">{pct(data.shift.test.share_with_history)}</td></tr>
                <tr><td>Emergency visits, prior year</td><td className="num">{num(data.shift.train.number_emergency_mean, 2)}</td><td className="num">{num(data.shift.test.number_emergency_mean, 2)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <p>
              For treatment escalation {SITE.brand} was the most accurate model
              on the newest admissions. For readmission a random forest coped
              best with the shift, and {SITE.brand} matched LightGBM and XGBoost.
              Logistic regression and the simple neural network lost the most
              accuracy. Choose &quot;Newest admissions&quot; above to see every model.
            </p>
          </div>
        </div>
      </Block>

      <Block eyebrow="Ablation" title="What each part of the model contributes" tint>
        <figure className="card">
          <BarList rows={ablRows}
            format={(v) => (Math.abs(v) < 0.0005 ? "0.000" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`)}
            ariaLabel="Contribution of each part" />
          <figcaption>
            AUROC gained from each part ({OUTCOMES[task].short.toLowerCase()}, three repeats); bars
            to the left mean the model did slightly better without it. Differences under about
            0.004 are within run-to-run noise.
          </figcaption>
        </figure>
      </Block>

      <Block eyebrow="Limitations" title="What to keep in mind">
        <div className="grid-2">
          <ul className="card" style={{ paddingLeft: 44, margin: 0 }}>
            <li>One public data source; external validation in another health system is still needed.</li>
            <li>The records have no calendar dates, so the time between stays is unknown.</li>
            <li>Only three diagnoses per stay are recorded, mostly as three-character ICD-9 codes.</li>
          </ul>
          <ul className="card" style={{ paddingLeft: 44, margin: 0 }}>
            <li>Readmissions to hospitals outside the network are not seen.</li>
            <li>The gain over gradient boosting is small and not statistically significant on a single test set.</li>
            <li>Not evaluated for clinical benefit; not a medical device.</li>
          </ul>
        </div>
      </Block>
    </>
  );
}
