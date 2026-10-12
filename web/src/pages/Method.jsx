import React, { useState } from "react";
import { BarList } from "../components/charts";
import { ErrorMsg, Loading, Section, Segmented } from "../components/ui";
import { OUTCOMES, count, num, pct } from "../format";
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

const STEPS = [
  {
    title: "Describe each stay",
    text: "Diagnoses, diabetes medicines, laboratory results, admission and discharge details and hospital use are turned into one summary for the stay. Each ICD-9 diagnosis code also borrows information from its parent groups, so rare codes are not left without data.",
  },
  {
    title: "Read the history",
    text: "Up to ten earlier stays of the same patient are read in order, together with whether each of them led to a readmission. The model also looks at what changed since the last stay, for example a new insulin prescription.",
  },
  {
    title: "Decide how much to trust it",
    text: "A learned gate weighs the history against the current stay, separately for each patient, taking into account how many earlier stays there are. With no history the model relies on the current stay alone.",
  },
  {
    title: "Add to a strong baseline",
    text: "The final model, TKGN-B, adds this history-based correction to the score of a LightGBM model trained on the same information. The network only has to learn what LightGBM misses.",
  },
];

export default function Method() {
  const { data, error, loading } = useApi({
    overview: "/overview",
    ablation: "/ablation",
    graph: "/graph",
  });
  const [task, setTask] = useState("readmit30");
  if (loading) return <Loading />;
  if (error) return <ErrorMsg error={error} />;

  const d = data.overview.dataset;
  const sp = data.overview.splits;
  const excluded = d.raw_encounters - d.cohort_encounters;
  const abl = data.ablation
    .filter((r) => r.task === task && PARTS[r.model])
    .map((r) => ({
      label: PARTS[r.model],
      value: -r.delta_auroc_vs_full,
      tip: r.delta_auroc_vs_full < 0
        ? `Removing it lowers AUROC by ${num(-r.delta_auroc_vs_full)}`
        : `Removing it raises AUROC by ${num(r.delta_auroc_vs_full)}`,
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <>
      <div className="intro">
        <div className="kicker">Method</div>
        <h1>How the study was done</h1>
        <p className="lead">
          The data are real hospital records, the model reads each patient&apos;s
          history in order, and every model was tested on patients it had never seen.
        </p>
      </div>

      <Section num="1" title="The data">
        <p>
          The Diabetes 130-US Hospitals database covers inpatient stays of
          people with diabetes at 130 US hospitals between 1999 and 2008. It is
          published by the UCI Machine Learning Repository under a CC BY 4.0
          licence and was first analysed by Strack and colleagues in 2014. The
          file is used exactly as released; nothing was simulated or filled in.
          A patient number links repeat stays of the same person, which is what
          makes a history-based model possible.
        </p>
        <div className="figures">
          <div className="figure-cell">
            <div className="figure-value">{count(d.raw_encounters)}</div>
            <div className="figure-label">stays in the public file</div>
          </div>
          <div className="figure-cell">
            <div className="figure-value">{count(d.cohort_encounters)}</div>
            <div className="figure-label">stays studied, after removing {count(excluded)} that ended in death or hospice care</div>
          </div>
          <div className="figure-cell">
            <div className="figure-value">{count(d.encounters_with_history)}</div>
            <div className="figure-label">stays with at least one earlier stay of the same patient</div>
          </div>
        </div>
      </Section>

      <Section num="2" title="The two questions">
        <div className="two-col">
          <div>
            <h3>30-day readmission</h3>
            <p>
              Asked at every discharge: is the patient admitted again within 30
              days? This happened after {pct(d.readmit30_rate)} of the{" "}
              {count(d.cohort_encounters)} stays.
            </p>
          </div>
          <div>
            <h3>Treatment escalation at the next stay</h3>
            <p>
              Asked for the {count(d.escalation_samples)} stays that were followed
              by another stay: at that next stay, was insulin started, a dose
              raised, or another diabetes drug added? This happened in{" "}
              {pct(d.escalation_rate)} of cases.
            </p>
          </div>
        </div>
        <p className="muted small">
          Only information available at discharge is used. The outcome of the
          current stay is never shown to any model.
        </p>
      </Section>

      <Section num="3" title="The model in four steps">
        <ol className="flow">
          {STEPS.map((s, i) => (
            <li key={s.title} className="flow-step">
              <div className="flow-num">{i + 1}</div>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
            </li>
          ))}
        </ol>
        <p className="muted small">
          TKGN stands for Temporal Knowledge-Gated Network; TKGN-B is TKGN on top
          of the boosting (LightGBM) score. The diagnosis graph has{" "}
          {count(data.graph.n_nodes)} concepts and {count(data.graph.n_cooccurrence_edges)}{" "}
          co-morbidity links, all learned from training stays only.
        </p>
      </Section>

      <Section num="4" title="How the models were tested">
        <div className="two-col">
          <div>
            <h3>Patients the models never saw</h3>
            <p>
              Patients, not single stays, were split into training (70%),
              validation (10%) and test (20%) groups, so no person appears in
              more than one group. This was repeated five times with different
              splits. In the first repeat the readmission test set held{" "}
              {count(sp.readmit30_grouped.n.test)} stays.
            </p>
          </div>
          <div>
            <h3>The newest admissions</h3>
            <p>
              Models were trained on the oldest 70% of admissions, tuned on the
              next 10% and tested on the most recent 20%. This is closer to how a
              hospital would use the model, and it exposes changes in the data
              over time.
            </p>
          </div>
          <div>
            <h3>Fair comparison</h3>
            <p>
              Five standard models and three published history-reading models
              received the same information as TKGN-B, including the earlier
              stays. All settings were chosen on validation data only.
            </p>
          </div>
        </div>
      </Section>

      <Section num="5" title="What each part of the model contributes">
        <div className="controls">
          <Segmented label="Outcome" value={task} onChange={setTask}
            options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
        </div>
        <figure>
          <BarList rows={abl} format={(v) => (Math.abs(v) < 0.0005 ? "0.000" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`)}
            ariaLabel="Contribution of each model part" />
          <figcaption>
            Each part was removed in turn and the model retrained (three repeats).
            Bars to the right show how much AUROC the part adds; bars to the left
            mean the model did slightly better without it. Differences under about
            0.004 are within run-to-run noise.
          </figcaption>
        </figure>
        <div className="note" style={{ marginTop: 18 }}>
          <p>
            The patient&apos;s earlier stays are the part that matters. The
            diagnosis hierarchy and co-morbidity graph did not improve accuracy,
            and a separate test with only 5 to 25% of the training data gave the
            same answer.
          </p>
        </div>
      </Section>
    </>
  );
}
