import React, { useState } from "react";
import { Link } from "react-router-dom";
import { PAPER_URL } from "../api";
import { ForestChart } from "../components/charts";
import { ErrorMsg, Legend, Loading, Section, Segmented } from "../components/ui";
import {
  MODELS, MODEL_ORDER, OUTCOMES, ROLE_COLOR, count, num, roleOf,
} from "../format";
import { useApi } from "../useFetch";

function pick(rows, task, split, model) {
  return rows.find((r) => r.task === task && r.split === split && r.model === model);
}

export default function Summary() {
  const { data, error, loading } = useApi({
    overview: "/overview",
    summary: "/summary",
    repeats: "/repeat-tests",
    ablation: "/ablation",
    kg: "/kg-efficiency",
  });
  const [task, setTask] = useState("readmit30");
  if (loading) return <Loading />;
  if (error) return <ErrorMsg error={error} />;

  const d = data.overview.dataset;
  const s = data.summary;
  const ours = (t, sp = "grouped") => pick(s, t, sp, "tkgn_b");
  const ref = (t, sp = "grouped") => pick(s, t, sp, "lightgbm");
  const wins = (t, m) =>
    data.repeats.find((r) => r.task === t && r.split === "grouped"
      && r.reference === "tkgn_b" && r.model === m);
  const abl = (t, m) => data.ablation.find((r) => r.task === t && r.model === m);
  const kgMin = Math.min(...data.kg.filter((r) => r.fraction < 1).map((r) => r.delta_auroc));
  const kgMax = Math.max(...data.kg.filter((r) => r.fraction < 1).map((r) => r.delta_auroc));
  const seqAlwaysBehind = ["gru", "retain", "transformer"].every((m) =>
    ["readmit30", "escalation"].every((t) => wins(t, m)?.wins === wins(t, m)?.n_repeats));

  const rows = MODEL_ORDER
    .map((m) => pick(s, task, "grouped", m))
    .filter(Boolean)
    .sort((a, b) => b.auroc - a.auroc)
    .map((r) => ({
      key: r.model,
      label: MODELS[r.model].name + (r.model === "tkgn_b" ? " (this project)" : ""),
      value: r.auroc,
      lo: r.auroc - r.auroc_sd,
      hi: r.auroc + r.auroc_sd,
      color: ROLE_COLOR[roleOf(r.model)],
      bold: r.model === "tkgn_b",
    }));

  const gap = (t) => ours(t).auroc - ref(t).auroc;

  return (
    <>
      <div className="intro">
        <div className="kicker">Research summary</div>
        <h1>Do earlier hospital stays improve risk prediction in diabetes?</h1>
        <p className="lead">
          Many people with diabetes are admitted to hospital more than once.
          This project asks whether a patient&apos;s earlier stays improve two
          predictions made at discharge: will the patient be readmitted within
          30 days, and will their diabetes treatment have to be stepped up at
          the next stay? Every result comes from {count(d.cohort_encounters)} real,
          de-identified stays at 130 hospitals in the United States.
        </p>
        <p className="small">
          <a href={PAPER_URL}>Read the manuscript (PDF)</a>
          <span className="muted"> · </span>
          <Link to="/patients">Try the model on a patient</Link>
        </p>
      </div>

      <div className="figures">
        <div className="figure-cell">
          <div className="figure-value">{count(d.cohort_encounters)}</div>
          <div className="figure-label">hospital stays from {count(d.patients)} patients</div>
          <div className="figure-compare">
            {count(d.patients_with_repeat_encounters)} patients were admitted more than once
          </div>
        </div>
        <div className="figure-cell">
          <div className="figure-value">{num(ours("readmit30").auroc)}</div>
          <div className="figure-label">AUROC of our model for 30-day readmission</div>
          <div className="figure-compare">LightGBM, the strongest standard model: {num(ref("readmit30").auroc)}</div>
        </div>
        <div className="figure-cell">
          <div className="figure-value">{num(ours("escalation", "temporal").auroc)}</div>
          <div className="figure-label">AUROC for treatment escalation on the newest admissions</div>
          <div className="figure-compare">LightGBM: {num(ref("escalation", "temporal").auroc)}</div>
        </div>
      </div>
      <p className="small muted" style={{ marginTop: 10 }}>
        AUROC is the chance that a model gives a higher risk to a patient who had
        the outcome than to one who did not. 0.5 is a coin toss and 1.0 is perfect.
      </p>

      <Section num="1" title="How well each model ranks patients by risk">
        <div className="controls">
          <Segmented label="Outcome" value={task} onChange={setTask}
            options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
        </div>
        <figure>
          <Legend items={[
            { label: "TKGN-B, developed in this project", color: ROLE_COLOR.ours },
            { label: "LightGBM, strongest standard model", color: ROLE_COLOR.reference },
            { label: "Other models", color: ROLE_COLOR.other },
          ]} />
          <ForestChart rows={rows} domain={[0.635, 0.705]}
            ticks={[0.64, 0.66, 0.68, 0.70]} format={(v) => v.toFixed(3)}
            tickFormat={(v) => v.toFixed(2)} ariaLabel="AUROC by model" />
          <figcaption>
            {OUTCOMES[task].question} Average AUROC on patients the models never
            saw, over five repeats; the line spans one standard deviation either
            side. Higher is better.
          </figcaption>
        </figure>
      </Section>

      <Section num="2" title="What we found">
        <div className="findings">
          <div className="finding">
            <h3>Our model is the most accurate, by a small margin</h3>
            <p>
              TKGN-B had the highest average AUROC for both outcomes. Its lead
              over LightGBM was {num(gap("readmit30"))} for readmission
              and {num(gap("escalation"))} for treatment escalation. Differences
              this small are not statistically significant, so the fair reading
              is that TKGN-B matches the best standard models and edges ahead of them.
            </p>
          </div>
          <div className="finding">
            <h3>It beats other models that read the visit history</h3>
            <p>
              GRU, RETAIN and the Transformer also use a patient&apos;s earlier
              stays.{" "}
              {seqAlwaysBehind
                ? "TKGN-B scored higher than each of them in every one of the five repeats, for both outcomes."
                : "TKGN-B scored higher than each of them in most repeats."}
            </p>
          </div>
          <div className="finding">
            <h3>Earlier stays carry the most useful extra information</h3>
            <p>
              Removing the patient&apos;s earlier stays from the model lowered
              AUROC by {num(-abl("readmit30", "tkgn_no_history").delta_auroc_vs_full)} for
              readmission and {num(-abl("escalation", "tkgn_no_history").delta_auroc_vs_full)} for
              treatment escalation, the largest loss of any part we removed.
            </p>
          </div>
          <div className="finding">
            <h3>The diagnosis-code graph did not help</h3>
            <p>
              Linking diagnosis codes through the ICD-9 hierarchy and a
              co-morbidity graph changed AUROC by between {num(kgMin)} and{" "}
              {num(kgMax)}, even when only 5 to 25% of the training data were
              used. We report this as a negative result.
            </p>
          </div>
        </div>
      </Section>

      <Section num="3" title="Where to go next">
        <div className="next-links">
          <div className="finding">
            <h3><Link to="/results">Results</Link></h3>
            <p>Every model on both outcomes, with significance tests and the factors that drive the predictions.</p>
          </div>
          <div className="finding">
            <h3><Link to="/method">How it works</Link></h3>
            <p>The data, the two outcomes, how the model is built and how it was tested.</p>
          </div>
          <div className="finding">
            <h3><Link to="/reliability">Reliability</Link></h3>
            <p>Whether the risks are well calibrated, fair across patient groups and stable over time.</p>
          </div>
        </div>
      </Section>
    </>
  );
}
