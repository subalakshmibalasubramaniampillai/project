import React, { useState } from "react";
import { predict } from "../api";
import { Icon } from "../components/ui";
import { COMMON_DIAGNOSES, diagnosisName, featureName, pct } from "../format";
import { SITE } from "../site";
import { useApi } from "../useFetch";

const AGES = ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"];
const ageLabel = (a) => (a || "").replace("[", "").replace(")", "").replace("-", " to ");
const ADMISSION_TYPE = { 1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma centre" };
const ADMISSION_SOURCE = { 7: "Emergency room", 1: "Physician referral", 2: "Clinic referral",
  4: "Transfer from another hospital", 6: "Transfer from another facility" };
const DISCHARGE = { 1: "Home", 6: "Home with home health care", 3: "Skilled nursing facility",
  22: "Rehabilitation facility", 2: "Another short-stay hospital", 23: "Long-term care hospital",
  5: "Other inpatient care", 7: "Left against medical advice" };
const DRUG = { No: "Not prescribed", Steady: "Steady dose", Up: "Dose increased", Down: "Dose reduced" };
const AFTER = { NO: "Not readmitted", ">30": "Readmitted after 30 days", "<30": "Readmitted within 30 days" };

const blankStay = () => ({
  age: "[60-70)", gender: "Female", race: "Caucasian",
  admission_type_id: "1", admission_source_id: "7", discharge_disposition_id: "1",
  time_in_hospital: 4, number_diagnoses: 7, num_lab_procedures: 43, num_procedures: 1,
  num_medications: 16, number_inpatient: 0, number_emergency: 0, number_outpatient: 0,
  diag_1: "428", diag_2: "250", diag_3: "401",
  A1Cresult: "", max_glu_serum: "", insulin: "Steady", metformin: "No",
  change: "No", diabetesMed: "Yes", readmitted: ">30",
});

const OUTCOME = {
  readmit30: {
    title: "Readmission within 30 days",
    happened: "were readmitted within 30 days",
    scale: 0.4,
  },
  escalation: {
    title: "Diabetes treatment stepped up at the next stay",
    happened: "had their treatment stepped up at the next stay",
    scale: 0.8,
  },
};

const BAND_TEXT = { lower: "Lower", moderate: "Moderate", higher: "Higher" };

function bandOf(info, p) {
  if (p >= info.cut_higher) return "higher";
  if (p >= info.cut_moderate) return "moderate";
  return "lower";
}

function Select({ label, value, onChange, options }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        {Object.entries(options).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
    </div>
  );
}

function NumberField({ label, value, onChange, min = 0, max = 100 }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" min={min} max={max} value={value ?? ""}
        onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function DiagnosisField({ label, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input list="dx-list" value={value ?? ""} placeholder="ICD-9 code"
        onChange={(e) => onChange(e.target.value)} />
      <div className="hint">{diagnosisName(value) || "Type a code, for example 428"}</div>
    </div>
  );
}

function driverText(c) {
  const name = featureName(c.feature);
  const lastStay = {
    hist_last_outcome_lt30: ["Last stay was followed by readmission within 30 days",
      "Last stay was not followed by readmission within 30 days"],
    hist_last_outcome_gt30: ["Last stay was followed by readmission after 30 days",
      "Last stay was not followed by readmission after 30 days"],
    hist_last_outcome_no: ["Last stay was not followed by readmission",
      "Last stay was followed by a readmission"],
  }[c.feature];
  if (lastStay) return c.value ? lastStay[0] : lastStay[1];
  if (c.feature.includes("=")) return c.value ? name : `${name}: no`;
  return `${name}: ${Number.isInteger(c.value) ? c.value : c.value.toFixed(1)}`;
}

function RiskCard({ task, result, info }) {
  const o = OUTCOME[task];
  const p = result.probability;
  const band = bandOf(info, p);
  const bandInfo = info.bands.find((b) => b.band === band);
  const toPos = (v) => `${Math.min(100, (v / o.scale) * 100)}%`;
  const reasons = result.top_contributions.filter((c) => Math.abs(c.shap) >= 0.02).slice(0, 4);
  return (
    <div className="result-card">
      <div className="risk-title">{o.title}</div>
      <div>
        <span className="risk-value">{pct(p)}</span>
        <span className={`risk-badge ${band}`}>
          <Icon name={band === "lower" ? "down" : "up"} size={14} />
          {BAND_TEXT[band]} risk
        </span>
      </div>
      <div className="gauge" aria-hidden="true">
        <span style={{ width: toPos(info.cut_moderate), background: "var(--risk-low)" }} />
        <span style={{ width: `calc(${toPos(info.cut_higher)} - ${toPos(info.cut_moderate)})`, background: "var(--risk-mid)" }} />
        <span style={{ flex: 1, background: "var(--risk-high)" }} />
      </div>
      <div className="gauge-marker"><i style={{ left: toPos(p) }} /></div>
      <div className="gauge-scale"><span>0%</span><span>{Math.round(o.scale * 100)}%</span></div>
      <p className="small" style={{ marginTop: 12 }}>
        Among {SITE.brand}&apos;s test patients in the {band} band,{" "}
        <strong>{pct(bandInfo.observed_rate)}</strong> {o.happened}. Across all
        test patients the rate was {pct(info.overall_rate)}.
      </p>
      {reasons.length > 0 && (
        <>
          <h4 style={{ marginTop: 16 }}>Main reasons</h4>
          <ul className="reason-list">
            {reasons.map((c) => (
              <li key={c.feature}>
                <span>{driverText(c)}</span>
                <span className={`dir ${c.shap > 0 ? "bad" : "good"}`}>
                  {c.shap > 0 ? "raises risk" : "lowers risk"}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default function Check() {
  const { data } = useApi({ bands: "/risk-bands", examples: "/examples" });
  const [current, setCurrent] = useState(blankStay());
  const [history, setHistory] = useState([]);
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [example, setExample] = useState("");

  const set = (key) => (v) => { setCurrent({ ...current, [key]: v }); setResults(null); };
  const shared = { age: current.age, gender: current.gender, race: current.race };

  function loadExample(id) {
    setExample(id);
    setResults(null);
    if (!id) return;
    const ex = data.examples.find((e) => String(e.patient_nbr) === id);
    const stays = ex.stays.map((s) => ({ ...blankStay(), ...Object.fromEntries(
      Object.entries(s).filter(([, v]) => v !== null && v !== "None")) }));
    setHistory(stays.slice(0, -1));
    setCurrent(stays[stays.length - 1]);
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const stays = [...history.map((h) => ({ ...h, ...shared })), { ...current, readmitted: "NO" }];
      const [r, e] = await Promise.all([predict("readmit30", stays), predict("escalation", stays)]);
      setResults({ readmit30: r, escalation: e });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap page">
      <span className="eyebrow">Discharge risk check</span>
      <h1 style={{ fontSize: "clamp(32px, 4vw, 44px)" }}>Check a patient before discharge</h1>
      <p className="lead">
        Enter today&apos;s stay and any earlier stays. {SITE.brand} returns the
        chance of readmission within 30 days and of needing stronger diabetes
        treatment at the next stay.
      </p>
      <div className="note warn no-print" style={{ marginBottom: 26 }}>
        <p>
          Research prototype. It supports, and does not replace, clinical
          judgement, and it has not been approved for patient care.
        </p>
      </div>

      <datalist id="dx-list">
        {COMMON_DIAGNOSES.map((d) => <option key={d.code} value={d.code}>{d.name}</option>)}
      </datalist>

      <div className="tool">
        <div className="tool-form">
          {data?.examples && (
            <div className="form-card" style={{ paddingTop: 18 }}>
              <div className="field">
                <label>Start from a real example patient (from the test data)</label>
                <select value={example} onChange={(e) => loadExample(e.target.value)}>
                  <option value="">Choose an example…</option>
                  {data.examples.map((e, i) => (
                    <option key={e.patient_nbr} value={e.patient_nbr}>
                      Example {String.fromCharCode(65 + i)}: {e.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          <div className="form-card">
            <h3><span className="step-dot">1</span>Patient</h3>
            <div className="form-grid">
              <Select label="Age (years)" value={current.age} onChange={set("age")}
                options={Object.fromEntries(AGES.map((a) => [a, ageLabel(a)]))} />
              <Select label="Sex" value={current.gender} onChange={set("gender")}
                options={{ Female: "Female", Male: "Male" }} />
              <Select label="Race" value={current.race} onChange={set("race")}
                options={{ Caucasian: "Caucasian", AfricanAmerican: "African American",
                  Hispanic: "Hispanic", Asian: "Asian", Other: "Other" }} />
            </div>

            <h3><span className="step-dot">2</span>Today&apos;s stay</h3>
            <div className="form-grid">
              <Select label="Type of admission" value={current.admission_type_id}
                onChange={set("admission_type_id")} options={ADMISSION_TYPE} />
              <Select label="Admitted from" value={current.admission_source_id}
                onChange={set("admission_source_id")} options={ADMISSION_SOURCE} />
              <Select label="Going to after discharge" value={current.discharge_disposition_id}
                onChange={set("discharge_disposition_id")} options={DISCHARGE} />
              <NumberField label="Days in hospital" value={current.time_in_hospital}
                onChange={set("time_in_hospital")} min={1} max={14} />
              <NumberField label="Medications given" value={current.num_medications}
                onChange={set("num_medications")} />
              <NumberField label="Laboratory tests" value={current.num_lab_procedures}
                onChange={set("num_lab_procedures")} max={140} />
              <NumberField label="Diagnoses recorded" value={current.number_diagnoses}
                onChange={set("number_diagnoses")} min={1} max={16} />
            </div>

            <h3><span className="step-dot">3</span>Diagnoses</h3>
            <div className="form-grid">
              <DiagnosisField label="Main diagnosis" value={current.diag_1} onChange={set("diag_1")} />
              <DiagnosisField label="Second diagnosis" value={current.diag_2} onChange={set("diag_2")} />
              <DiagnosisField label="Third diagnosis" value={current.diag_3} onChange={set("diag_3")} />
            </div>

            <h3><span className="step-dot">4</span>Diabetes care</h3>
            <div className="form-grid">
              <Select label="HbA1c result" value={current.A1Cresult} onChange={set("A1Cresult")}
                options={{ "": "Not measured", Norm: "Normal", ">7": "Above 7%", ">8": "Above 8%" }} />
              <Select label="Insulin" value={current.insulin} onChange={set("insulin")} options={DRUG} />
              <Select label="Metformin" value={current.metformin} onChange={set("metformin")} options={DRUG} />
              <Select label="Diabetes medicines changed" value={current.change} onChange={set("change")}
                options={{ No: "No", Ch: "Yes" }} />
            </div>

            <h3><span className="step-dot">5</span>In the 12 months before</h3>
            <div className="form-grid">
              <NumberField label="Hospital admissions" value={current.number_inpatient}
                onChange={set("number_inpatient")} />
              <NumberField label="Emergency visits" value={current.number_emergency}
                onChange={set("number_emergency")} />
              <NumberField label="Outpatient visits" value={current.number_outpatient}
                onChange={set("number_outpatient")} />
            </div>
          </div>

          <div className="form-card">
            <h3><span className="step-dot">6</span>Earlier stays of this patient</h3>
            <p className="small muted">
              Oldest first. These are what make {SITE.brand} different from a
              one-stay score; leave empty for a first admission.
            </p>
            {history.map((h, i) => {
              const upd = (key) => (v) => {
                setHistory(history.map((x, j) => (j === i ? { ...x, [key]: v } : x)));
                setResults(null);
              };
              return (
                <div className="stay-row" key={i}>
                  <div className="stay-row-head">
                    <span>Earlier stay {i + 1}{i === history.length - 1 ? " (most recent)" : ""}</span>
                    <button type="button" className="btn text small"
                      onClick={() => { setHistory(history.filter((_, j) => j !== i)); setResults(null); }}>
                      Remove
                    </button>
                  </div>
                  <div className="form-grid">
                    <DiagnosisField label="Main diagnosis" value={h.diag_1} onChange={upd("diag_1")} />
                    <NumberField label="Days in hospital" value={h.time_in_hospital}
                      onChange={upd("time_in_hospital")} min={1} max={14} />
                    <Select label="Insulin" value={h.insulin} onChange={upd("insulin")} options={DRUG} />
                    <Select label="Went to" value={h.discharge_disposition_id}
                      onChange={upd("discharge_disposition_id")} options={DISCHARGE} />
                    <Select label="What happened next" value={h.readmitted}
                      onChange={upd("readmitted")} options={AFTER} />
                  </div>
                </div>
              );
            })}
            <button type="button" className="btn ghost small"
              onClick={() => { setHistory([...history, { ...blankStay(), ...shared }]); setResults(null); }}>
              + Add an earlier stay
            </button>
          </div>

          <button type="button" className="btn" onClick={run} disabled={busy || !data?.bands}>
            {busy ? "Calculating…" : "Calculate risk"}
          </button>
          {error && <div className="error">{error}</div>}
        </div>

        <aside className="tool-result" aria-live="polite">
          {!results && (
            <div className="result-card empty">
              <p style={{ margin: "0 auto" }}>
                Fill in the form, or start from an example patient, then press
                <strong> Calculate risk</strong>. Both risks appear here.
              </p>
            </div>
          )}
          {results && data?.bands && (
            <>
              <div className="result-card" style={{ paddingBottom: 16 }}>
                <div className="risk-title">Patient summary</div>
                <p className="small" style={{ marginBottom: 6 }}>
                  {current.gender === "Female" ? "Woman" : "Man"} aged {ageLabel(current.age)},{" "}
                  {current.time_in_hospital} days in hospital, main diagnosis{" "}
                  {current.diag_1} ({diagnosisName(current.diag_1).toLowerCase() || "not recognised"}).
                </p>
                <div className="timeline-mini" aria-label="Stays used">
                  {history.map((h, i) => (
                    <React.Fragment key={i}>
                      <span className={`dot${h.readmitted === "<30" ? " event" : ""}`}
                        title={`Earlier stay ${i + 1}: ${AFTER[h.readmitted]}`} />
                      <span className="line" />
                    </React.Fragment>
                  ))}
                  <span className="dot now" title="Today" />
                </div>
                <div className="small muted">
                  {history.length
                    ? `${history.length} earlier stay${history.length > 1 ? "s" : ""} used. Filled circle: readmitted within 30 days.`
                    : "First admission: no earlier stays used."}
                </div>
              </div>
              <RiskCard task="readmit30" result={results.readmit30} info={data.bands.readmit30} />
              <RiskCard task="escalation" result={results.escalation} info={data.bands.escalation} />
              <div className="no-print" style={{ display: "flex", gap: 10 }}>
                <button type="button" className="btn ghost small" onClick={() => window.print()}>
                  <Icon name="print" size={16} /> Print summary
                </button>
              </div>
              <p className="small muted" style={{ marginTop: 14 }}>
                Reasons come from the boosting part of the model (SHAP values).
                The treatment step-up risk applies if the patient is admitted again.
              </p>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
