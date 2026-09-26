import React, { useState } from "react";
import { useFetch } from "../useFetch";
import { getSchema, predict } from "../api";
import { ErrorBox, Loading, Selector, TASKS, fmt, pct } from "../common";

const AGES = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
  "[60-70)", "[70-80)", "[80-90)", "[90-100)"];

const NUMERIC = [
  ["time_in_hospital", "Days in hospital"],
  ["num_lab_procedures", "Lab procedures"],
  ["num_procedures", "Procedures"],
  ["num_medications", "Medications"],
  ["number_outpatient", "Outpatient visits (prior yr)"],
  ["number_emergency", "Emergency visits (prior yr)"],
  ["number_inpatient", "Inpatient stays (prior yr)"],
  ["number_diagnoses", "Number of diagnoses"],
];

const blank = () => ({
  age: "[60-70)", gender: "Female", race: "Caucasian",
  admission_type_id: 1, discharge_disposition_id: 1, admission_source_id: 7,
  time_in_hospital: 4, num_lab_procedures: 43, num_procedures: 1,
  num_medications: 16, number_outpatient: 0, number_emergency: 0,
  number_inpatient: 0, number_diagnoses: 7,
  diag_1: "428", diag_2: "250", diag_3: "401",
  A1Cresult: "", max_glu_serum: "", insulin: "Steady", metformin: "No",
  change: "No", diabetesMed: "Yes", readmitted: ">30",
});

function Field({ label, children }) {
  return <div className="field"><label>{label}</label>{children}</div>;
}

function EncounterForm({ value, onChange, isIndex, schema }) {
  const set = (k) => (e) => onChange({ ...value, [k]: e.target.value });
  const opts = (col) => schema.categorical[col] || [];
  return (
    <div className="form-grid">
      <Field label="Age band">
        <select value={value.age} onChange={set("age")}>
          {AGES.map((a) => <option key={a}>{a}</option>)}
        </select>
      </Field>
      <Field label="Gender">
        <select value={value.gender} onChange={set("gender")}>
          <option>Female</option><option>Male</option>
        </select>
      </Field>
      <Field label="Race">
        <select value={value.race} onChange={set("race")}>
          {opts("race").map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Admission type id">
        <select value={value.admission_type_id} onChange={set("admission_type_id")}>
          {opts("admission_type_id").map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Discharge disposition id">
        <select value={value.discharge_disposition_id} onChange={set("discharge_disposition_id")}>
          {opts("discharge_disposition_id").map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      <Field label="Admission source id">
        <select value={value.admission_source_id} onChange={set("admission_source_id")}>
          {opts("admission_source_id").map((r) => <option key={r}>{r}</option>)}
        </select>
      </Field>
      {NUMERIC.map(([k, label]) => (
        <Field key={k} label={label}>
          <input type="number" min="0" value={value[k]} onChange={set(k)} />
        </Field>
      ))}
      {["diag_1", "diag_2", "diag_3"].map((k, i) => (
        <Field key={k} label={`Diagnosis ${i + 1} (ICD-9)`}>
          <input list="dx-codes" value={value[k]} onChange={set(k)} />
        </Field>
      ))}
      <Field label="HbA1c result">
        <select value={value.A1Cresult} onChange={set("A1Cresult")}>
          <option value="">Not measured</option><option>Norm</option>
          <option>&gt;7</option><option>&gt;8</option>
        </select>
      </Field>
      <Field label="Max glucose serum">
        <select value={value.max_glu_serum} onChange={set("max_glu_serum")}>
          <option value="">Not measured</option><option>Norm</option>
          <option>&gt;200</option><option>&gt;300</option>
        </select>
      </Field>
      {["insulin", "metformin"].map((k) => (
        <Field key={k} label={k[0].toUpperCase() + k.slice(1)}>
          <select value={value[k]} onChange={set(k)}>
            {schema.drug_states.map((s) => <option key={s}>{s}</option>)}
          </select>
        </Field>
      ))}
      <Field label="Medication changed">
        <select value={value.change} onChange={set("change")}>
          <option value="No">No</option><option value="Ch">Yes</option>
        </select>
      </Field>
      <Field label="Diabetes medication">
        <select value={value.diabetesMed} onChange={set("diabetesMed")}>
          <option>Yes</option><option>No</option>
        </select>
      </Field>
      {!isIndex && (
        <Field label="Outcome of this stay">
          <select value={value.readmitted} onChange={set("readmitted")}>
            <option value="NO">Not readmitted</option>
            <option value=">30">Readmitted after 30 days</option>
            <option value="<30">Readmitted within 30 days</option>
          </select>
        </Field>
      )}
    </div>
  );
}

export default function RiskCalculator() {
  const schema = useFetch(getSchema);
  const [task, setTask] = useState("readmit30");
  const [history, setHistory] = useState([]);
  const [index, setIndex] = useState(blank());
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  if (schema.loading) return <Loading what="form" />;
  if (schema.error) return <ErrorBox error={schema.error} />;

  const submit = async () => {
    setBusy(true); setError(null);
    try {
      setResult(await predict(task, [...history, { ...index, readmitted: "NO" }]));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Risk calculator</h2>
        <p className="page-sub">
          Enter the current (index) hospital stay and, optionally, earlier
          stays of the same patient. TKGN-B returns a risk estimate with its
          history gate and the main drivers of its gradient-boosting anchor.
          Research prototype — not for clinical decisions.
        </p>
      </div>
      <datalist id="dx-codes">
        {schema.data.common_diagnoses.map((d) => <option key={d} value={d} />)}
      </datalist>
      <Selector label="Task" value={task} onChange={setTask} options={TASKS} />

      {history.map((h, i) => (
        <div key={i} className="card encounter-card">
          <h4>
            Earlier stay {i + 1}
            <button className="btn secondary small"
              onClick={() => setHistory(history.filter((_, j) => j !== i))}>Remove</button>
          </h4>
          <EncounterForm value={h} schema={schema.data} isIndex={false}
            onChange={(v) => setHistory(history.map((x, j) => (j === i ? v : x)))} />
        </div>
      ))}
      <div>
        <button className="btn secondary" onClick={() => setHistory([...history, blank()])}>
          + Add earlier stay
        </button>
      </div>

      <div className="card">
        <h3>Current stay</h3>
        <EncounterForm value={index} onChange={setIndex} isIndex schema={schema.data} />
        <div style={{ marginTop: 14 }}>
          <button className="btn" onClick={submit} disabled={busy}>
            {busy ? "Scoring…" : "Estimate risk"}
          </button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}
      {result && (
        <div className="card">
          <h3>{TASKS[result.task]}: {pct(result.probability)}</h3>
          <div className="risk-meter"><div style={{ width: `${Math.min(100, result.probability * 100)}%` }} /></div>
          <div className="stat-row" style={{ marginTop: 14 }}>
            <div className="stat"><div className="label">TKGN-B risk</div>
              <div className="value accent">{pct(result.probability)}</div></div>
            <div className="stat"><div className="label">GBDT anchor only</div>
              <div className="value">{pct(result.gbdt_anchor_probability)}</div></div>
            <div className="stat"><div className="label">Earlier stays used</div>
              <div className="value">{result.prior_encounters}</div></div>
            <div className="stat"><div className="label">History gate</div>
              <div className="value">{fmt(result.gate_history)}</div></div>
          </div>
          {result.history_attention.length > 0 && (
            <p className="muted small">
              Attention over earlier stays:{" "}
              {result.history_attention.map((a) => `stay ${a.encounter}: ${fmt(a.weight, 2)}`).join(" · ")}
            </p>
          )}
          <h3 style={{ marginTop: 12 }}>Largest drivers (TreeSHAP, log-odds)</h3>
          <table>
            <thead><tr><th>Feature</th><th>Value</th><th>Contribution</th></tr></thead>
            <tbody>
              {result.top_contributions.map((c) => (
                <tr key={c.feature}>
                  <td>{c.feature}</td><td className="num">{fmt(c.value, 2)}</td>
                  <td className="num" style={{ color: c.shap > 0 ? "#ffb4b7" : "#b8f2e2" }}>
                    {c.shap > 0 ? "+" : ""}{fmt(c.shap)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">{result.note}</p>
        </div>
      )}
    </div>
  );
}
