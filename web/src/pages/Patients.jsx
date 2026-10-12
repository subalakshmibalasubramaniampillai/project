import React, { useEffect, useMemo, useState } from "react";
import { getPatient, getPatients, predict } from "../api";
import { RiskTimeline } from "../components/charts";
import { ErrorMsg, Loading, Segmented } from "../components/ui";
import {
  COMMON_DIAGNOSES, OUTCOMES, count, diagnosisName, featureName, pct,
} from "../format";
import { useApi } from "../useFetch";

// ── example patients ─────────────────────────────────────────────

const READMIT_TEXT = { "<30": "Readmitted within 30 days", ">30": "Readmitted after 30 days", NO: "Not readmitted" };

function outcomeOf(task, e) {
  if (task === "readmit30") return e.readmitted === "<30";
  return e.y_escalation === 1;
}

function outcomeText(task, e) {
  if (task === "readmit30") return READMIT_TEXT[e.readmitted] || e.readmitted;
  if (e.y_escalation === -1 || e.y_escalation === null) return "No later stay";
  return e.y_escalation === 1 ? "Stepped up at next stay" : "Not stepped up";
}

function Explorer() {
  const [task, setTask] = useState("readmit30");
  const [list, setList] = useState(null);
  const [selected, setSelected] = useState("");
  const [patient, setPatient] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setList(null);
    setPatient(null);
    getPatients(task).then((rows) => {
      setList(rows);
      // start with a patient whose stays include both outcomes
      const pick = rows.find((r) => r.encounters >= 4 && r.encounters <= 8
        && r.events >= 1 && r.events < r.encounters) || rows[0];
      setSelected(String(pick.patient_nbr));
    }).catch((e) => setError(e.message));
  }, [task]);

  useEffect(() => {
    if (!selected) return;
    setPatient(null);
    getPatient(selected, task).then(setPatient).catch((e) => setError(e.message));
  }, [selected, task]);

  if (error) return <ErrorMsg error={error} />;
  if (!list) return <Loading />;

  const stays = patient?.encounters || [];
  const points = stays.map((e) => ({
    stay: e.order + 1,
    risk: e.p_tkgn_b ?? null,
    outcome: outcomeOf(task, e),
  }));

  return (
    <>
      <p>
        These are real, de-identified patients from the test set: the model
        never saw them during training. The line shows the risk TKGN-B gave at
        each discharge; the dots underneath show what actually happened (filled
        dot: {OUTCOMES[task].event}).
      </p>
      <div className="controls">
        <Segmented label="Outcome" value={task} onChange={setTask}
          options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
        <div style={{ minWidth: 260 }}>
          <div className="control-label">Patient</div>
          <select className="plain" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {list.map((p) => (
              <option key={p.patient_nbr} value={p.patient_nbr}>
                Patient {p.patient_nbr} · {p.encounters} stays
              </option>
            ))}
          </select>
        </div>
      </div>
      {!patient && <Loading />}
      {patient && (
        <>
          <figure>
            <RiskTimeline points={points} eventLabel={OUTCOMES[task].event} />
            <figcaption>
              Predicted risk at each stay of patient {patient.patient_nbr}, in order.
            </figcaption>
          </figure>
          <div className="table-wrap" style={{ marginTop: 22 }}>
            <table>
              <thead>
                <tr>
                  <th className="num">Stay</th>
                  <th>Age</th>
                  <th>Main diagnosis</th>
                  <th className="num">Days</th>
                  <th className="num">Medications</th>
                  <th>Insulin</th>
                  <th>HbA1c</th>
                  <th>What happened</th>
                  <th className="num">Predicted risk</th>
                </tr>
              </thead>
              <tbody>
                {stays.map((e) => (
                  <tr key={e.encounter_id}>
                    <td className="num">{e.order + 1}</td>
                    <td>{(e.age || "").replace("[", "").replace(")", "")}</td>
                    <td>{e.diag_1 ? `${e.diag_1} · ${diagnosisName(e.diag_1)}` : "–"}</td>
                    <td className="num">{e.time_in_hospital}</td>
                    <td className="num">{e.num_medications}</td>
                    <td>{e.insulin === "No" ? "none" : e.insulin.toLowerCase()}</td>
                    <td>{!e.A1Cresult || e.A1Cresult === "None" ? "not measured" : e.A1Cresult}</td>
                    <td>{outcomeText(task, e)}</td>
                    <td className="num">{e.p_tkgn_b === null || e.p_tkgn_b === undefined ? "–" : pct(e.p_tkgn_b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}

// ── risk calculator ──────────────────────────────────────────────

const AGES = ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"];
const ADMISSION_TYPE = { 1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma centre" };
const ADMISSION_SOURCE = { 7: "Emergency room", 1: "Physician referral", 2: "Clinic referral",
  4: "Transfer from another hospital", 6: "Transfer from another facility" };
const DISCHARGE = { 1: "Home", 6: "Home with home health care", 3: "Skilled nursing facility",
  22: "Rehabilitation facility", 2: "Another short-stay hospital", 23: "Long-term care hospital",
  7: "Left against medical advice" };
const DRUG = { No: "Not prescribed", Steady: "Steady dose", Up: "Dose increased", Down: "Dose reduced" };

const blankStay = () => ({
  age: "[60-70)", gender: "Female", race: "Caucasian",
  admission_type_id: "1", admission_source_id: "7", discharge_disposition_id: "1",
  time_in_hospital: 4, number_diagnoses: 7, num_lab_procedures: 43, num_procedures: 1,
  num_medications: 16, number_inpatient: 0, number_emergency: 0, number_outpatient: 0,
  diag_1: "428", diag_2: "250", diag_3: "401",
  A1Cresult: "", max_glu_serum: "", insulin: "Steady", metformin: "No",
  change: "No", diabetesMed: "Yes", readmitted: ">30",
});

function Select({ label, value, onChange, options }) {
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {Object.entries(options).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
    </div>
  );
}

function NumberField({ label, value, onChange, min = 0, max = 100 }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" min={min} max={max} value={value}
        onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

function StayForm({ value, onChange, earlier }) {
  const set = (key) => (v) => onChange({ ...value, [key]: v });

  if (earlier) {
    // earlier stays: only the details that differ most between stays
    return (
      <div className="form-grid">
        <Select label="Type of admission" value={value.admission_type_id}
          onChange={set("admission_type_id")} options={ADMISSION_TYPE} />
        <Select label="Discharged to" value={value.discharge_disposition_id}
          onChange={set("discharge_disposition_id")} options={DISCHARGE} />
        <NumberField label="Days in hospital" value={value.time_in_hospital}
          onChange={set("time_in_hospital")} min={1} max={14} />
        <NumberField label="Medications given" value={value.num_medications}
          onChange={set("num_medications")} />
        <div className="field">
          <label>Main diagnosis (ICD-9)</label>
          <input list="dx-list" value={value.diag_1} onChange={(e) => set("diag_1")(e.target.value)} />
          <div className="hint">{diagnosisName(value.diag_1) || "Enter an ICD-9 code"}</div>
        </div>
        <Select label="Insulin" value={value.insulin} onChange={set("insulin")} options={DRUG} />
        <Select label="What happened after this stay" value={value.readmitted}
          onChange={set("readmitted")}
          options={{ NO: "Not readmitted", ">30": "Readmitted after 30 days", "<30": "Readmitted within 30 days" }} />
      </div>
    );
  }

  return (
    <>
      <fieldset>
        <legend>Patient</legend>
        <div className="form-grid">
          <Select label="Age (years)" value={value.age} onChange={set("age")}
            options={Object.fromEntries(AGES.map((a) => [a, a.replace("[", "").replace(")", "").replace("-", " to ")]))} />
          <Select label="Sex" value={value.gender} onChange={set("gender")}
            options={{ Female: "Female", Male: "Male" }} />
          <Select label="Race" value={value.race} onChange={set("race")}
            options={{ Caucasian: "Caucasian", AfricanAmerican: "African American",
              Hispanic: "Hispanic", Asian: "Asian", Other: "Other" }} />
        </div>
      </fieldset>
      <fieldset>
        <legend>This stay</legend>
        <div className="form-grid">
          <Select label="Type of admission" value={value.admission_type_id}
            onChange={set("admission_type_id")} options={ADMISSION_TYPE} />
          <Select label="Admitted from" value={value.admission_source_id}
            onChange={set("admission_source_id")} options={ADMISSION_SOURCE} />
          <Select label="Discharged to" value={value.discharge_disposition_id}
            onChange={set("discharge_disposition_id")} options={DISCHARGE} />
          <NumberField label="Days in hospital" value={value.time_in_hospital}
            onChange={set("time_in_hospital")} min={1} max={14} />
          <NumberField label="Medications given" value={value.num_medications}
            onChange={set("num_medications")} />
          <NumberField label="Laboratory tests" value={value.num_lab_procedures}
            onChange={set("num_lab_procedures")} max={140} />
          <NumberField label="Diagnoses recorded" value={value.number_diagnoses}
            onChange={set("number_diagnoses")} min={1} max={16} />
        </div>
      </fieldset>
      <fieldset>
        <legend>In the year before this stay</legend>
        <div className="form-grid">
          <NumberField label="Inpatient stays" value={value.number_inpatient}
            onChange={set("number_inpatient")} />
          <NumberField label="Emergency visits" value={value.number_emergency}
            onChange={set("number_emergency")} />
          <NumberField label="Outpatient visits" value={value.number_outpatient}
            onChange={set("number_outpatient")} />
        </div>
      </fieldset>
      <fieldset>
        <legend>Diagnoses (ICD-9)</legend>
        <div className="form-grid">
          {["diag_1", "diag_2", "diag_3"].map((k, i) => (
            <div className="field" key={k}>
              <label>{["Main diagnosis", "Second diagnosis", "Third diagnosis"][i]}</label>
              <input list="dx-list" value={value[k]} onChange={(e) => set(k)(e.target.value)} />
              <div className="hint">{diagnosisName(value[k]) || "Enter an ICD-9 code"}</div>
            </div>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend>Diabetes care</legend>
        <div className="form-grid">
          <Select label="HbA1c result" value={value.A1Cresult} onChange={set("A1Cresult")}
            options={{ "": "Not measured", Norm: "Normal", ">7": "Above 7%", ">8": "Above 8%" }} />
          <Select label="Insulin" value={value.insulin} onChange={set("insulin")} options={DRUG} />
          <Select label="Metformin" value={value.metformin} onChange={set("metformin")} options={DRUG} />
          <Select label="Diabetes medication changed" value={value.change} onChange={set("change")}
            options={{ No: "No", Ch: "Yes" }} />
        </div>
      </fieldset>
    </>
  );
}

function driverText(c) {
  const name = featureName(c.feature);
  if (c.feature.includes("=")) return c.value ? name : `${name}: no`;
  return `${name}: ${Number.isInteger(c.value) ? c.value : c.value.toFixed(1)}`;
}

function Calculator() {
  const { data } = useApi({ overview: "/overview" });
  const [task, setTask] = useState("readmit30");
  const [history, setHistory] = useState([]);
  const [current, setCurrent] = useState(blankStay());
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const base = data?.overview.dataset;
  const average = base ? (task === "readmit30" ? base.readmit30_rate : base.escalation_rate) : null;

  const shared = { age: current.age, gender: current.gender, race: current.race };
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const stays = [...history.map((h) => ({ ...h, ...shared })), { ...current, readmitted: "NO" }];
      setResult({ ...(await predict(task, stays)), task });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const drivers = useMemo(() => (result ? result.top_contributions
    .filter((c) => Math.abs(c.shap) >= 0.02).slice(0, 6) : []), [result]);

  return (
    <>
      <p>
        Enter the details of a hospital stay at discharge. Adding earlier stays
        of the same patient lets the model use their history. The estimate is for
        research and teaching only.
      </p>
      <datalist id="dx-list">
        {COMMON_DIAGNOSES.map((d) => <option key={d.code} value={d.code}>{d.name}</option>)}
      </datalist>
      <div className="controls">
        <Segmented label="Estimate the risk of" value={task}
          onChange={(t) => { setTask(t); setResult(null); }}
          options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
      </div>

      <div className="panel">
        <StayForm value={current} onChange={setCurrent} />
      </div>

      <h3 style={{ marginTop: 28 }}>Earlier stays of this patient</h3>
      {history.length === 0 && (
        <p className="muted small">None added. The estimate will treat this as a first stay.</p>
      )}
      {history.map((h, i) => (
        <div key={i} className="stay-card">
          <div className="stay-card-head" style={{ marginBottom: 12 }}>
            <strong>Earlier stay {i + 1}{i === history.length - 1 ? " (most recent)" : ""}</strong>
            <button type="button" className="btn link"
              onClick={() => setHistory(history.filter((_, j) => j !== i))}>Remove</button>
          </div>
          <StayForm value={h} earlier
            onChange={(v) => setHistory(history.map((x, j) => (j === i ? v : x)))} />
        </div>
      ))}
      <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
        <button type="button" className="btn secondary"
          onClick={() => setHistory([...history, { ...blankStay(), ...shared }])}>
          Add an earlier stay
        </button>
        <button type="button" className="btn" onClick={submit} disabled={busy}>
          {busy ? "Calculating…" : "Estimate risk"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {result && (
        <div className="panel" style={{ marginTop: 28 }}>
          <div className="kicker">Estimated risk of {OUTCOMES[result.task].short.toLowerCase()}</div>
          <div className="result-figure">{pct(result.probability)}</div>
          <div className="meter"><div style={{ width: `${Math.min(100, result.probability * 100)}%` }} /></div>
          <p className="small muted">
            {average !== null && <>The average across all {count(task === "readmit30"
              ? base.cohort_encounters : base.escalation_samples)} stays in the data is {pct(average)}. </>}
            {result.prior_encounters
              ? `The estimate used ${result.prior_encounters} earlier stay${result.prior_encounters > 1 ? "s" : ""}.`
              : "No earlier stays were used."}
          </p>
          {drivers.length > 0 && (
            <>
              <h3 style={{ marginTop: 18 }}>Main reasons for this estimate</h3>
              <table>
                <tbody>
                  {drivers.map((c) => (
                    <tr key={c.feature}>
                      <td>{driverText(c)}</td>
                      <td className={c.shap > 0 ? "verdict-worse" : "verdict-better"}>
                        {c.shap > 0 ? "raises risk" : "lowers risk"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small" style={{ marginTop: 10 }}>
                Based on SHAP values of the LightGBM part of the model.
              </p>
            </>
          )}
        </div>
      )}
    </>
  );
}

export default function Patients() {
  const [tab, setTab] = useState("examples");
  return (
    <>
      <div className="intro">
        <div className="kicker">Patients</div>
        <h1>The model at the level of one patient</h1>
        <p className="lead">
          Follow real patients from the test set across their hospital stays, or
          enter a stay of your own to see the estimate the model would give.
        </p>
      </div>
      <div className="tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === "examples"}
          onClick={() => setTab("examples")}>Example patients</button>
        <button type="button" role="tab" aria-selected={tab === "calculator"}
          onClick={() => setTab("calculator")}>Estimate a risk</button>
      </div>
      {tab === "examples" ? <Explorer /> : <Calculator />}
    </>
  );
}
