import React, { useEffect, useState } from "react";
import { getPatient, getPatients } from "../api";
import { RiskTimeline } from "../components/charts";
import { Chips, ErrorMsg, Loading } from "../components/ui";
import { OUTCOMES, diagnosisName, pct } from "../format";
import { SITE } from "../site";

const READMIT_TEXT = { "<30": "Readmitted within 30 days", ">30": "Readmitted after 30 days", NO: "Not readmitted" };

function outcomeOf(task, e) {
  return task === "readmit30" ? e.readmitted === "<30" : e.y_escalation === 1;
}

function outcomeText(task, e) {
  if (task === "readmit30") return READMIT_TEXT[e.readmitted] || e.readmitted;
  if (e.y_escalation === -1 || e.y_escalation === null) return "No later stay";
  return e.y_escalation === 1 ? "Treatment stepped up next stay" : "Not stepped up";
}

export default function Cases() {
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

  const stays = patient?.encounters || [];
  const points = stays.map((e) => ({ stay: e.order + 1, risk: e.p_tkgn_b ?? null, outcome: outcomeOf(task, e) }));

  return (
    <div className="wrap page">
      <span className="eyebrow">Case studies</span>
      <h1 style={{ fontSize: "clamp(32px, 4vw, 46px)" }}>Real patients, stay by stay</h1>
      <p className="lead">
        These are de-identified patients from the test data. {SITE.brand} never
        saw them while it was being built. The line is the risk it gave at each
        discharge; the circles below show what happened.
      </p>
      {error && <ErrorMsg error={error} />}
      <div className="controls">
        <Chips label="Outcome" value={task} onChange={setTask}
          options={{ readmit30: OUTCOMES.readmit30.short, escalation: OUTCOMES.escalation.short }} />
        {list && (
          <div style={{ minWidth: 280 }}>
            <div className="control-label">Patient</div>
            <select className="plain" value={selected} onChange={(e) => setSelected(e.target.value)}>
              {list.map((p) => (
                <option key={p.patient_nbr} value={p.patient_nbr}>
                  Patient {p.patient_nbr}: {p.encounters} stays, {p.events} with the outcome
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      {!patient && !error && <Loading />}
      {patient && (
        <>
          <figure className="card">
            <RiskTimeline points={points} eventLabel={OUTCOMES[task].event} />
            <figcaption>Filled circle: {OUTCOMES[task].event}. Open circle: it did not happen.</figcaption>
          </figure>
          <div className="card" style={{ marginTop: 20 }}>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th className="num">Stay</th><th>Age</th><th>Main diagnosis</th><th className="num">Days</th>
                    <th>Insulin</th><th>HbA1c</th><th>What happened</th><th className="num">{SITE.brand} risk</th>
                  </tr>
                </thead>
                <tbody>
                  {stays.map((e) => (
                    <tr key={e.encounter_id}>
                      <td className="num">{e.order + 1}</td>
                      <td>{(e.age || "").replace("[", "").replace(")", "")}</td>
                      <td>{e.diag_1 ? `${e.diag_1} · ${diagnosisName(e.diag_1)}` : "–"}</td>
                      <td className="num">{e.time_in_hospital}</td>
                      <td>{e.insulin === "No" ? "none" : e.insulin.toLowerCase()}</td>
                      <td>{!e.A1Cresult || e.A1Cresult === "None" ? "not measured" : e.A1Cresult}</td>
                      <td>{outcomeText(task, e)}</td>
                      <td className="num"><strong>{e.p_tkgn_b == null ? "–" : pct(e.p_tkgn_b)}</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
