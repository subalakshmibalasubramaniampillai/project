import React, { useEffect, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter,
  ComposedChart, Tooltip, XAxis, YAxis,
} from "recharts";
import { useFetch } from "../useFetch";
import { getPatient, getPatients } from "../api";
import { COLORS, ErrorBox, Loading, Selector, TASKS, fmt } from "../common";

export default function PatientExplorer() {
  const [task, setTask] = useState("readmit30");
  const list = useFetch(() => getPatients(task), [task]);
  const [selected, setSelected] = useState("");
  const [patient, setPatient] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (list.data && list.data.length && !selected) {
      setSelected(String(list.data[0].patient_nbr));
    }
  }, [list.data, selected]);

  useEffect(() => {
    if (!selected) return;
    setError(null);
    getPatient(selected, task).then(setPatient).catch((e) => setError(e.message));
  }, [selected, task]);

  if (list.loading) return <Loading what="patients" />;
  if (list.error) return <ErrorBox error={list.error} />;

  const encounters = patient?.encounters || [];
  const chart = encounters.map((e) => ({
    stay: e.order + 1,
    lightgbm: e.p_lightgbm,
    tkgn: e.p_tkgn,
    tkgn_b: e.p_tkgn_b,
    outcome: e.y === null || e.y === undefined ? null : e.y,
  }));

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Patient explorer</h2>
        <p className="page-sub">
          Real de-identified test-set patients with at least two stays in the
          held-out partition. Risks were produced by models that never saw
          these patients during training.
        </p>
      </div>
      <div className="toolbar">
        <Selector label="Task" value={task}
          onChange={(t) => { setTask(t); setSelected(""); setPatient(null); }}
          options={TASKS} />
        <div className="field inline-field">
          <label>Patient (number of stays)</label>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {list.data.map((p) => (
              <option key={p.patient_nbr} value={p.patient_nbr}>
                {p.patient_nbr} ({p.encounters})
              </option>
            ))}
          </select>
        </div>
      </div>
      {error && <div className="alert error">{error}</div>}
      {patient && (
        <>
          <div className="card">
            <h3>Predicted risk at each stay (dots: observed outcome)</h3>
            <div className="chart-box">
              <ResponsiveContainer>
                <ComposedChart data={chart} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
                  <XAxis dataKey="stay" tick={{ fill: "#9aa7c0", fontSize: 11 }}
                    label={{ value: "Stay", position: "insideBottomRight", fill: "#9aa7c0" }} />
                  <YAxis domain={[0, 1]} tick={{ fill: "#9aa7c0", fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: "#171f33", border: "1px solid #2a3550" }}
                    formatter={(v) => fmt(v)} />
                  <Legend />
                  <Line dataKey="tkgn_b" name="TKGN-B" stroke={COLORS.tkgn_b} strokeWidth={2} connectNulls />
                  <Line dataKey="tkgn" name="TKGN" stroke={COLORS.tkgn} connectNulls />
                  <Line dataKey="lightgbm" name="LightGBM" stroke={COLORS.lightgbm} connectNulls />
                  <Scatter dataKey="outcome" name="Observed outcome" fill="#e5484d" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <h3>Encounters</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Stay</th><th>Age</th><th>Days</th><th>Meds</th><th>Prior inpt.</th>
                    <th>Dx 1</th><th>Dx 2</th><th>Dx 3</th><th>HbA1c</th><th>Insulin</th>
                    <th>Readmitted</th><th>TKGN-B</th><th>History gate</th></tr>
                </thead>
                <tbody>
                  {encounters.map((e) => (
                    <tr key={e.encounter_id}>
                      <td>{e.order + 1}</td><td>{e.age}</td>
                      <td className="num">{e.time_in_hospital}</td>
                      <td className="num">{e.num_medications}</td>
                      <td className="num">{e.number_inpatient}</td>
                      <td>{e.diag_1 ?? "—"}</td><td>{e.diag_2 ?? "—"}</td><td>{e.diag_3 ?? "—"}</td>
                      <td>{!e.A1Cresult || e.A1Cresult === "None" ? "not measured" : e.A1Cresult}</td><td>{e.insulin}</td>
                      <td>{e.readmitted}</td>
                      <td className="num">{fmt(e.p_tkgn_b)}</td>
                      <td className="num">{fmt(e.gate_history)}</td>
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
