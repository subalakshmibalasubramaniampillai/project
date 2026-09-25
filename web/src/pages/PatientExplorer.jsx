import React, { useState } from "react";
import { useFetch } from "../useFetch";
import { getPatients, getDataset } from "../api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid,
} from "recharts";

export default function PatientExplorer() {
  const { data: patientList, loading: listLoading, error: listError } =
    useFetch(getPatients);
  const [selected, setSelected] = useState("");
  const { data: patientData, loading: pLoading } = useFetch(
    () => getDataset(selected),
    [selected]
  );

  if (listLoading) return <div className="loading">Loading patients…</div>;
  if (listError) return <div className="alert error">{listError}</div>;

  const visits = patientData?.visits || [];
  const valueKeys = [
    "hba1c", "glucose", "bmi", "systolic_bp", "diastolic_bp",
  ];

  const chartData = visits.map((v) => ({
    name: `V${v.visit}`,
    ...Object.fromEntries(valueKeys.map((k) => [k, Number(v[k])])),
  }));

  const options = Array.isArray(patientList)
    ? patientList
    : patientList?.patients || [];

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Patient explorer</h2>
        <p className="page-sub">
          Longitudinal trends and raw visit data for a single patient.
        </p>
      </div>

      <div className="card" style={{ maxWidth: 420 }}>
        <div className="field">
          <label htmlFor="patient-select">Patient</label>
          <select
            id="patient-select"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            <option value="">Select a patient…</option>
            {options.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {selected && (
        <>
          <div className="card">
            <h3>HbA1c and glucose trajectory</h3>
            {pLoading ? (
              <div className="loading">Loading patient…</div>
            ) : chartData.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData} margin={{ left: -18, right: 12 }}>
                  <CartesianGrid stroke="#2a3550" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#9aa7c0" />
                  <YAxis stroke="#9aa7c0" />
                  <Tooltip
                    contentStyle={{ background: "#1d2740", border: "1px solid #2a3550" }}
                    labelStyle={{ color: "#e6ebf5" }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="hba1c" stroke="#4f8ef7" dot={false} />
                  <Line type="monotone" dataKey="glucose" stroke="#38d9a9" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty">No visit data for this patient.</div>
            )}
          </div>

          <div className="card">
            <h3>{selected} — visits</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {valueKeys.map((k) => (
                      <th key={k}>{k}</th>
                    ))}
                    <th>conditions</th>
                  </tr>
                </thead>
                <tbody>
                  {visits.map((v) => (
                    <tr key={v.visit}>
                      {valueKeys.map((k) => (
                        <td key={k} className="num">
                          {Number(v[k]).toFixed(2)}
                        </td>
                      ))}
                      <td>{v.conditions}</td>
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