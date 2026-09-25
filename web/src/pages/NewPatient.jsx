import React, { useState } from "react";
import { predict } from "../api";

const DEFAULT_VISIT = {
  age: 52, bmi: 31.2, hba1c: 6.1, glucose: 132,
  systolic_bp: 138, diastolic_bp: 86, conditions: "hypertension",
};

const MAX_VISITS = 4;

export default function NewPatient() {
  const [patientId, setPatientId] = useState("NEW-0001");
  const [nVisits, setNVisits] = useState(1);
  const [visits, setVisits] = useState([{ ...DEFAULT_VISIT }]);
  const [conditions, setConditions] = useState("hypertension");
  const [useTagnn, setUseTagnn] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);

  const setVisitField = (idx, field, value) => {
    setVisits((prev) => prev.map((v, i) =>
      i === idx ? { ...v, [field]: Number(value) } : v
    ));
  };

  const handleVisitsChange = (e) => {
    const n = Math.max(1, Math.min(MAX_VISITS, Number(e.target.value)));
    setNVisits(n);
    setVisits((prev) => {
      const copy = [...prev];
      while (copy.length < n) copy.push({ ...DEFAULT_VISIT });
      return copy.slice(0, n);
    });
  };

  const submit = async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const today = new Date().toISOString().slice(0, 10);
      const payload = visits.map((v, i) => ({
        patient_id: patientId,
        visit: i + 1,
        date: today,
        ...v,
        conditions,
      }));
      const res = await predict(payload, useTagnn);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const showing = result === null && error === null;

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">New patient</h2>
        <p className="page-sub">
          Enter one or more visits. A single visit gives a baseline
          prediction; multiple visits enable longitudinal features.
        </p>
      </div>

      <div className="card">
        <div className="form-grid" style={{ marginBottom: 16 }}>
          <div className="field">
            <label>Patient ID</label>
            <input
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Visits</label>
            <input
              type="number"
              min={1}
              max={MAX_VISITS}
              value={nVisits}
              onChange={handleVisitsChange}
            />
          </div>
          <div className="field">
            <label>Condition</label>
            <select value={conditions} onChange={(e) => setConditions(e.target.value)}>
              <option value="hypertension">hypertension</option>
              <option value="none">none</option>
              <option value="unknown">unknown</option>
            </select>
          </div>
          <div className="field" style={{ display: "flex", alignItems: "flex-end" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8, margin: 0 }}>
              <input
                type="checkbox"
                checked={useTagnn}
                onChange={(e) => setUseTagnn(e.target.checked)}
                style={{ width: "auto" }}
              />
              Use TAGNN (novel model)
            </label>
          </div>
        </div>

        {visits.map((v, idx) => (
          <div className="card" key={idx} style={{ marginBottom: 12 }}>
            <h3>Visit {idx + 1}</h3>
            <div className="form-grid">
              <div className="field">
                <label>Age</label>
                <input type="number" value={v.age}
                  onChange={(e) => setVisitField(idx, "age", e.target.value)} />
              </div>
              <div className="field">
                <label>BMI</label>
                <input type="number" step="0.1" value={v.bmi}
                  onChange={(e) => setVisitField(idx, "bmi", e.target.value)} />
              </div>
              <div className="field">
                <label>HbA1c</label>
                <input type="number" step="0.1" value={v.hba1c}
                  onChange={(e) => setVisitField(idx, "hba1c", e.target.value)} />
              </div>
              <div className="field">
                <label>Glucose</label>
                <input type="number" step="1" value={v.glucose}
                  onChange={(e) => setVisitField(idx, "glucose", e.target.value)} />
              </div>
              <div className="field">
                <label>Systolic BP</label>
                <input type="number" step="1" value={v.systolic_bp}
                  onChange={(e) => setVisitField(idx, "systolic_bp", e.target.value)} />
              </div>
              <div className="field">
                <label>Diastolic BP</label>
                <input type="number" step="1" value={v.diastolic_bp}
                  onChange={(e) => setVisitField(idx, "diastolic_bp", e.target.value)} />
              </div>
            </div>
          </div>
        ))}

        <button className="btn" onClick={submit} disabled={running}>
          {running ? "Running prediction…" : "Run prediction"}
        </button>

        {(result || error) && (
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setResult(null);
              setError(null);
            }}
            style={{ marginLeft: 14, color: "var(--muted)", fontSize: 13 }}
          >
            New input
          </a>
        )}
      </div>

      {error && <div className="alert error">{error}</div>}

      {result && (
        <>
          <div className="stat-row">
            <div className="stat">
              <div className="label">Model</div>
              <div className="value">{result.model}</div>
            </div>
            <div className="stat">
              <div className="label">Mode</div>
              <div className="value">{result.mode}</div>
            </div>
            <div className="stat">
              <div className="label">Prediction</div>
              <div className="value accent">
                {result.probability >= 0.5
                  ? "progression signal"
                  : "no progression signal"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Probability</div>
              <div className="value">{Number(result.probability).toFixed(3)}</div>
            </div>
          </div>

          <div className="card">
            <h3>Model contribution</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.model_contribution || []).map((c, i) => (
                    <tr key={i}>
                      <td>{c.feature}</td>
                      <td className="num">{Number(c.value).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.subgraph_nodes > 0 && (
              <div className="loading" style={{ padding: "8px 0" }}>
                Subgraph: {result.subgraph_nodes} nodes, {result.subgraph_edges}{" "}
                edges
              </div>
            )}
          </div>

          <div className="alert warn">{result.interpretation}</div>
        </>
      )}

      {showing && (
        <div className="alert info">
          This form runs inference only — it never retrains the model.
        </div>
      )}
    </div>
  );
}