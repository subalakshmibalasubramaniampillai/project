import React from "react";
import { useFetch } from "../useFetch";
import { getExplanation } from "../api";

export default function Explanation() {
  const { data, loading, error } = useFetch(getExplanation);

  if (loading) return <div className="loading">Loading…</div>;
  if (error) return <div className="alert error">{error}</div>;

  const exp = typeof data === "string" ? JSON.parse(data) : data;
  const contributions = exp.model_contribution || [];

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Prediction + explanation</h2>
        <p className="page-sub">
          A saved feature-contribution probe for one example patient from the
          test split.
        </p>
      </div>

      <div className="stat-row">
        <div className="stat">
          <div className="label">Example patient</div>
          <div className="value">{exp.patient_id}</div>
        </div>
        <div className="stat">
          <div className="label">Model</div>
          <div className="value">{exp.model}</div>
        </div>
        <div className="stat">
          <div className="label">Prediction probability</div>
          <div className="value accent">
            {Number(exp.prediction_probability).toFixed(3)}
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Feature contribution probe</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {contributions.map((c, i) => (
                <tr key={i}>
                  <td>{c.feature}</td>
                  <td className="num">{Number(c.value).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="alert warn">{exp.interpretation}</div>
    </div>
  );
}