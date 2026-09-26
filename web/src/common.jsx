import React from "react";

export const TASKS = {
  readmit30: "30-day readmission",
  escalation: "Treatment escalation at next stay",
};

export const SPLITS = {
  grouped: "Patient-grouped (5 repeats)",
  temporal: "Temporal / prospective",
};

export const LABELS = {
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  mlp: "MLP",
  gru: "GRU",
  retain: "RETAIN",
  transformer: "Transformer",
  tkgn: "TKGN",
  tkgn_b: "TKGN-B",
};

export const PROPOSED = new Set(["tkgn", "tkgn_b"]);

export const COLORS = {
  logistic_regression: "#8a94a8",
  random_forest: "#6c7a96",
  xgboost: "#b07cf0",
  lightgbm: "#4f8ef7",
  mlp: "#7a8bb0",
  gru: "#59c3c3",
  retain: "#3aa39f",
  transformer: "#2d8a86",
  tkgn: "#f4a340",
  tkgn_b: "#38d9a9",
};

export function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

export function pct(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function pValue(p) {
  if (p === null || p === undefined) return "—";
  if (p < 0.001) return "<0.001";
  return Number(p).toFixed(3);
}

export function Selector({ value, onChange, options, label }) {
  return (
    <div className="field inline-field">
      {label && <label>{label}</label>}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {Object.entries(options).map(([k, v]) => (
          <option key={k} value={k}>
            {v}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ModelName({ model }) {
  return (
    <span className="pill-wrap">
      <span className="swatch" style={{ background: COLORS[model] || "#888" }} />
      {LABELS[model] || model}
      {PROPOSED.has(model) && (
        <span className="pill novel" style={{ marginLeft: 6 }}>
          proposed
        </span>
      )}
    </span>
  );
}

export function Loading({ what = "results" }) {
  return <div className="loading">Loading {what}…</div>;
}

export function ErrorBox({ error }) {
  return (
    <div className="alert error">
      {error}. Run <code>python -m src.run_pipeline</code> to generate results.
    </div>
  );
}
