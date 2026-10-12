import React from "react";

export function Section({ num, title, children, id }) {
  return (
    <section className="section" id={id}>
      <div className="section-head">
        {num && <span className="section-num">{num}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

export function Segmented({ label, value, onChange, options }) {
  return (
    <div>
      {label && <div className="control-label">{label}</div>}
      <div className="segmented" role="group" aria-label={label}>
        {Object.entries(options).map(([key, text]) => (
          <button
            key={key}
            type="button"
            aria-pressed={value === key}
            onClick={() => onChange(key)}
          >
            {text}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}>
          <span
            className={`legend-key${it.line ? " line" : ""}`}
            style={{ background: it.color }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function Loading() {
  return <div className="loading">Loading…</div>;
}

export function ErrorMsg({ error }) {
  return (
    <div className="error">
      The results could not be loaded ({error}). Start the API with{" "}
      <code>uvicorn src.api:app</code> after running the pipeline.
    </div>
  );
}
