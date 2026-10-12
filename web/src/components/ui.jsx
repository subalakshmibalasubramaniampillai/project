import React from "react";

export function Chips({ label, value, onChange, options }) {
  return (
    <div>
      {label && <div className="control-label">{label}</div>}
      <div className="chips" role="group" aria-label={label}>
        {Object.entries(options).map(([key, text]) => (
          <button key={key} type="button" className="chip" aria-pressed={value === key}
            onClick={() => onChange(key)}>
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
          <span className={`legend-key${it.line ? " line" : ""}`} style={{ background: it.color }} />
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
      This part of the site could not load ({error}). The StayTrace service may
      not be running.
    </div>
  );
}

// Small line icons drawn for this site.
const PATHS = {
  history: "M4 12h16M6 12a2 2 0 1 0 0 .01M12 12a2 2 0 1 0 0 .01M18 12a3 3 0 1 0 0 .01",
  gate: "M5 19V5h14v14M9 19v-6a3 3 0 0 1 6 0v6",
  layers: "M12 4l8 4-8 4-8-4 8-4zM4 12l8 4 8-4M4 16l8 4 8-4",
  split: "M12 4v6M12 10l-6 6M12 10l6 6M6 16v4M18 16v4",
  shield: "M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3zM9 12l2 2 4-4",
  stethoscope: "M6 3v6a4 4 0 0 0 8 0V3M10 13v3a4 4 0 0 0 8 0v-2M18 14a2 2 0 1 0 0-.01",
  person: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  chart: "M4 20V4M4 20h16M8 16v-5M12 16V8M16 16v-3",
  arrow: "M5 12h14M13 6l6 6-6 6",
  up: "M12 19V5M6 11l6-6 6 6",
  down: "M12 5v14M6 13l6 6 6-6",
  print: "M7 9V4h10v5M7 17H5v-6h14v6h-2M7 14h10v6H7z",
};

export function Icon({ name, size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

export function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
      <rect width="34" height="34" rx="9" fill="#f3efe4" />
      <path d="M6 20h22" stroke="#1f4a3c" strokeWidth="2" strokeLinecap="round" />
      <circle cx="9" cy="20" r="2.6" fill="#1f4a3c" />
      <circle cx="16" cy="20" r="2.6" fill="#1f4a3c" />
      <circle cx="24.5" cy="20" r="4" fill="#c4552e" />
      <path d="M24.5 13V9" stroke="#c4552e" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
