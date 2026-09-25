import React from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import Overview from "./pages/Overview";
import PatientExplorer from "./pages/PatientExplorer";
import Explanation from "./pages/Explanation";
import NewPatient from "./pages/NewPatient";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import Ablation from "./pages/Ablation";
import Clustering from "./pages/Clustering";
import Fusion from "./pages/Fusion";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/patient", label: "Patient explorer" },
  { to: "/explanation", label: "Explanation" },
  { to: "/new-patient", label: "New patient" },
  { to: "/graph", label: "Knowledge graph" },
  { to: "/ablation", label: "Ablation" },
  { to: "/clustering", label: "Clustering" },
  { to: "/fusion", label: "Fusion" },
];

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◇</span>
          <div>
            <h1>Progression Lab</h1>
            <p>Longitudinal diabetes progression research system</p>
          </div>
        </div>
        <nav className="topnav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                isActive ? "nav-link active" : "nav-link"
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/patient" element={<PatientExplorer />} />
          <Route path="/explanation" element={<Explanation />} />
          <Route path="/new-patient" element={<NewPatient />} />
          <Route path="/graph" element={<KnowledgeGraph />} />
          <Route path="/ablation" element={<Ablation />} />
          <Route path="/clustering" element={<Clustering />} />
          <Route path="/fusion" element={<Fusion />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        Research and decision-support prototype — not a diagnostic tool.
      </footer>
    </div>
  );
}