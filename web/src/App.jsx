import React from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import Overview from "./pages/Overview";
import Statistics from "./pages/Statistics";
import Ablation from "./pages/Ablation";
import LearningCurve from "./pages/LearningCurve";
import Calibration from "./pages/Calibration";
import Fairness from "./pages/Fairness";
import Explainability from "./pages/Explainability";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import PatientExplorer from "./pages/PatientExplorer";
import RiskCalculator from "./pages/RiskCalculator";

const NAV = [
  { to: "/", label: "Results", end: true },
  { to: "/statistics", label: "Statistics" },
  { to: "/ablation", label: "Ablation" },
  { to: "/learning-curve", label: "Data efficiency" },
  { to: "/calibration", label: "Calibration" },
  { to: "/fairness", label: "Fairness" },
  { to: "/explain", label: "Explainability" },
  { to: "/graph", label: "Knowledge graph" },
  { to: "/patients", label: "Patient explorer" },
  { to: "/calculator", label: "Risk calculator" },
];

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◇</span>
          <div>
            <h1>TKGN Lab</h1>
            <p>Knowledge-graph temporal modelling on real diabetes inpatient records</p>
          </div>
        </div>
        <nav className="topnav">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/statistics" element={<Statistics />} />
          <Route path="/ablation" element={<Ablation />} />
          <Route path="/learning-curve" element={<LearningCurve />} />
          <Route path="/calibration" element={<Calibration />} />
          <Route path="/fairness" element={<Fairness />} />
          <Route path="/explain" element={<Explainability />} />
          <Route path="/graph" element={<KnowledgeGraph />} />
          <Route path="/patients" element={<PatientExplorer />} />
          <Route path="/calculator" element={<RiskCalculator />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        Data: Diabetes 130-US Hospitals 1999–2008 (UCI, CC BY 4.0). Research and
        decision-support prototype — not a diagnostic tool.
      </footer>
    </div>
  );
}
