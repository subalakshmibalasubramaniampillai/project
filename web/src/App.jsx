import React, { useEffect } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import About from "./pages/About";
import Method from "./pages/Method";
import Patients from "./pages/Patients";
import Reliability from "./pages/Reliability";
import Results from "./pages/Results";
import Summary from "./pages/Summary";
import { SITE } from "./site";

const NAV = [
  { to: "/", label: "Summary", end: true },
  { to: "/results", label: "Results" },
  { to: "/method", label: "How it works" },
  { to: "/reliability", label: "Reliability" },
  { to: "/patients", label: "Patients" },
  { to: "/about", label: "About" },
];

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);
  return null;
}

export default function App() {
  const byline = [SITE.authors, SITE.affiliation].filter(Boolean).join(" · ");
  return (
    <>
      <ScrollToTop />
      <header className="masthead">
        <div className="masthead-inner">
          <NavLink to="/" className="site-title">{SITE.title}</NavLink>
          <div className="site-byline">{byline || SITE.subtitle}</div>
          <nav className="nav" aria-label="Main">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end}
                className={({ isActive }) => (isActive ? "active" : "")}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<Summary />} />
          <Route path="/results" element={<Results />} />
          <Route path="/method" element={<Method />} />
          <Route path="/reliability" element={<Reliability />} />
          <Route path="/patients" element={<Patients />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="footer-inner">
          <span>
            Data: Diabetes 130-US Hospitals, 1999 to 2008 (UCI Machine Learning
            Repository, CC BY 4.0).
          </span>
          <span>Research prototype. Not for clinical use.</span>
        </div>
      </footer>
    </>
  );
}
