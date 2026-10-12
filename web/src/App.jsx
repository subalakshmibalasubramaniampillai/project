import React, { useEffect } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { PAPER_URL } from "./api";
import { Logo } from "./components/ui";
import About from "./pages/About";
import Cases from "./pages/Cases";
import Check from "./pages/Check";
import Evidence from "./pages/Evidence";
import ForPatients from "./pages/ForPatients";
import Home from "./pages/Home";
import { SITE } from "./site";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/patients", label: "For patients" },
  { to: "/cases", label: "Case studies" },
  { to: "/evidence", label: "Evidence" },
  { to: "/about", label: "About" },
];

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => window.scrollTo(0, 0), [pathname]);
  return null;
}

export default function App() {
  return (
    <>
      <ScrollToTop />
      <header className="topbar">
        <div className="wrap">
          <Link to="/" className="brand" aria-label={`${SITE.brand} home`}>
            <Logo />
            <span className="brand-name">{SITE.brand}</span>
          </Link>
          <nav className="nav" aria-label="Main">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end}
                className={({ isActive }) => (isActive ? "active" : "")}>
                {item.label}
              </NavLink>
            ))}
            <NavLink to="/check" className={({ isActive }) => `cta${isActive ? " active" : ""}`}>
              Check a patient
            </NavLink>
          </nav>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/check" element={<Check />} />
          <Route path="/patients" element={<ForPatients />} />
          <Route path="/cases" element={<Cases />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="wrap">
          <div>
            <h4>{SITE.brand}</h4>
            <p>{SITE.tagline}.</p>
          </div>
          <div>
            <h4>Use</h4>
            <p><Link to="/check">Check a patient</Link></p>
            <p><Link to="/patients">Guide for patients and families</Link></p>
            <p><Link to="/cases">Case studies</Link></p>
          </div>
          <div>
            <h4>Research</h4>
            <p><Link to="/evidence">Evidence</Link></p>
            <p><a href={PAPER_URL}>Manuscript (PDF)</a></p>
            <p><a href={SITE.repository}>Code</a></p>
          </div>
          <div>
            <h4>Important</h4>
            <p>Research prototype, not a medical device. Data: Diabetes 130-US Hospitals, UCI (CC BY 4.0).</p>
          </div>
        </div>
      </footer>
    </>
  );
}
