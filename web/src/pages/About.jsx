import React from "react";
import { PAPER_URL } from "../api";
import { SITE } from "../site";

export default function About() {
  const byline = [SITE.authors, SITE.affiliation].filter(Boolean).join(" · ");
  return (
    <div className="wrap page">
      <span className="eyebrow">About</span>
      <h1 style={{ fontSize: "clamp(32px, 4vw, 46px)" }}>About {SITE.brand}</h1>
      {byline && <p className="lead">{byline}</p>}
      <p className="lead">
        {SITE.brand} is a research project on {SITE.title.toLowerCase()}. It
        combines a sequence model of each patient&apos;s hospital history with
        gradient boosting, and it was built and tested only on public,
        de-identified hospital records.
      </p>
      <div className="grid-2" style={{ marginTop: 32 }}>
        <div className="card">
          <h3>Documents and code</h3>
          <p><a href={PAPER_URL}>Manuscript (PDF)</a></p>
          <p><a href={SITE.repository}>Source code, results and models</a></p>
          <p>
            Data: <a href="https://archive.ics.uci.edu/dataset/296">Diabetes 130-US
            Hospitals for Years 1999 to 2008</a>, UCI Machine Learning Repository
            (CC BY 4.0). Strack B, et al. BioMed Research International.
            2014;2014:781670.
          </p>
        </div>
        <div className="card">
          <h3>Intended use</h3>
          <p>
            {SITE.brand} is a research prototype. It has not been validated for
            clinical care, is not a medical device, and must not be used on its
            own to make decisions about individual patients.
          </p>
          {SITE.contact && <p>Contact: <a href={`mailto:${SITE.contact}`}>{SITE.contact}</a></p>}
        </div>
      </div>
    </div>
  );
}
