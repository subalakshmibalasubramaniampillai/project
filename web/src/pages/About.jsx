import React from "react";
import { PAPER_URL } from "../api";
import { Section } from "../components/ui";
import { SITE } from "../site";

export default function About() {
  const byline = [SITE.authors, SITE.affiliation].filter(Boolean).join(" · ");
  return (
    <>
      <div className="intro">
        <div className="kicker">About this project</div>
        <h1>{SITE.title}</h1>
        {byline && <p className="lead">{byline}</p>}
        <p>
          This site presents a study of 30-day readmission and treatment
          escalation in people with diabetes, using the public Diabetes 130-US
          Hospitals records. It accompanies a manuscript that describes the
          methods and results in full.
        </p>
      </div>

      <Section num="1" title="Documents and code">
        <table>
          <tbody>
            <tr>
              <td>Manuscript</td>
              <td><a href={PAPER_URL}>Download the PDF</a></td>
            </tr>
            <tr>
              <td>Source code and results</td>
              <td><a href={SITE.repository}>{SITE.repository.replace("https://", "")}</a></td>
            </tr>
            <tr>
              <td>Dataset</td>
              <td>
                <a href="https://archive.ics.uci.edu/dataset/296">
                  Diabetes 130-US Hospitals for Years 1999 to 2008
                </a>{" "}
                (UCI Machine Learning Repository, CC BY 4.0)
              </td>
            </tr>
            <tr>
              <td>Original data paper</td>
              <td>
                Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c
                measurement on hospital readmission rates: analysis of 70,000
                clinical database patient records. BioMed Research International.
                2014;2014:781670.
              </td>
            </tr>
          </tbody>
        </table>
      </Section>

      <Section num="2" title="Intended use">
        <p>
          The models and this website are research prototypes. They have not
          been validated for clinical care, are not a medical device, and must
          not be used to make decisions about individual patients.
        </p>
        <p>
          The records are de-identified and were released for research by the
          original authors. No new patient data were collected for this project.
        </p>
      </Section>

      {SITE.contact && (
        <Section num="3" title="Contact">
          <p><a href={`mailto:${SITE.contact}`}>{SITE.contact}</a></p>
        </Section>
      )}
    </>
  );
}
