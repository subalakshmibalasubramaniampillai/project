import React from "react";
import { Link } from "react-router-dom";
import { ErrorMsg, Loading } from "../components/ui";
import { pct } from "../format";
import { SITE } from "../site";
import { useApi } from "../useFetch";

export default function ForPatients() {
  const { data, error, loading } = useApi({ overview: "/overview" });
  if (loading) return <div className="wrap page"><Loading /></div>;
  if (error) return <div className="wrap page"><ErrorMsg error={error} /></div>;
  const d = data.overview.dataset;

  return (
    <div className="wrap page plain">
      <span className="eyebrow">For patients and families</span>
      <h1 style={{ fontSize: "clamp(32px, 4vw, 46px)" }}>Going home from hospital with diabetes</h1>
      <p className="lead">
        This page explains, in everyday words, what {SITE.brand} looks at and
        what you can ask your care team before you leave hospital.
      </p>

      <div className="grid-2" style={{ marginTop: 36 }}>
        <div className="card">
          <h2>What is a readmission?</h2>
          <p>
            A readmission is when someone goes back into hospital soon after
            leaving, usually within 30 days. In the hospital records used to
            build {SITE.brand}, this happened after about 1 in 9 stays
            ({pct(d.readmit30_rate)}) of people with diabetes.
          </p>
          <p>
            Many readmissions can be prevented with good planning: a follow-up
            appointment, clear medicine instructions and support at home.
          </p>
        </div>
        <div className="card">
          <h2>Why earlier hospital stays matter</h2>
          <p>
            Your past hospital stays tell a story: what you were treated for,
            how your diabetes medicines changed, and whether you needed to come
            back. {SITE.brand} reads that story, not only today&apos;s stay.
          </p>
          <p>
            In our tests, the earlier stays were the single most useful piece
            of information for predicting what happens next.
          </p>
        </div>
      </div>

      <section style={{ marginTop: 48 }}>
        <h2>What was linked with coming back to hospital</h2>
        <p>
          In the records of {d.cohort_encounters.toLocaleString("en-US")} hospital stays,
          these things were linked with a higher chance of readmission. They
          describe patterns across many people; they do not decide what will
          happen to you.
        </p>
        <div className="grid-3">
          <div className="card">
            <h3>Several hospital stays in the past year</h3>
            <p>The more admissions in the 12 months before, the higher the chance of another one.</p>
          </div>
          <div className="card">
            <h3>Going to a care facility</h3>
            <p>People who went to a rehabilitation or nursing facility after discharge came back more often than those who went home.</p>
          </div>
          <div className="card">
            <h3>Heart failure and emergency visits</h3>
            <p>A diagnosis of heart failure, and several emergency visits in the past year, were also linked with higher risk.</p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: 48 }}>
        <h2>When diabetes treatment needs to change</h2>
        <p>
          {SITE.brand} also estimates whether diabetes treatment is likely to
          be stepped up at a next hospital stay, for example by starting
          insulin or adding a medicine. This happened in about a third of
          repeat stays ({pct(d.escalation_rate)}). People whose diabetes
          medicines had recently been increased were more likely to need a
          further step-up; people already on a steady insulin dose were less
          likely.
        </p>
      </section>

      <section style={{ marginTop: 48 }}>
        <h2>Questions to ask before you leave hospital</h2>
        <ul className="question-list">
          <li>When is my follow-up appointment, and who is it with?</li>
          <li>Have any of my diabetes medicines changed, and why?</li>
          <li>How often should I check my blood sugar at home, and what numbers should worry me?</li>
          <li>Who do I call if my blood sugar is too high or too low, or if I feel worse?</li>
          <li>Do I need help at home, such as a nurse visit or equipment?</li>
          <li>Which warning signs mean I should come back to hospital straight away?</li>
        </ul>
      </section>

      <section style={{ marginTop: 48 }}>
        <div className="note">
          <h3 style={{ marginTop: 4 }}>What {SITE.brand} can and cannot do</h3>
          <p>
            It gives your care team an estimate based on patterns in past
            hospital records. It cannot examine you, does not know everything
            about your health, and is a research tool that has not been
            approved for routine care. Always follow the advice of your own
            doctors and nurses, and in an emergency call your local emergency
            number.
          </p>
        </div>
        <p style={{ marginTop: 22 }}>
          Health professionals can find the technical details on the{" "}
          <Link to="/evidence">evidence page</Link>.
        </p>
      </section>
    </div>
  );
}
