import React from "react";
import { Link } from "react-router-dom";
import { ErrorMsg, Icon, Loading } from "../components/ui";
import { count, num, pct } from "../format";
import { SITE } from "../site";
import { useApi } from "../useFetch";

function HeroArt() {
  // An illustration of the idea, not a real patient.
  const stays = [
    { x: 40, label: "Stay 1", sub: "Heart failure", event: true },
    { x: 140, label: "Stay 2", sub: "Pneumonia", event: false },
    { x: 240, label: "Stay 3", sub: "High blood sugar", event: true },
  ];
  return (
    <div className="hero-art">
      <div className="small muted" style={{ marginBottom: 10 }}>Illustration</div>
      <svg viewBox="0 0 420 250" width="100%" role="img"
        aria-label="A patient's earlier stays feeding into today's risk estimate">
        <line x1="40" y1="64" x2="360" y2="64" stroke="#cfc4b0" strokeWidth="2" />
        {stays.map((s) => (
          <g key={s.label}>
            <circle cx={s.x} cy="64" r="11" fill={s.event ? "#2d6550" : "#fffdf8"}
              stroke="#2d6550" strokeWidth="3" />
            <text x={s.x} y="34" textAnchor="middle" fontSize="15" fontWeight="700" fill="#1d2420">{s.label}</text>
            <text x={s.x} y="98" textAnchor="middle" fontSize="13" fill="#4c544e">{s.sub}</text>
          </g>
        ))}
        <circle cx="360" cy="64" r="16" fill="#c4552e" />
        <text x="360" y="34" textAnchor="middle" fontSize="15" fontWeight="700" fill="#c4552e">Today</text>
        <text x="360" y="98" textAnchor="middle" fontSize="13" fill="#4c544e">Discharge</text>
        <path d="M360 108 L360 128" stroke="#c4552e" strokeWidth="2" />
        <path d="M354 122 L360 130 L366 122" stroke="#c4552e" strokeWidth="2" fill="none" />
        <rect x="200" y="136" width="200" height="74" rx="12" fill="#fffdf8" stroke="#e4dccd" />
        <text x="216" y="160" fontSize="13" fill="#4c544e">Readmission in 30 days</text>
        <text x="216" y="194" fontSize="28" fontWeight="700" fill="#1d2420" fontFamily="Fraunces Variable, Georgia, serif">24%</text>
        <rect x="292" y="176" width="92" height="24" rx="12" fill="#b23a2a" />
        <text x="338" y="193" fontSize="13" fontWeight="700" fill="#fff" textAnchor="middle">Higher</text>
        <circle cx="48" cy="150" r="7" fill="#2d6550" />
        <text x="62" y="155" fontSize="13" fill="#4c544e">readmitted within</text>
        <text x="62" y="172" fontSize="13" fill="#4c544e">30 days after that stay</text>
        <circle cx="48" cy="198" r="7" fill="#fffdf8" stroke="#2d6550" strokeWidth="2.5" />
        <text x="62" y="203" fontSize="13" fill="#4c544e">not readmitted</text>
      </svg>
    </div>
  );
}

export default function Home() {
  const { data, error, loading } = useApi({
    overview: "/overview",
    summary: "/summary",
    bands: "/risk-bands",
    repeats: "/repeat-tests",
    ablation: "/ablation",
    gates: "/gates",
  });
  if (loading) return <div className="wrap page"><Loading /></div>;
  if (error) return <div className="wrap page"><ErrorMsg error={error} /></div>;

  const d = data.overview.dataset;
  const testN = data.overview.splits.readmit30_grouped.n.test;
  const rb = data.bands.readmit30.bands;
  const ratio = rb[2].observed_rate / rb[0].observed_rate;
  const auc = (t, m, sp = "grouped") =>
    data.summary.find((r) => r.task === t && r.split === sp && r.model === m)?.auroc;
  const abl = (t) => -data.ablation.find((r) => r.task === t && r.model === "tkgn_no_history").delta_auroc_vs_full;
  const gate = (t, k) => data.gates.find((g) => g.task === t && g.prior_encounters === k)?.gate_history;
  const beatSeq = ["gru", "retain", "transformer"].every((m) => ["readmit30", "escalation"].every((t) => {
    const r = data.repeats.find((x) => x.task === t && x.split === "grouped" && x.reference === "tkgn_b" && x.model === m);
    return r && r.wins === r.n_repeats;
  }));

  const novelties = [
    {
      icon: "history",
      title: "Reads the whole hospital history",
      text: "Most discharge risk scores look only at the current stay. StayTrace also reads up to ten earlier stays of the same patient, in order, including how each of them ended.",
      evidence: `Taking the history away lowered accuracy (AUROC) by ${num(abl("readmit30"))} for readmission and ${num(abl("escalation"))} for treatment step-up, more than removing any other part.`,
    },
    {
      icon: "gate",
      title: "Knows how much to trust that history",
      text: "A learned gate decides, patient by patient, how much the past should change today's estimate, taking into account how many earlier stays there are. One model serves first-time and frequent patients alike.",
      evidence: `For treatment step-up the gate opened further as the history grew, from ${num(gate("escalation", "0"), 2)} with no earlier stay to ${num(gate("escalation", "5+"), 2)} with five or more.`,
    },
    {
      icon: "layers",
      title: "Built on top of the strongest standard method",
      text: "StayTrace starts from the score of a gradient-boosting model (LightGBM) and adds only what the history reveals. It keeps what already works and corrects what it misses.",
      evidence: beatSeq
        ? "It scored higher than three published history-based models (GRU, RETAIN, Transformer) in every repeat, for both questions."
        : "It scored higher than three published history-based models in most repeats.",
    },
    {
      icon: "split",
      title: "Answers two questions at once",
      text: "Besides readmission within 30 days, it estimates whether diabetes treatment will need to be stepped up at the next stay, a question that readmission tools do not usually address.",
      evidence: `On the newest admissions, the step-up estimate reached an AUROC of ${num(auc("escalation", "tkgn_b", "temporal"))}, the best of all ten models tested.`,
    },
    {
      icon: "shield",
      title: "Tested the way a clinical tool should be",
      text: "Checked on patients it never saw, on the most recent admissions, for honest percentages, and separately for women and men, racial groups and age groups.",
      evidence: `Patients it placed in its higher band were readmitted ${ratio.toFixed(1)} times as often as those in its lower band.`,
    },
  ];

  return (
    <>
      <section className="hero">
        <div className="wrap">
          <div>
            <span className="eyebrow">For care teams looking after people with diabetes</span>
            <h1>Know who is likely to come back, before they leave hospital.</h1>
            <p className="lead">
              {SITE.brand} estimates two risks at discharge: being readmitted
              within 30 days, and needing stronger diabetes treatment at the next
              stay. Unlike most tools, it learns from the patient&apos;s earlier
              hospital stays, not just today&apos;s.
            </p>
            <div className="hero-actions">
              <Link className="btn" to="/check">Check a patient <Icon name="arrow" size={18} /></Link>
              <Link className="btn ghost" to="/patients">I am a patient or carer</Link>
            </div>
          </div>
          <HeroArt />
        </div>
      </section>

      <section className="band">
        <div className="wrap">
          <div className="trust">
            <div><strong>{count(d.cohort_encounters)}</strong><span>real hospital stays used to build and test it</span></div>
            <div><strong>130</strong><span>US hospitals in the source records</span></div>
            <div><strong>{count(testN)}</strong><span>test stays from patients it had never seen</span></div>
            <div><strong>{ratio.toFixed(1)}×</strong><span>readmission rate in its higher band compared with its lower band</span></div>
          </div>
        </div>
      </section>

      <section className="band paper">
        <div className="wrap">
          <div className="band-head">
            <span className="eyebrow">What is new</span>
            <h2>What makes {SITE.brand} different</h2>
            <p className="lead">
              Five ideas developed in this project, each with what the tests on
              real patient records showed.
            </p>
          </div>
          <div className="grid-3">
            {novelties.map((n) => (
              <div className="card" key={n.title}>
                <div className="icon-badge"><Icon name={n.icon} /></div>
                <span className="new-tag">New</span>
                <h3>{n.title}</h3>
                <p>{n.text}</p>
                <div className="evidence">{n.evidence}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="band">
        <div className="wrap">
          <div className="band-head">
            <span className="eyebrow">How it works</span>
            <h2>Three steps at discharge</h2>
          </div>
          <ol className="steps">
            <li>
              <h3>Describe today&apos;s stay</h3>
              <p>Age, diagnoses, length of stay, medicines, HbA1c and where the patient goes after discharge.</p>
            </li>
            <li>
              <h3>Add earlier stays</h3>
              <p>Each earlier admission and whether it was followed by a readmission. More history gives a better-informed estimate.</p>
            </li>
            <li>
              <h3>Read the two risks</h3>
              <p>Each risk comes as a percentage, a band (lower, moderate or higher) and the main reasons behind it.</p>
            </li>
          </ol>
          <div className="hero-actions">
            <Link className="btn" to="/check">Open the risk check</Link>
          </div>
        </div>
      </section>

      <section className="band tint">
        <div className="wrap grid-2" style={{ alignItems: "center" }}>
          <div>
            <span className="eyebrow">Does it work?</span>
            <h2>The bands separate patients clearly</h2>
            <p>
              Among {count(data.bands.readmit30.n)} test stays, patients in{" "}
              {SITE.brand}&apos;s higher band were readmitted within 30 days{" "}
              {pct(rb[2].observed_rate)} of the time. In the lower band it
              was {pct(rb[0].observed_rate)}. The overall rate was{" "}
              {pct(data.bands.readmit30.overall_rate)}.
            </p>
            <p>
              Ranking accuracy (AUROC) was {num(auc("readmit30", "tkgn_b"))} for
              readmission and {num(auc("escalation", "tkgn_b"))} for treatment
              step-up, the highest of the ten models compared, though only
              slightly ahead of the best standard method.
            </p>
            <Link to="/evidence">See the full evidence</Link>
          </div>
          <div className="card">
            <h3 style={{ fontSize: 18 }}>Readmitted within 30 days, by risk band</h3>
            {rb.map((b) => (
              <div key={b.band} style={{ margin: "14px 0" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 15 }}>
                  <span><strong style={{ textTransform: "capitalize" }}>{b.band}</strong>{" "}
                    <span className="muted">({pct(b.share, 0)} of patients)</span></span>
                  <strong>{pct(b.observed_rate)}</strong>
                </div>
                <div style={{ background: "#efe6d6", borderRadius: 6, height: 12, marginTop: 6 }}>
                  <div style={{
                    width: `${(b.observed_rate / rb[2].observed_rate) * 100}%`, height: "100%", borderRadius: 6,
                    background: { lower: "var(--risk-low)", moderate: "var(--risk-mid)", higher: "var(--risk-high)" }[b.band],
                  }} />
                </div>
              </div>
            ))}
            <p className="small muted" style={{ marginTop: 12 }}>
              Bands are cut at the middle and the top fifth of predicted risk.
            </p>
          </div>
        </div>
      </section>

      <section className="band">
        <div className="wrap grid-2">
          <div className="card">
            <div className="icon-badge"><Icon name="stethoscope" /></div>
            <h3>For doctors and nurses</h3>
            <p>Use the risk check at discharge to see which patients may need an early follow-up, a medicines review or more support at home.</p>
            <Link className="btn small" to="/check">Check a patient</Link>
          </div>
          <div className="card">
            <div className="icon-badge"><Icon name="person" /></div>
            <h3>For patients and families</h3>
            <p>A plain guide to what readmission means, why earlier hospital stays matter, and questions to ask before going home.</p>
            <Link className="btn small ghost" to="/patients">Read the guide</Link>
          </div>
        </div>
      </section>
    </>
  );
}
