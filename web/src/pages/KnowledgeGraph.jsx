import React, { useMemo, useState } from "react";
import { useFetch } from "../useFetch";
import { getGraph } from "../api";
import { ErrorBox, Loading, fmt } from "../common";

function CooccurrenceMap({ edges }) {
  // simple circular layout of the strongest co-morbidity edges
  const nodes = useMemo(() => {
    const ids = [...new Set(edges.flatMap((e) => [e.a, e.b]))];
    return ids.map((id, i) => {
      const angle = (2 * Math.PI * i) / ids.length;
      return { id, x: 250 + 200 * Math.cos(angle), y: 250 + 200 * Math.sin(angle) };
    });
  }, [edges]);
  const pos = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const maxP = Math.max(...edges.map((e) => e.ppmi));
  return (
    <svg viewBox="0 0 500 500" style={{ width: "100%", maxWidth: 520 }}
      role="img" aria-label="Co-morbidity graph">
      {edges.map((e) => (
        <line key={`${e.a}-${e.b}`} x1={pos[e.a].x} y1={pos[e.a].y}
          x2={pos[e.b].x} y2={pos[e.b].y} stroke="#38d9a9"
          strokeOpacity={0.25 + 0.6 * (e.ppmi / maxP)} strokeWidth={1 + 2 * (e.ppmi / maxP)} />
      ))}
      {nodes.map((n) => (
        <g key={n.id}>
          <circle cx={n.x} cy={n.y} r={11} fill="#1d2740" stroke="#4f8ef7" />
          <text x={n.x} y={n.y + 3} textAnchor="middle" fontSize="8" fill="#e6ebf5">{n.id}</text>
        </g>
      ))}
    </svg>
  );
}

export default function KnowledgeGraph() {
  const { data, loading, error } = useFetch(getGraph);
  const [filter, setFilter] = useState("");
  if (loading) return <Loading what="knowledge graph" />;
  if (error) return <ErrorBox error={error} />;

  const edges = data.top_cooccurrence.filter(
    (e) => !filter || e.a.startsWith(filter) || e.b.startsWith(filter));
  const levels = data.nodes_by_level;

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Clinical knowledge graph</h2>
        <p className="page-sub">
          Ontology edges come from the public ICD-9-CM hierarchy
          (code → subcategory → category → chapter) and the pharmacological
          classes of anti-diabetic drugs. Co-morbidity edges are positive PMI
          between diagnosis categories counted on training encounters only.
        </p>
      </div>
      <div className="stat-row">
        <div className="stat"><div className="label">Concept nodes</div>
          <div className="value">{data.n_nodes}</div></div>
        <div className="stat"><div className="label">Distinct ICD-9 codes</div>
          <div className="value">{data.n_codes}</div></div>
        <div className="stat"><div className="label">Categories</div>
          <div className="value">{levels.category}</div></div>
        <div className="stat"><div className="label">Chapters</div>
          <div className="value">{levels.chapter}</div></div>
        <div className="stat"><div className="label">Co-morbidity edges</div>
          <div className="value">{data.n_cooccurrence_edges}</div></div>
      </div>
      <div className="grid grid-2">
        <div className="card">
          <h3>Strongest co-morbidity links</h3>
          <CooccurrenceMap edges={data.top_cooccurrence.slice(0, 30)} />
        </div>
        <div className="card">
          <h3>Co-morbidity edges (PPMI, ≥50 shared encounters)</h3>
          <div className="field" style={{ marginBottom: 8 }}>
            <label>Filter by ICD-9 category prefix</label>
            <input value={filter} placeholder="e.g. 250 or 4"
              onChange={(e) => setFilter(e.target.value.trim())} />
          </div>
          <div className="table-wrap" style={{ maxHeight: 420, overflowY: "auto" }}>
            <table>
              <thead><tr><th>Category A</th><th>Category B</th><th>PPMI</th><th>Encounters</th></tr></thead>
              <tbody>
                {edges.map((e) => (
                  <tr key={`${e.a}-${e.b}`}>
                    <td>{e.a}</td><td>{e.b}</td>
                    <td className="num">{fmt(e.ppmi, 2)}</td>
                    <td className="num">{e.count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="card">
        <h3>ICD-9-CM chapters (training encounters mentioning each)</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Chapter</th><th>Description</th><th>Encounters</th></tr></thead>
            <tbody>
              {data.chapters.map((c) => (
                <tr key={c.id}><td>{c.id}</td><td>{c.name}</td>
                  <td className="num">{c.count.toLocaleString()}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
