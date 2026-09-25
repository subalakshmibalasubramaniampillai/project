import React, { useState } from "react";
import { useFetch } from "../useFetch";
import { getGraph } from "../api";

export default function KnowledgeGraph() {
  const { data, loading, error } = useFetch(getGraph);
  const [selected, setSelected] = useState("");

  if (loading) return <div className="loading">Loading knowledge graph…</div>;
  if (error) return <div className="alert error">{error}</div>;

  const graph = data;
  // a node is a patient if it does not start with "condition:"
  const isPatient = (id) => !String(id).startsWith("condition:");

  const neighborsOf = (id) => {
    const seen = new Map();
    graph.edges.forEach((e) => {
      if (e.source === id) seen.set(e.target, e);
      if (e.target === id) seen.set(e.source, e);
    });
    return Array.from(seen.values());
  };

  const selectedNeighbors = selected ? neighborsOf(selected) : [];

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Knowledge graph</h2>
        <p className="page-sub">
          {graph.node_count} nodes · {graph.edge_count} edges. Condition
          links are dataset observations; shared-condition similarity edges
          are modeling assumptions.
        </p>
      </div>

      <div className="card" style={{ maxWidth: 520 }}>
        <div className="field">
          <label>Inspect node</label>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            <option value="">Select a node…</option>
            {graph.nodes.map((id) => (
              <option key={id} value={id}>
                {id} {isPatient(id) ? "" : "(condition)"}
              </option>
            ))}
          </select>
        </div>
      </div>

      {selected && (
        <div className="card">
          <h3>
            {selected}{" "}
            <span className={`pill ${isPatient(selected) ? "baseline" : "novel"}`}>
              {isPatient(selected) ? "patient" : "condition"}
            </span>
          </h3>
          <p>{selectedNeighbors.length} neighbors</p>
          {selectedNeighbors.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Neighbor</th>
                    <th>Relation</th>
                    <th>Justification</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedNeighbors.map((edge, i) => (
                    <tr key={i}>
                      <td>
                        {edge.source === selected ? edge.target : edge.source}
                      </td>
                      <td>{edge.relation}</td>
                      <td>{edge.justification}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty">No neighbors.</div>
          )}
        </div>
      )}

      <div className="card">
        <h3>Edges (first 100 of {graph.edge_count})</h3>
        <div className="table-wrap" style={{ maxHeight: 380, overflowY: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Relation</th>
                <th>Target</th>
                <th>Justification</th>
              </tr>
            </thead>
            <tbody>
              {graph.edges.slice(0, 100).map((e, i) => (
                <tr key={i}>
                  <td>{e.source}</td>
                  <td>{e.relation}</td>
                  <td>{e.target}</td>
                  <td>{e.justification}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}