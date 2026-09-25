import React from "react";
import { useFetch } from "../useFetch";
import { getClusters } from "../api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell,
} from "recharts";

const COLORS = ["#4f8ef7", "#38d9a9", "#f4a340", "#a78bfa", "#e5484d"];

function fmt(v) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(3);
}

export default function Clustering() {
  const { data, loading, error } = useFetch(getClusters);

  if (loading) return <div className="loading">Loading…</div>;
  if (error) return <div className="alert error">{error}</div>;

  if (!data.clusters) {
    return (
      <div className="stack">
        <h2 className="page-title">Patient trajectory clusters</h2>
        <div className="alert warn">
          No trajectory clustering available for this run.
        </div>
      </div>
    );
  }

  const clusters = data.clusters;
  const counts = Object.entries(data.counts || {}).map(([cluster, n]) => ({
    cluster: `C${cluster}`,
    patients: n,
  }));

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Patient trajectory clusters</h2>
        <p className="page-sub">
          Patients grouped by HbA1c trajectory shape (slope + curvature)
          using KMeans.
        </p>
      </div>

      <div className="card">
        <h3>Patients per cluster</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={counts} margin={{ left: -18, right: 12 }}>
            <CartesianGrid stroke="#2a3550" strokeDasharray="3 3" />
            <XAxis dataKey="cluster" stroke="#9aa7c0" />
            <YAxis stroke="#9aa7c0" />
            <Tooltip
              contentStyle={{ background: "#1d2740", border: "1px solid #2a3550" }}
              labelStyle={{ color: "#e6ebf5" }}
            />
            <Bar dataKey="patients" radius={[4, 4, 0, 0]}>
              {counts.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Cluster characteristics</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cluster</th>
                <th>Patients</th>
                <th>Mean HbA1c slope</th>
                <th>Progression rate</th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((c) => (
                <tr key={c.traj_cluster}>
                  <td>
                    <span className="pill baseline">C{c.traj_cluster}</span>
                  </td>
                  <td className="num">{c.n_patients}</td>
                  <td className="num">{fmt(c.hba1c_slope)}</td>
                  <td className="num">{fmt(c.target)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}