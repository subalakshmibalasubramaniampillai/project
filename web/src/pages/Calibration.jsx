import React, { useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { useFetch } from "../useFetch";
import { getCalibration } from "../api";
import { COLORS, ErrorBox, LABELS, Loading, SPLITS, Selector, TASKS, fmt } from "../common";

const SHOWN = ["logistic_regression", "lightgbm", "gru", "tkgn", "tkgn_b"];

export default function Calibration() {
  const { data, loading, error } = useFetch(getCalibration);
  const [task, setTask] = useState("readmit30");
  const [split, setSplit] = useState("grouped");
  if (loading) return <Loading what="calibration" />;
  if (error) return <ErrorBox error={error} />;

  const curves = data[`${task}_${split}`] || {};
  const models = SHOWN.filter((m) => curves[m]);
  const maxP = Math.max(0.05, ...models.flatMap((m) =>
    curves[m].flatMap((b) => [b.mean_predicted, b.observed])));

  return (
    <div className="stack">
      <div>
        <h2 className="page-title">Calibration</h2>
        <p className="page-sub">
          Reliability curves over ten equal-frequency risk bins of the test
          set. A well-calibrated model lies on the diagonal: among encounters
          given 20% risk, about 20% experience the outcome.
        </p>
      </div>
      <div className="toolbar">
        <Selector label="Task" value={task} onChange={setTask} options={TASKS} />
        <Selector label="Design" value={split} onChange={setSplit} options={SPLITS} />
      </div>
      <div className="card">
        <div className="chart-box" style={{ height: 420 }}>
          <ResponsiveContainer>
            <LineChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3550" />
              <XAxis type="number" dataKey="mean_predicted" domain={[0, maxP]}
                tickFormatter={(v) => v.toFixed(2)}
                label={{ value: "Mean predicted risk", position: "bottom", fill: "#9aa7c0" }}
                tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <YAxis type="number" domain={[0, maxP]} tickFormatter={(v) => v.toFixed(2)}
                tick={{ fill: "#9aa7c0", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#171f33", border: "1px solid #2a3550" }}
                formatter={(v) => fmt(v, 3)} />
              <Legend verticalAlign="top" />
              <Line data={[{ mean_predicted: 0, observed: 0 },
                { mean_predicted: maxP, observed: maxP }]}
                dataKey="observed" name="Perfect" stroke="#9aa7c0"
                strokeDasharray="4 4" dot={false} />
              {models.map((m) => (
                <Line key={m} data={curves[m]} dataKey="observed" name={LABELS[m]}
                  stroke={COLORS[m]} strokeWidth={2} dot />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
