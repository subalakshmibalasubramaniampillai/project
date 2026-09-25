const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const getMetrics = () => request("/metrics");
export const getAblation = () => request("/ablation");
export const getSplit = () => request("/split");
export const getFusion = () => request("/fusion");
export const getExplanation = () => request("/explanation");
export const getPatients = () => request("/patients");
export const getDataset = (patientId) =>
  request(`/dataset${patientId ? `?patient_id=${encodeURIComponent(patientId)}` : ""}`);
export const getClusters = () => request("/clusters");
export const getGraph = () => request("/graph");

export const predict = (visits, useTagnn) =>
  request("/predict", {
    method: "POST",
    body: JSON.stringify({ visits, use_tagnn: useTagnn }),
  });

export const ingest = (payload) =>
  request("/ingest", { method: "POST", body: JSON.stringify(payload) });