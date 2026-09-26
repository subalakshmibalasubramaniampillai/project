const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => d.msg).join("; ")
      : body.detail;
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const getOverview = () => request("/overview");
export const getSummary = () => request("/summary");
export const getSignificance = () => request("/significance");
export const getRepeatTests = () => request("/repeat-tests");
export const getAblation = () => request("/ablation");
export const getSubgroups = () => request("/subgroups");
export const getCalibration = () => request("/calibration");
export const getLearningCurve = () => request("/learning-curve");
export const getGates = () => request("/gates");
export const getExplanations = () => request("/explanations");
export const getGraph = () => request("/graph");
export const getSchema = () => request("/schema");
export const getPatients = (task) =>
  request(`/patients?task=${encodeURIComponent(task)}`);
export const getPatient = (id, task) =>
  request(`/patients/${encodeURIComponent(id)}?task=${encodeURIComponent(task)}`);
export const predict = (task, encounters) =>
  request("/predict", {
    method: "POST",
    body: JSON.stringify({ task, encounters }),
  });
