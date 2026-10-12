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
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const get = (path) => request(path);
export const PAPER_URL = `${BASE}/paper.pdf`;

export const getPatients = (task) =>
  request(`/patients?task=${encodeURIComponent(task)}`);
export const getPatient = (id, task) =>
  request(`/patients/${encodeURIComponent(id)}?task=${encodeURIComponent(task)}`);
export const predict = (task, encounters) =>
  request("/predict", {
    method: "POST",
    body: JSON.stringify({ task, encounters }),
  });
