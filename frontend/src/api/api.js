const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  listHcps: () => request("/hcps"),
  createHcp: (payload) => request("/hcps", { method: "POST", body: JSON.stringify(payload) }),

  listInteractions: (hcpId) => request(`/interactions${hcpId ? `?hcp_id=${hcpId}` : ""}`),
  createInteraction: (payload) =>
    request("/interactions", { method: "POST", body: JSON.stringify(payload) }),
  updateInteraction: (id, payload) =>
    request(`/interactions/${id}`, { method: "PATCH", body: JSON.stringify(payload) }),
  checkCompliance: (id) => request(`/interactions/${id}/compliance`),

  sendChatMessage: (payload) => request("/chat", { method: "POST", body: JSON.stringify(payload) }),
};
