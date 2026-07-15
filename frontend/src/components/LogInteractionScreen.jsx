import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { fetchHcps, fetchHistory, selectHcp, selectInteraction, createHcp } from "../store/interactionsSlice";
import StructuredForm from "./StructuredForm.jsx";
import ChatInterface from "./ChatInterface.jsx";

export default function LogInteractionScreen() {
  const dispatch = useDispatch();
  const { hcps, selectedHcpId, history, selectedInteraction, lastSubmission, draftHcp } = useSelector((s) => s.interactions);
  const [mode, setMode] = useState("form"); // "form" | "chat"
  const [showModal, setShowModal] = useState(false);
  const [newHcpData, setNewHcpData] = useState({ name: "", specialty: "", institution: "", email: "", phone: "" });

  const handleCreateHcp = (e) => {
    e.preventDefault();
    if (!newHcpData.name.trim()) return;
    dispatch(createHcp(newHcpData));
    setShowModal(false);
    setNewHcpData({ name: "", specialty: "", institution: "", email: "", phone: "" });
  };

  // Initial load: fetch all doctors
  useEffect(() => {
    dispatch(fetchHcps());
  }, [dispatch]);

  // Re-fetch interaction history whenever the selected doctor changes
  useEffect(() => {
    if (selectedHcpId) dispatch(fetchHistory(selectedHcpId));
  }, [dispatch, selectedHcpId]);

  // Re-fetch history AND hcp list live whenever a new interaction is saved via chat
  // This ensures pre-seeded doctors and newly-enriched profiles update in real-time
  useEffect(() => {
    if (!lastSubmission) return;
    const hcpId = lastSubmission.hcp_id || selectedHcpId;
    if (hcpId) dispatch(fetchHistory(hcpId));
    dispatch(fetchHcps());
  }, [dispatch, lastSubmission]);

  const selectedHcp = hcps.find((h) => h.id === selectedHcpId) || draftHcp;

  const capitalizeSpecialty = (s) => {
    if (!s) return "General";
    return s.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  };

  return (
    <div className="screen">
      <div className="screen-header">
        <div>
          <p className="eyebrow">Log Interaction</p>
          <h1>Record an HCP interaction</h1>
          <p>The form on the left is controlled and populated automatically by the AI assistant on the right.</p>
        </div>
      </div>

      <div className="layout split-layout">
        {/* Left Side: HCP & Interaction Details Form */}
        <div className="left-panel">
          <div className="left-panel-top-row">
            {/* HCP Profile */}
            <div className="card hcp-card">
              <p className="section-title">Healthcare Professional Profile</p>
              <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
                <select
                  className="hcp-select"
                  value={selectedHcpId || ""}
                  onChange={(e) => dispatch(selectHcp(e.target.value))}
                  style={{ marginBottom: 0 }}
                >
                  <option value="" disabled>
                    {draftHcp ? `${draftHcp.name} (Drafting...)` : "Select or type name in chat to resolve…"}
                  </option>
                  {hcps.map((h) => (
                    <option key={h.id} value={h.id}>
                      {h.name} — {capitalizeSpecialty(h.specialty)}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => setShowModal(true)}
                  style={{
                    padding: "0 14px",
                    height: "41px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "20px",
                    lineHeight: 1,
                    borderRadius: "8px",
                  }}
                  title="Add New Doctor Profile"
                >
                  +
                </button>
              </div>

              {selectedHcp ? (
                <div className="hcp-profile-info">
                  <h3 style={{ fontSize: "16px" }}>{selectedHcp.name}</h3>
                  <div style={{ fontSize: "12.5px", display: "flex", flexDirection: "column", gap: "4px", marginTop: "8px" }}>
                    <div><strong>Specialty:</strong> {capitalizeSpecialty(selectedHcp.specialty)}</div>
                    <div><strong>Institution:</strong> {selectedHcp.institution || "—"}</div>
                    <div><strong>Email:</strong> {selectedHcp.email || "—"}</div>
                    <div><strong>Phone:</strong> {selectedHcp.phone || "—"}</div>
                  </div>
                </div>
              ) : (
                <p className="empty-state" style={{ marginTop: "8px", fontSize: "12.5px" }}>No HCP selected.</p>
              )}
            </div>

            {/* Recent Interactions / History */}
            <div className="card history-card">
              <p className="section-title">Recent Interactions</p>
              <div className="history-list" style={{ maxHeight: "150px", overflowY: "auto" }}>
                {history.length === 0 && <p className="empty-state" style={{ fontSize: "12.5px" }}>No interactions logged yet.</p>}
                {history.slice(0, 5).map((h) => {
                  const isActive = selectedInteraction?.id === h.id;
                  return (
                    <div
                      className={`history-item ${isActive ? "active" : ""}`}
                      key={h.id}
                      onClick={() => dispatch(selectInteraction(h))}
                      style={{
                        padding: "8px",
                        fontSize: "12.5px",
                        cursor: "pointer",
                        borderRadius: "6px",
                        marginBottom: "6px",
                        backgroundColor: isActive ? "rgba(0, 102, 102, 0.08)" : "#f9f9f9",
                        border: isActive ? "1px solid rgba(0, 102, 102, 0.2)" : "1px solid #eee",
                        transition: "all 0.2s ease"
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#666", fontWeight: 500 }}>
                        <span>{h.interaction_date ? new Date(h.interaction_date).toLocaleDateString("en-US", { day: 'numeric', month: 'short', year: 'numeric' }) : "—"}</span>
                        <span style={{ textTransform: "capitalize" }}>{h.sentiment}</span>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                        <span className="chip" style={{ padding: "2px 6px", fontSize: "10px", margin: 0, backgroundColor: "#e2f0f0", color: "#006666" }}>
                          {h.interaction_type}
                        </span>
                      </div>
                      <div style={{ marginTop: "6px", fontWeight: 500, color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {h.topics_discussed ? h.topics_discussed.join(", ") : h.summary || "—"}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          
          <div className="card form-card">
            <StructuredForm />
          </div>
        </div>

        {/* Right Side: Chat Panel */}
        <div className="right-panel card chat-card">
          <ChatInterface selectedHcpId={selectedHcpId} />
        </div>
      </div>

      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2 style={{ margin: "0 0 16px", fontSize: "18px", fontWeight: "700" }}>Add New HCP Profile</h2>
            <form onSubmit={handleCreateHcp}>
              <div className="field">
                <label>Doctor Name (Required)</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Dr. Foreman Kumar"
                  value={newHcpData.name}
                  onChange={(e) => setNewHcpData({ ...newHcpData, name: e.target.value })}
                />
              </div>
              <div className="field">
                <label>Specialty</label>
                <input
                  type="text"
                  placeholder="e.g. Neurosurgeon"
                  value={newHcpData.specialty}
                  onChange={(e) => setNewHcpData({ ...newHcpData, specialty: e.target.value })}
                />
              </div>
              <div className="field">
                <label>Institution</label>
                <input
                  type="text"
                  placeholder="e.g. Princeton Plainsboro"
                  value={newHcpData.institution}
                  onChange={(e) => setNewHcpData({ ...newHcpData, institution: e.target.value })}
                />
              </div>
              <div className="row-2">
                <div className="field">
                  <label>Email</label>
                  <input
                    type="email"
                    placeholder="e.g. email@hospital.com"
                    value={newHcpData.email}
                    onChange={(e) => setNewHcpData({ ...newHcpData, email: e.target.value })}
                  />
                </div>
                <div className="field">
                  <label>Phone</label>
                  <input
                    type="text"
                    placeholder="e.g. 123-456-7890"
                    value={newHcpData.phone}
                    onChange={(e) => setNewHcpData({ ...newHcpData, phone: e.target.value })}
                  />
                </div>
              </div>
              <div style={{ display: "flex", gap: "10px", justifyContent: "flex-end", marginTop: "24px" }}>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => {
                    setShowModal(false);
                    setNewHcpData({ name: "", specialty: "", institution: "", email: "", phone: "" });
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Create Profile
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
