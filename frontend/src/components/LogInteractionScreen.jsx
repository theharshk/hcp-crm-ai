import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { fetchHcps, fetchHistory, selectHcp, selectInteraction } from "../store/interactionsSlice";
import StructuredForm from "./StructuredForm.jsx";
import ChatInterface from "./ChatInterface.jsx";

export default function LogInteractionScreen() {
  const dispatch = useDispatch();
  const { hcps, selectedHcpId, history, selectedInteraction, lastSubmission, draftHcp } = useSelector((s) => s.interactions);
  const [mode, setMode] = useState("form"); // "form" | "chat"

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
              <select
                className="hcp-select"
                value={selectedHcpId || ""}
                onChange={(e) => dispatch(selectHcp(e.target.value))}
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
    </div>
  );
}
