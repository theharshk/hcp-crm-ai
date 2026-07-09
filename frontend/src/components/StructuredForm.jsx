import React from "react";
import { useSelector } from "react-redux";

export default function StructuredForm() {
  const { lastSubmission, selectedInteraction, draftInteraction, complianceWarnings, history } = useSelector((s) => s.interactions);

  const isSelected = !!selectedInteraction || !!draftInteraction;
  const isDraft = !!draftInteraction && !selectedInteraction;

  const interaction = selectedInteraction || draftInteraction || {
    interaction_type: "",
    sentiment: "",
    summary: "",
    topics_discussed: [],
    products_discussed: [],
    raw_notes: "",
    samples_provided: [],
  };

  const getPreferredType = () => {
    if (!history || history.length === 0) return "—";
    const counts = {};
    history.forEach((h) => {
      if (h.interaction_type) {
        counts[h.interaction_type] = (counts[h.interaction_type] || 0) + 1;
      }
    });
    const keys = Object.keys(counts);
    if (keys.length === 0) return "—";
    return keys.reduce((a, b) => (counts[a] > counts[b] ? a : b));
  };

  const lastInteraction = history && history.length > 0 ? history[0] : null;
  const lastInteractionDate = lastInteraction
    ? new Date(lastInteraction.interaction_date).toLocaleDateString("en-US", { day: 'numeric', month: 'short', year: 'numeric' })
    : "—";

  const topicsStr = isSelected
    ? (Array.isArray(interaction.topics_discussed) && interaction.topics_discussed.length > 0
        ? interaction.topics_discussed.join(", ")
        : interaction.topics_discussed && interaction.topics_discussed.length > 0 ? interaction.topics_discussed : "")
    : "";

  const productsStr = isSelected
    ? (Array.isArray(interaction.products_discussed) && interaction.products_discussed.length > 0
        ? interaction.products_discussed.join(", ")
        : interaction.products_discussed && interaction.products_discussed.length > 0 ? interaction.products_discussed : "")
    : "";

  const samples = isSelected ? (interaction.samples_provided || []) : [];

  if (!isSelected) {
    // Mode 1: HCP Overview Mode
    return (
      <div>
        <p className="section-title">Healthcare Professional Overview</p>
        
        <div className="compliance-banner" style={{ background: "var(--teal-dim)", color: "var(--teal)", marginBottom: "20px" }}>
          Select any "Recent Interaction" above to view its detailed logs and raw notes.
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">Total Interactions</span>
            <span className="stat-value">{history.length}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Last Interaction</span>
            <span className="stat-value" style={{ fontSize: "16px" }}>{lastInteractionDate}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Preferred Type</span>
            <span className="stat-value" style={{ fontSize: "16px" }}>{getPreferredType()}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <p className="section-title">
        {isDraft ? "Interaction Draft (Live Populating)" : "Interaction Details"}
      </p>

      {isDraft && (
        <div className="compliance-banner" style={{ background: "rgba(0, 102, 102, 0.05)", color: "var(--teal)", border: "1px dashed rgba(0, 102, 102, 0.3)" }}>
          AI Assistant is extracting information from your conversation in real-time...
        </div>
      )}

      {selectedInteraction && lastSubmission?.id === selectedInteraction.id && (
        <div className={`compliance-banner ${complianceWarnings.length === 0 ? "ok" : ""}`}>
          {complianceWarnings.length === 0
            ? "Logged successfully. No compliance issues detected."
            : complianceWarnings.join(" ")}
        </div>
      )}

      {selectedInteraction && lastSubmission?.id !== selectedInteraction.id && (
        <div className="compliance-banner ok">
          Viewing logged interaction from {interaction.interaction_date ? new Date(interaction.interaction_date).toLocaleDateString("en-US", { day: 'numeric', month: 'short', year: 'numeric' }) : "—"}.
        </div>
      )}

      {/* Render generated Summary ONLY for saved interactions */}
      {!isDraft && (
        <div className="field">
          <label>Summary</label>
          <textarea
            value={interaction.summary || "—"}
            readOnly
            rows={3}
            style={{
              fontFamily: "inherit",
              fontSize: "13px",
              lineHeight: "1.5",
              resize: "none",
              backgroundColor: "#fcfdfd"
            }}
          />
        </div>
      )}

      <div className="row-2">
        <div className="field">
          <label>Interaction Type</label>
          <select value={interaction.interaction_type || ""} disabled>
            <option value="">—</option>
            <option value="Meeting">Meeting</option>
            <option value="Video Call">Video Call</option>
            <option value="Email">Email</option>
          </select>
        </div>
        <div className="field">
          <label>Sentiment</label>
          <select value={interaction.sentiment || ""} disabled>
            <option value="">—</option>
            <option value="positive">Positive</option>
            <option value="neutral">Neutral</option>
            <option value="negative">Negative</option>
          </select>
        </div>
      </div>

      <div className="row-2">
        <div className="field">
          <label>Topics Discussed</label>
          <input
            type="text"
            value={topicsStr || "—"}
            readOnly
            placeholder="—"
          />
        </div>
        <div className="field">
          <label>Medications Discussed</label>
          <input
            type="text"
            value={productsStr || "—"}
            readOnly
            placeholder="—"
          />
        </div>
      </div>

      <div className="field">
        <label>Detailed Notes</label>
        <textarea
          value={interaction.raw_notes || "—"}
          readOnly
          placeholder="—"
          rows={4}
        />
      </div>

      <div className="field">
        <label>Samples Provided</label>
        {samples.length > 0 ? (
          <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {samples.map((s, idx) => (
              <div key={idx} className="chip" style={{ backgroundColor: "#f0f0f0", color: "#333", border: "1px solid #ddd" }}>
                {s.product} ({s.qty} units)
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: "#888", fontSize: "13px", marginTop: "4px" }}>—</div>
        )}
      </div>
    </div>
  );
}
