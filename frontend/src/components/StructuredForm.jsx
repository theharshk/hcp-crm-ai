import React from "react";
import { useSelector } from "react-redux";

export default function StructuredForm() {
  const { lastSubmission, selectedInteraction, draftInteraction, complianceWarnings, history } = useSelector((s) => s.interactions);

  // If selectedInteraction or draftInteraction is active, render details. Otherwise, render empty/combined state.
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

  const topicsStr = isSelected
    ? (Array.isArray(interaction.topics_discussed)
        ? interaction.topics_discussed.join(", ")
        : interaction.topics_discussed || "")
    : "";

  const productsStr = isSelected
    ? (Array.isArray(interaction.products_discussed)
        ? interaction.products_discussed.join(", ")
        : interaction.products_discussed || "")
    : "";

  const samples = isSelected ? (interaction.samples_provided || []) : [];

  // Generate combined summary of all interactions if no specific one is chosen
  const getCombinedSummary = () => {
    if (!history || history.length === 0) return "";
    return history
      .map((h) => {
        const dateStr = h.interaction_date
          ? new Date(h.interaction_date).toLocaleDateString()
          : "—";
        return `• [${dateStr} - ${h.interaction_type}] ${h.summary || "(no summary)"}`;
      })
      .join("\n");
  };

  const summaryValue = selectedInteraction
    ? (interaction.summary || "")
    : (draftInteraction?.summary || getCombinedSummary());

  return (
    <div>
      <p className="section-title">
        {isDraft
          ? "Interaction Details (Drafting Live...)"
          : selectedInteraction
          ? "Interaction Details (AI Populated)"
          : "All Interactions Summary (HCP Overview)"}
      </p>

      {!isSelected && (
        <div className="compliance-banner" style={{ background: "var(--teal-dim)", color: "var(--teal)" }}>
          Showing collective history. Click any "Recent Interaction" above to view its detailed logs and notes.
        </div>
      )}

      {isDraft && (
        <div className="compliance-banner" style={{ background: "rgba(0, 102, 102, 0.05)", color: "var(--teal)", border: "1px dashed rgba(0, 102, 102, 0.3)" }}>
          AI Assistant is drafting interaction details live from the chat...
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
          Viewing logged interaction from {interaction.interaction_date ? new Date(interaction.interaction_date).toLocaleDateString() : "—"}.
        </div>
      )}

      <div className="row-2">
        <div className="field">
          <label>Interaction Type</label>
          <select value={interaction.interaction_type} disabled>
            {!isSelected && <option value="">—</option>}
            <option value="Meeting">Meeting</option>
            <option value="Video Call">Video Call</option>
            <option value="Email">Email</option>
          </select>
        </div>
        <div className="field">
          <label>Sentiment</label>
          <select value={interaction.sentiment} disabled>
            {!isSelected && <option value="">—</option>}
            <option value="positive">Positive</option>
            <option value="neutral">Neutral</option>
            <option value="negative">Negative</option>
          </select>
        </div>
      </div>

      <div className="field">
        <label>Summary</label>
        {isSelected ? (
          <input
            type="text"
            value={summaryValue}
            readOnly
            placeholder="AI-generated summary will appear here..."
          />
        ) : (
          <textarea
            value={summaryValue}
            readOnly
            rows={history.length > 2 ? Math.min(history.length, 6) : 3}
            placeholder="No interactions logged yet for this doctor."
            style={{
              fontFamily: "inherit",
              fontSize: "13px",
              lineHeight: "1.5",
              resize: "none",
              backgroundColor: "#fcfdfd"
            }}
          />
        )}
      </div>

      <div className="row-2">
        <div className="field">
          <label>Topics Discussed</label>
          <input
            type="text"
            value={topicsStr}
            readOnly
            placeholder={isSelected ? "AI-extracted topics..." : "—"}
          />
        </div>
        <div className="field">
          <label>Products Discussed</label>
          <input
            type="text"
            value={productsStr}
            readOnly
            placeholder={isSelected ? "Products discussed or 'No Product Discussed'" : "—"}
          />
        </div>
      </div>

      <div className="field">
        <label>Detailed Notes</label>
        <textarea
          value={interaction.raw_notes || ""}
          readOnly
          placeholder={isSelected ? "The rep's raw conversation notes will be displayed here..." : "—"}
        />
      </div>

      {samples.length > 0 && (
        <div className="field">
          <label>Samples Provided</label>
          <div style={{ marginTop: 6 }}>
            {samples.map((s, idx) => (
              <div key={idx} className="chip">
                {s.product} ({s.qty} units)
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
