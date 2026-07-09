import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { addUserMessage, sendMessage } from "../store/chatSlice";

export default function ChatInterface({ selectedHcpId }) {
  const dispatch = useDispatch();
  const { messages, status } = useSelector((s) => s.chat);
  const [draft, setDraft] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!draft.trim()) return;
    dispatch(addUserMessage(draft));
    dispatch(sendMessage({ message: draft, hcpId: selectedHcpId }));
    setDraft("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-panel">
      <p className="section-title">Agent Conversation</p>
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="empty-state">
            Try: "Met Dr. Sharma today, discussed the new cardio study, left 10 samples of Drug A,
            she wants the phase III data by Friday."
          </div>
        )}
        {messages.map((m) => (
          <div className={`msg ${m.role}`} key={m.id}>
            {m.content}
            {m.toolCalls && m.toolCalls.length > 0 && (
              <div className="tool-note">
                {m.toolCalls.map((tc) => tc.tool).join(" · ")}
              </div>
            )}
          </div>
        ))}
        {status === "loading" && <div className="msg assistant">Thinking…</div>}
      </div>
      <div className="chat-input-row">
        <textarea
          placeholder="Describe the interaction, or ask about a past one…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="btn-primary" onClick={handleSend} disabled={status === "loading"}>
          Send
        </button>
      </div>
    </div>
  );
}
