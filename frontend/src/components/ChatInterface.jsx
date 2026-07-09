import React, { useRef, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { addUserMessage, sendMessage, resetChat } from "../store/chatSlice";

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

  const handleButtonClick = (buttonText) => {
    dispatch(addUserMessage(buttonText));
    dispatch(sendMessage({ message: buttonText, hcpId: selectedHcpId }));
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const parseMessageButtons = (content) => {
    if (!content) return { cleanContent: "", buttons: [] };
    const regex = /\[([^\]]+)\]/g;
    const buttons = [];
    let match;
    while ((match = regex.exec(content)) !== null) {
      buttons.push(match[1].trim());
    }
    // Clean content by removing all [Option Text] patterns and clean up trailing spaces/newlines
    const cleanContent = content.replace(regex, "").trim();
    return { cleanContent, buttons };
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
        {messages.map((m, idx) => {
          const isAssistant = m.role === "assistant";
          const isLast = idx === messages.length - 1;
          
          let displayContent = m.content;
          let buttons = [];
          
          if (isAssistant) {
            const parsed = parseMessageButtons(m.content);
            displayContent = parsed.cleanContent;
            if (isLast) {
              buttons = parsed.buttons;
            }
          }

          return (
            <React.Fragment key={m.id}>
              <div className={`msg ${m.role}`}>
                {displayContent}
                {m.toolCalls && m.toolCalls.length > 0 && (
                  <div className="tool-note">
                    {m.toolCalls.map((tc) => tc.tool).join(" · ")}
                  </div>
                )}
              </div>
              {buttons.length > 0 && (
                <div className="chat-buttons">
                  {buttons.map((btn, bIdx) => (
                    <button
                      key={bIdx}
                      className="chat-btn"
                      onClick={() => handleButtonClick(btn)}
                    >
                      {btn}
                    </button>
                  ))}
                </div>
              )}
            </React.Fragment>
          );
        })}
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
