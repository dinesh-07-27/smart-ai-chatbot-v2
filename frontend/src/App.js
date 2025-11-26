// frontend/src/App.js
import React, { useState, useRef, useEffect } from "react";

// 👇 NEW: base URL for backend – env var in deploy, localhost in dev
const API_BASE =
  process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([]); // {sender, text, meta}
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim()) return;
    const user = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { sender: "You", text: user }]);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Backend expects { text: ... }
        body: JSON.stringify({ text: user, session_id: sessionId }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`HTTP ${res.status}: ${txt}`);
      }

      const data = await res.json();

      // store session id if returned
      if (data.session_id && !sessionId) setSessionId(data.session_id);

      // Build bot message text with metadata
      let botText = data.answer ?? "(no answer)";
      let meta = {};
      if (data.timing_ms) meta.timing_ms = data.timing_ms;
      if (data.source) meta.source = data.source;
      if (data.cached !== undefined) meta.cached = data.cached;

      setMessages((prev) => [
        ...prev,
        { sender: "Bot", text: botText, meta },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "Bot",
          text: "Error: " + (err.message || err),
          meta: { error: true },
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div
      style={{
        maxWidth: 760,
        margin: "18px auto",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <h2 style={{ textAlign: "center" }}>Smart AI Customer Support Bot</h2>

      <div
        style={{
          border: "1px solid #ddd",
          padding: 12,
          height: 480,
          overflowY: "auto",
          background: "#fafafa",
          borderRadius: 6,
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              textAlign: m.sender === "You" ? "right" : "left",
              margin: "10px 0",
              padding: "6px 8px",
              borderRadius: 6,
              background: m.sender === "You" ? "#e6f7ff" : "#fff",
              display: "inline-block",
              maxWidth: "92%",
            }}
          >
            <div style={{ fontSize: 13, color: "#333" }}>
              <strong>{m.sender}:</strong>
            </div>
            <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
              {m.text}
            </div>

            {/* metadata display */}
            {m.meta && (
              <div style={{ marginTop: 6, fontSize: 12, color: "#666" }}>
                {m.meta.source && (
                  <span style={{ marginRight: 10 }}>📚 {m.meta.source}</span>
                )}
                {m.meta.cached !== undefined && m.meta.cached === true && (
                  <span style={{ marginRight: 10 }}>⚡ cached</span>
                )}
                {m.meta.timing_ms && (
                  <span>
                    ⏱ search: {m.meta.timing_ms.search ?? "-"} ms • gen:{" "}
                    {m.meta.timing_ms.generate ?? "-"} ms • total:{" "}
                    {m.meta.timing_ms.total ?? "-"} ms
                  </span>
                )}
                {m.meta.error && (
                  <span style={{ color: "red", marginLeft: 8 }}>Error</span>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
        <textarea
          style={{ flex: 1, padding: 10, minHeight: 46, resize: "vertical" }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask something... (Shift+Enter for newline, Enter to send)"
          onKeyDown={handleKey}
        />
        <div style={{ width: 120, display: "flex", flexDirection: "column" }}>
          <button
            onClick={send}
            disabled={loading}
            style={{
              padding: "10px 12px",
              background: loading ? "#ccc" : "#1976d2",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: loading ? "default" : "pointer",
            }}
          >
            {loading ? "Thinking..." : "Send"}
          </button>

          <div style={{ marginTop: 8, fontSize: 12, color: "#555" }}>
            <div>Session: {sessionId ? sessionId.slice(0, 8) : "—"}</div>
            <div style={{ marginTop: 6 }}>
              <small>Tip: keep queries short for faster replies</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
