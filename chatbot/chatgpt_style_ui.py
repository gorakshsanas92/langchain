import streamlit as st


st.set_page_config(
    page_title="ChatGPT Concept UI",
    page_icon="💬",
    layout="wide",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg-body: #05060c;
    --bg-card: #111322;
    --border: #202339;
    --text-primary: #f5f7ff;
    --text-muted: #8f95b2;
    --accent: #8652ff;
    --accent-2: #9f58ff;
    --accent-3: #4f32ff;
}

#MainMenu, header, footer {
    visibility: hidden;
}

.stApp {
    background: radial-gradient(circle at top, #15172b 0%, var(--bg-body) 45%, #05060c 100%);
    font-family: 'Inter', system-ui, sans-serif;
}

.block-container {
    padding-top: 24px;
    padding-bottom: 24px;
    padding-left: min(5vw, 48px);
    padding-right: min(5vw, 48px);
}

.app-shell {
    display: flex;
    gap: 32px;
    max-width: 1200px;
    margin: 0 auto 48px;
    color: var(--text-primary);
}

.nav-panel {
    width: 280px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    box-shadow: 0 8px 40px rgba(5, 6, 12, 0.5);
}

.nav-logo {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.nav-logo span {
    font-weight: 600;
    color: var(--text-primary);
}

.search-box {
    position: relative;
}

.search-box input {
    width: 100%;
    padding: 12px 42px 12px 16px;
    border-radius: 18px;
    border: 1px solid #22263c;
    background: #0c0f1d;
    color: var(--text-primary);
    font-size: 14px;
}

.search-box kbd {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    background: #16192a;
    border-radius: 8px;
    padding: 2px 6px;
    border: 1px solid #262a44;
    font-size: 11px;
    color: var(--text-muted);
}

.nav-button {
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(120deg, rgba(134, 82, 255, 0.15), rgba(79, 50, 255, 0.25));
    border: 1px solid rgba(134, 82, 255, 0.5);
    border-radius: 18px;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 14px;
}

.nav-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.nav-section label {
    text-transform: uppercase;
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

.nav-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0c0f1d;
    border-radius: 16px;
    padding: 12px 14px;
    border: 1px solid transparent;
    transition: border 0.2s ease;
}

.nav-item:hover {
    border-color: rgba(134, 82, 255, 0.4);
}

.nav-item span {
    font-size: 13px;
    color: var(--text-primary);
}

.nav-item small {
    font-size: 12px;
    color: var(--text-muted);
}

.profile-card {
    margin-top: auto;
    background: #0c0f1d;
    border-radius: 20px;
    padding: 16px;
    border: 1px solid #20243a;
    display: flex;
    gap: 12px;
    align-items: center;
}

.avatar {
    width: 46px;
    height: 46px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ff8b5f, #f04c8a);
    display: grid;
    place-items: center;
    font-weight: 700;
    font-size: 18px;
}

.profile-card span {
    font-size: 13px;
    color: var(--text-muted);
}

.chat-panel {
    flex: 1;
    background: linear-gradient(145deg, rgba(134, 82, 255, 0.08), rgba(20, 24, 41, 0.9));
    border: 1px solid rgba(134, 82, 255, 0.3);
    border-radius: 36px;
    padding: 36px;
    display: flex;
    flex-direction: column;
    gap: 32px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
}

.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-header h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
}

.chip {
    border-radius: 999px;
    padding: 6px 14px;
    background: rgba(134, 82, 255, 0.2);
    border: 1px solid rgba(134, 82, 255, 0.5);
    font-size: 13px;
    color: var(--text-primary);
}

.conversation {
    background: rgba(5, 6, 12, 0.55);
    border-radius: 28px;
    padding: 28px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.bubble {
    border-radius: 18px;
    padding: 18px 20px;
    position: relative;
}

.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #c963ff, #8c4bff);
    color: white;
    max-width: 420px;
}

.bubble-bot {
    align-self: flex-start;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    max-width: 520px;
}

.bot-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 10px;
}

.bot-meta span {
    background: rgba(134, 82, 255, 0.15);
    padding: 4px 8px;
    border-radius: 12px;
    border: 1px solid rgba(134, 82, 255, 0.35);
    color: var(--text-primary);
}

.bubble-actions {
    display: flex;
    gap: 12px;
    margin-top: 12px;
}

.bubble-actions button {
    background: rgba(134, 82, 255, 0.08);
    border: 1px solid rgba(134, 82, 255, 0.3);
    border-radius: 26px;
    color: var(--text-primary);
    padding: 6px 14px;
    font-size: 13px;
}

.input-card {
    background: rgba(5, 6, 12, 0.8);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.input-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.input-row input {
    flex: 1;
    background: #0f1324;
    border-radius: 18px;
    border: 1px solid #242943;
    padding: 16px;
    color: var(--text-primary);
    font-size: 15px;
}

.input-row button {
    width: 46px;
    height: 46px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(120deg, var(--accent), var(--accent-3));
    color: white;
    font-size: 20px;
}

.helper-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--text-muted);
}

.helper-row span {
    display: flex;
    align-items: center;
    gap: 6px;
}

.badges {
    display: flex;
    gap: 10px;
}

.badges .badge {
    padding: 6px 12px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(134,82,255,0.08);
    font-size: 13px;
}

@media (max-width: 980px) {
    .app-shell {
        flex-direction: column;
    }
    .nav-panel {
        width: 100%;
    }
}
</style>
"""

APP_HTML = """
<div class="app-shell">
  <aside class="nav-panel">
    <div class="nav-logo">
      <span>ChatGPT</span>
      <small>v3.2 beta</small>
    </div>
    <div class="search-box">
      <input placeholder="Search" />
      <kbd>⌘K</kbd>
    </div>
    <div class="nav-button">
      <span style="font-size:18px">＋</span>
      <div>
        <div>New Chat</div>
        <small style="color:var(--text-muted); font-weight:400;">Start a fresh prompt</small>
      </div>
    </div>
    <div class="nav-section">
      <label>Recent</label>
      <div class="nav-item">
        <span>Who is Andres The Designer?</span>
        <small>2m</small>
      </div>
      <div class="nav-item">
        <span>AB testing ideas for a clothing e-commerce store</span>
        <small>12m</small>
      </div>
      <div class="nav-item">
        <span>How to redesign the ChatGPT interface</span>
        <small>1h</small>
      </div>
    </div>
    <div class="nav-section">
      <label>Quick actions</label>
      <div class="nav-item">
        <span>Prompt templates</span>
        <small>→</small>
      </div>
      <div class="nav-item">
        <span>Discover</span>
        <small>→</small>
      </div>
      <div class="nav-item">
        <span>Settings</span>
        <small>→</small>
      </div>
    </div>
    <div class="profile-card">
      <div class="avatar">AG</div>
      <div>
        <strong>Andres Gonzalez</strong>
        <br />
        <span>Product Designer</span>
      </div>
    </div>
  </aside>
  <section class="chat-panel">
    <div class="chat-header">
      <div>
        <h2>Who is Andres The Designer?</h2>
        <p style="margin:4px 0 0; color:var(--text-muted); font-size:14px;">Conversation · today · 8:45 PM</p>
      </div>
      <div class="chip">Focus mode</div>
    </div>
    <div class="conversation">
      <div class="bubble bubble-user">
        Who is Andres The Designer?
      </div>
      <div class="bubble bubble-bot">
        <div class="bot-meta">
          <span>used keymate.ai</span>
          <small>Smart profile lookup</small>
        </div>
        <p style="margin:0 0 10px; line-height:1.6;">
          Andres The Designer is a rising digital product designer, YouTube educator, and entrepreneur.
          He sometimes imagines himself as Batman. Instead of fighting crime, he designs squares.
          He is vengeance.
        </p>
        <div class="bubble-actions">
          <button>👍</button>
          <button>👎</button>
          <button>⋯</button>
        </div>
      </div>
      <div class="badges">
        <div class="badge">Regenerate</div>
        <div class="badge">Share</div>
        <div class="badge">4.0</div>
      </div>
    </div>
    <div class="input-card">
      <div class="input-row">
        <input value="How do I hire this guy?!?" />
        <button>➤</button>
      </div>
      <div class="helper-row">
        <span>ChatGPT may produce inaccurate information about people, places, or facts.</span>
        <span>⌘⏎ to send</span>
      </div>
    </div>
  </section>
</div>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(APP_HTML, unsafe_allow_html=True)
