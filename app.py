import streamlit as st
from dotenv import load_dotenv
import os
import json

load_dotenv()

from odoo_connector import OdooConnector
from ai_engine import OdooAIEngine
from vector_store import VectorStore

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Odoo AI",
    page_icon="🤖",
    layout="wide",          # wide for side panels
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
html, body, .stApp { background: #0d0f14 !important; color: #e2e8f0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.5rem 6rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330 !important; }
[data-testid="stSidebar"] > div { padding: 1.2rem !important; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] input {
    background: #1a1e2a !important; border: 1px solid #2a3045 !important;
    border-radius: 8px !important; color: #e2e8f0 !important; font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #3b82f6 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-size: 0.82rem !important; font-weight: 600 !important;
    padding: 0.5rem 1rem !important; width: 100% !important;
}

/* Chat */
[data-testid="stChatMessageContent"] {
    background: #1a1e2a !important; border: 1px solid #2a3045 !important;
    border-radius: 14px !important; padding: 0.8rem 1.1rem !important;
    color: #e2e8f0 !important; font-size: 0.9rem !important; line-height: 1.65 !important;
}
[data-testid="stChatMessageContent"] table { width:100% !important; border-collapse:collapse !important; font-size:0.82rem !important; }
[data-testid="stChatMessageContent"] th { background:#1e293b !important; padding:0.5rem 0.8rem !important; text-align:left !important; font-size:0.72rem !important; text-transform:uppercase !important; color:#94a3b8 !important; border-bottom:1px solid #2a3045 !important; }
[data-testid="stChatMessageContent"] td { padding:0.5rem 0.8rem !important; border-bottom:1px solid #1e2330 !important; color:#cbd5e1 !important; }
[data-testid="stChatMessageContent"] strong { color:#f1f5f9 !important; }
[data-testid="stChatInput"] > div { background:#1a1e2a !important; border:1px solid #2a3045 !important; border-radius:12px !important; }
[data-testid="stChatInput"] textarea { background:transparent !important;  }
[data-testid="stChatInput"] textarea::placeholder { color:#475569 !important; }
[data-testid="stChatInput"] > div:focus-within { border-color:#3b82f6 !important; }
[data-testid="stChatInput"] button { background:#3b82f6 !important; border-radius:8px !important; }

/* Metric cards */
.stat-card { background:#1a1e2a; border:1px solid #2a3045; border-radius:10px; padding:10px; text-align:center; margin-bottom:6px; }
.stat-num  { font-size:1.4rem; font-weight:700; }
.stat-lbl  { font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:0.06em; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#2a3045; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages":   [],
    "engine":     None,
    "vs":         None,
    "connected":  False,
    "session_id": None,
    "log_ids":    [],   # parallel to messages — None for user msgs, str for assistant
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Init VectorStore once ──────────────────────────────────────────────────────
@st.cache_resource
def get_vector_store():
    return VectorStore(path="./chroma_db")

vs = get_vector_store()

# ── Auto-connect from .env ─────────────────────────────────────────────────────
if not st.session_state.connected:
    _u = os.getenv("ODOO_URL",""); _d = os.getenv("ODOO_DB","")
    _n = os.getenv("ODOO_USER",""); _p = os.getenv("ODOO_PASSWORD","")
    _g = os.getenv("GROQ_API_KEY","")
    if all([_u, _d, _n, _p, _g]):
        try:
            conn = OdooConnector(_u, _d, _n, _p)
            if conn.test_connection():
                engine = OdooAIEngine(conn, _g, vector_store=vs)
                sid    = vs.start_session(user=_n, llm_provider="groq")
                engine.set_session(sid)
                st.session_state.engine     = engine
                st.session_state.vs         = vs
                st.session_state.session_id = sid
                st.session_state.connected  = True
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — connection + stats
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
                padding-bottom:1rem;border-bottom:1px solid #1e2330">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px">🤖</div>
        <div>
            <div style="font-weight:700;font-size:0.9rem;color:#f1f5f9">Odoo AI</div>
            <div style="font-size:0.65rem;color:#475569">ChromaDB · Local Storage</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Status
    dot_color = "#10b981" if st.session_state.connected else "#ef4444"
    dot_label = "Connected · Logging to ChromaDB" if st.session_state.connected else "Not connected"
    st.markdown(f"""<div style="display:flex;align-items:center;gap:7px;
        background:rgba({'16,185,129' if st.session_state.connected else '239,68,68'},0.08);
        border:1px solid rgba({'16,185,129' if st.session_state.connected else '239,68,68'},0.2);
        border-radius:7px;padding:6px 10px;margin-bottom:1rem;
        font-size:0.75rem;color:{dot_color};font-weight:500">
        <div style="width:6px;height:6px;background:{dot_color};border-radius:50%"></div>
        {dot_label}</div>""", unsafe_allow_html=True)

    # Credentials
    with st.expander("⚙️ Connection", expanded=not st.session_state.connected):
        odoo_url  = st.text_input("Odoo URL",    value=os.getenv("ODOO_URL",""),      placeholder="https://your-odoo.com", label_visibility="collapsed")
        odoo_db   = st.text_input("Database",    value=os.getenv("ODOO_DB",""),       placeholder="database",              label_visibility="collapsed")
        odoo_user = st.text_input("Username",    value=os.getenv("ODOO_USER",""),     placeholder="admin",                 label_visibility="collapsed")
        odoo_pass = st.text_input("Password",    value=os.getenv("ODOO_PASSWORD",""), placeholder="password", type="password", label_visibility="collapsed")
        groq_key  = st.text_input("Groq API Key",value=os.getenv("GROQ_API_KEY",""), placeholder="gsk_...",  type="password", label_visibility="collapsed")

        if st.button("🔌 Connect"):
            if not all([odoo_url, odoo_db, odoo_user, odoo_pass, groq_key]):
                st.error("Fill in all fields.")
            else:
                with st.spinner("Connecting..."):
                    try:
                        conn = OdooConnector(odoo_url, odoo_db, odoo_user, odoo_pass)
                        if conn.test_connection():
                            engine = OdooAIEngine(conn, groq_key, vector_store=vs)
                            sid    = vs.start_session(user=odoo_user, llm_provider="groq")
                            engine.set_session(sid)
                            st.session_state.engine     = engine
                            st.session_state.vs         = vs
                            st.session_state.session_id = sid
                            st.session_state.connected  = True
                            st.session_state.messages   = []
                            st.session_state.log_ids    = []
                            st.rerun()
                        else:
                            st.error("Odoo connection failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Stats
    if st.session_state.connected:
        st.markdown("---")
        st.markdown('<p style="font-size:0.7rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">📊 All-Time Stats</p>', unsafe_allow_html=True)
        stats = vs.get_stats()
        if stats:
            c1, c2 = st.columns(2)
            with c1:
                acc = stats.get("accuracy", 0)
                acc_c = "#10b981" if acc >= 85 else "#f59e0b" if acc >= 60 else "#ef4444"
                st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#3b82f6">{stats.get("total",0)}</div><div class="stat-lbl">Total</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#10b981">{stats.get("correct",0)}</div><div class="stat-lbl">✅ Correct</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:{acc_c}">{acc}%</div><div class="stat-lbl">Accuracy</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#ef4444">{stats.get("wrong",0)}</div><div class="stat-lbl">❌ Wrong</div></div>', unsafe_allow_html=True)

            training_count = stats.get("training", 0)
            st.markdown(f'<div class="stat-card" style="margin-top:4px"><div class="stat-num" style="color:#a78bfa">{training_count}</div><div class="stat-lbl">📚 Training Examples</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Export training data
        if st.button("📥 Export Training JSONL"):
            jsonl = vs.export_training_jsonl()
            if jsonl:
                st.download_button(
                    label    = "⬇️ Download training.jsonl",
                    data     = jsonl,
                    file_name= "training_data.jsonl",
                    mime     = "application/jsonlines",
                )
            else:
                st.info("No training data yet. Rate answers as ❌ Wrong and add corrected queries.")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.log_ids  = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — chat (left) + semantic search panel (right)
# ══════════════════════════════════════════════════════════════════════════════
col_chat, col_panel = st.columns([3, 1.2])

with col_chat:
    st.markdown("""<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
        padding-bottom:1rem;border-bottom:1px solid #1e2330">
        <div style="width:34px;height:34px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px">🤖</div>
        <div>
            <div style="font-size:1.05rem;font-weight:700;color:#f1f5f9">Odoo AI Assistant</div>
            <div style="font-size:0.7rem;color:#475569">Ask anything · Stored in ChromaDB</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.connected:
        st.markdown("""<div style="text-align:center;padding:3rem 1rem;color:#475569">
            <div style="font-size:2rem;margin-bottom:1rem">🔌</div>
            <div style="font-size:0.95rem;font-weight:500;color:#94a3b8;margin-bottom:0.4rem">Not connected</div>
            <div style="font-size:0.8rem">Fill in your credentials in the sidebar.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Welcome
    if not st.session_state.messages:
        st.markdown("""<div style="background:#1a1e2a;border:1px solid #2a3045;border-radius:14px;
            padding:1.3rem;margin-bottom:1rem;text-align:center">
            <div style="font-size:1.3rem;margin-bottom:0.4rem">👋</div>
            <div style="font-size:0.9rem;font-weight:600;color:#f1f5f9;margin-bottom:0.3rem">Ready</div>
            <div style="font-size:0.78rem;color:#64748b;line-height:1.6">
                <span style="color:#60a5fa">"Show unpaid invoices"</span> ·
                <span style="color:#60a5fa">"Top customers this month"</span> ·
                <span style="color:#60a5fa">"Low stock products"</span><br/>
                <span style="font-size:0.7rem;color:#334155;margin-top:4px;display:block">
                    Rate each answer ✅ / ⚠️ / ❌ — wrong answers build your training dataset
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Chat history
    msg_list    = st.session_state.messages
    log_id_list = st.session_state.log_ids

    i = 0
    while i < len(msg_list):
        msg = msg_list[i]
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
            i += 1
            if i < len(msg_list) and msg_list[i]["role"] == "assistant":
                a_msg  = msg_list[i]
                log_id = log_id_list[i] if i < len(log_id_list) else None
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(a_msg["content"])

                    # Query viewer
                    if a_msg.get("query"):
                        with st.expander("👁 View Generated Query"):
                            st.code(
                                json.dumps(a_msg["query"], indent=2)
                                if isinstance(a_msg["query"], dict)
                                else str(a_msg["query"]),
                                language="json"
                            )

                    # Feedback buttons
                    if log_id:
                        fb = a_msg.get("feedback")
                        if not fb:
                            cols = st.columns([1, 1, 1, 5])
                            with cols[0]:
                                if st.button("✅", key=f"ok_{log_id}", help="Correct"):
                                    vs.submit_feedback(log_id, "correct")
                                    msg_list[i]["feedback"] = "correct"
                                    st.rerun()
                            with cols[1]:
                                if st.button("⚠️", key=f"pt_{log_id}", help="Partial"):
                                    vs.submit_feedback(log_id, "partial")
                                    msg_list[i]["feedback"] = "partial"
                                    st.rerun()
                            with cols[2]:
                                if st.button("❌", key=f"no_{log_id}", help="Wrong"):
                                    vs.submit_feedback(log_id, "wrong")
                                    msg_list[i]["feedback"] = "wrong"
                                    st.rerun()
                        else:
                            icons = {"correct":"✅ Correct","partial":"⚠️ Partial","wrong":"❌ Wrong"}
                            st.caption(f"Rated: {icons.get(fb, fb)}")

                        # Wrong → show corrected query input
                        if a_msg.get("feedback") == "wrong":
                            with st.expander("✏️ Add Corrected Query (for training)"):
                                corrected = st.text_area(
                                    "Paste the correct JSON query:",
                                    key=f"fix_{log_id}",
                                    placeholder='{"operation":"search_read","model":"sale.order",...}',
                                    height=100,
                                )
                                note = st.text_input("What was wrong?", key=f"note_{log_id}")
                                if st.button("💾 Save to Training Data", key=f"save_{log_id}"):
                                    vs.submit_feedback(log_id, "wrong", note=note, corrected=corrected)
                                    st.success("Saved to training_data collection!")
                i += 1
        else:
            i += 1

    # Input
    if prompt := st.chat_input("Ask anything about your Odoo data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.log_ids.append(None)

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(""):
                result = st.session_state.engine.ask(prompt)
            answer = result["answer"]
            log_id = result.get("log_id")
            query  = result.get("query")

            st.markdown(answer)

            if query:
                with st.expander("👁 View Generated Query"):
                    st.code(
                        json.dumps(query, indent=2)
                        if isinstance(query, dict) else str(query),
                        language="json"
                    )

            if log_id:
                cols = st.columns([1, 1, 1, 5])
                with cols[0]:
                    if st.button("✅", key=f"ok_{log_id}_n"):
                        vs.submit_feedback(log_id, "correct")
                with cols[1]:
                    if st.button("⚠️", key=f"pt_{log_id}_n"):
                        vs.submit_feedback(log_id, "partial")
                with cols[2]:
                    if st.button("❌", key=f"no_{log_id}_n"):
                        vs.submit_feedback(log_id, "wrong")

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "query": query, "feedback": None,
        })
        st.session_state.log_ids.append(log_id)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — semantic search on past questions
# ══════════════════════════════════════════════════════════════════════════════
with col_panel:
    st.markdown("""<div style="font-size:0.8rem;font-weight:600;color:#94a3b8;
        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;
        padding-bottom:8px;border-bottom:1px solid #1e2330">
        🔍 Similar Past Questions
    </div>""", unsafe_allow_html=True)

    if not st.session_state.connected:
        st.markdown('<p style="font-size:0.78rem;color:#334155">Connect first.</p>', unsafe_allow_html=True)
    elif vs.query_logs.count() == 0:
        st.markdown('<p style="font-size:0.78rem;color:#334155">No history yet. Ask some questions first.</p>', unsafe_allow_html=True)
    else:
        search_q = st.text_input(
            "Search",
            placeholder="e.g. unpaid invoices",
            label_visibility="collapsed",
            key="semantic_search",
        )
        show_all = st.toggle("Include unrated", value=False)

        if search_q:
            hits = vs.find_similar_questions(
                search_q, n=6, only_correct=not show_all
            )
            if hits:
                for h in hits:
                    sim_pct = int(h["similarity"] * 100)
                    fb_icon = {"correct":"✅","partial":"⚠️","wrong":"❌","pending":"🕐"}.get(h["feedback"],"")
                    sim_color = "#10b981" if sim_pct > 75 else "#f59e0b" if sim_pct > 50 else "#64748b"
                    st.markdown(f"""<div style="background:#1a1e2a;border:1px solid #2a3045;
                        border-radius:10px;padding:10px 12px;margin-bottom:8px">
                        <div style="font-size:0.78rem;color:#e2e8f0;font-weight:500;margin-bottom:4px">
                            {fb_icon} {h['question'][:80]}{'…' if len(h['question'])>80 else ''}
                        </div>
                        <div style="font-size:0.7rem;color:#475569">
                            {h.get('model','') or '—'}  ·
                            <span style="color:{sim_color};font-weight:600">{sim_pct}% match</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-size:0.78rem;color:#334155">No similar questions found.</p>', unsafe_allow_html=True)
        else:
            # Show recent logs when no search
            st.markdown('<p style="font-size:0.72rem;color:#475569;margin-bottom:8px">Recent queries:</p>', unsafe_allow_html=True)
            recent = vs.get_recent_logs(n=8)
            for log in recent:
                fb_icon = {"correct":"✅","partial":"⚠️","wrong":"❌","pending":"🕐"}.get(log.get("feedback"),"🕐")
                q = log.get("question","")
                st.markdown(f"""<div style="background:#1a1e2a;border:1px solid #2a3045;
                    border-radius:8px;padding:8px 10px;margin-bottom:6px">
                    <div style="font-size:0.75rem;color:#cbd5e1">{fb_icon} {q[:70]}{'…' if len(q)>70 else ''}</div>
                    <div style="font-size:0.68rem;color:#334155;margin-top:2px">
                        {log.get('odoo_model','—')} · {log.get('execution_ms',0)}ms
                    </div>
                </div>""", unsafe_allow_html=True)
