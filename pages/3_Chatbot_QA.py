# pages/3_Chatbot_QA.py

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from backend.chat_logger import LOG_PATH, log_turn, source_info
from backend.config import RAGConfig
from backend.hybrid_rag import hybrid_answer_question
from backend.rag_pipeline import answer_question as rag_answer_question

CHAT_DB_PATH = Path("chat_sessions.json")


# ---------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------
def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


# ---------------------------------------------------------------------
# Chat DB helpers (whole conversations saved with "New chat")
# ---------------------------------------------------------------------
def load_chat_db() -> List[Dict[str, Any]]:
    if CHAT_DB_PATH.exists():
        try:
            with open(CHAT_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def append_chat_to_db(history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    db = load_chat_db()
    next_id = max([c.get("id", 0) for c in db], default=0) + 1
    title = next((m["content"] for m in history if m["role"] == "user"), f"Chat {next_id}")
    db.append({"id": next_id, "title": title[:80], "history": history})
    with open(CHAT_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Sources panel: country, code/case, articles, excerpt (transparency)
# ---------------------------------------------------------------------
def show_sources(sources: List[dict], contexts: List[str]) -> None:
    rows = []
    for s, text in zip(sources, contexts):
        body = text.split("\n", 1)[-1]  # text without the metadata header
        rows.append({
            "Source": s["label"],
            "Country": s["country"],
            "Type": "Civil code" if s["type"] == "code" else "Past case",
            "Law": s["law"],
            "Article / Case": f"Art. {s['article']}" if s["type"] == "code" else s["case_id"],
            "Articles cited": ", ".join(s["articles_cited"]) if s["type"] == "case" else "",
            "Similarity": s["score"],
            "Excerpt": body[:300] + ("..." if len(body) > 300 else ""),
        })
    with st.expander(f"📎 Sources ({len(rows)})", expanded=True):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("💬 Legal RAG Chatbot – Italy, Estonia, Slovenia")

config = get_config()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]

col_top1, col_top2, col_top3 = st.columns([1, 1, 2])
with col_top1:
    if st.button("🆕 New chat (save current)"):
        append_chat_to_db(st.session_state.chat_history)
        st.session_state.chat_history = []
        st.session_state.session_id = uuid.uuid4().hex[:8]
with col_top2:
    if st.button("🗑 Clear current chat"):
        st.session_state.chat_history = []
with col_top3:
    st.caption(f"Every turn is logged in `{LOG_PATH}` · saved chats: {len(load_chat_db())}")

# Architecture used for the answers
mode = st.radio(
    "Architecture",
    ["Task A – Single agent (ReAct)", "Task B – Multi-agent (supervisor)"],
    index=1 if config.use_multiagent else 0,
    horizontal=True,
)
config.use_multiagent = mode.startswith("Task B")
show_trace = st.checkbox("Show reasoning trace", value=True)

# ---------------------------------------------------------------------
# Render existing history (sources and trace are kept for every answer)
# ---------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"], msg["contexts"])
        if show_trace and msg.get("trace"):
            with st.expander("🔍 Reasoning trace"):
                st.markdown(msg["trace"])

# ---------------------------------------------------------------------
# Chat input + answer
# ---------------------------------------------------------------------
user_input = st.chat_input("Ask about inheritance or divorce law in Italy, Estonia or Slovenia...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            if config.agentic_mode == "hybrid_legal":
                answer, docs, trace, _ = hybrid_answer_question(user_input, config, show_reasoning=True)
            else:
                answer, docs, trace = rag_answer_question(user_input, config, show_reasoning=True)
            latency = time.time() - start

        st.markdown(answer)
        sources = source_info(docs)
        contexts = [d.page_content for d in docs]
        if sources:
            show_sources(sources, contexts)
        if show_trace and trace:
            with st.expander("🔍 Reasoning trace"):
                st.markdown(trace)
        st.caption(f"{'Task B' if config.use_multiagent else 'Task A'} · {latency:.1f}s")

    # structured log of the turn (used by the RAGAS evaluation)
    log_turn(
        user_input, answer, docs, config,
        mode="multi" if config.use_multiagent else "single",
        latency_s=latency, session_id=st.session_state.session_id, trace=trace,
    )
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer,
        "sources": sources, "contexts": contexts, "trace": trace,
    })
