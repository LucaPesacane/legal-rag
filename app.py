import os

import streamlit as st
from dotenv import load_dotenv

from backend.config import RAGConfig
from backend.embeddings import get_embedding_model
from backend.vector_store import load_vector_store

# Load env vars from .env (OPENAI_API_KEY, etc.)
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Playground (LangChain)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Agentic RAG Playground (LangChain)")

for key_name, provider in [("GROQ_API_KEY", "Groq"), ("OPENAI_API_KEY", "OpenAI")]:
    if os.getenv(key_name):
        st.caption(f"🔑 {key_name} loaded from .env")
    else:
        st.caption(f"⚠️ {key_name} not set (needed only to use {provider}).")

# Warm up: load the embedding model and the FAISS DBs once at startup,
# so that the first question does not wait for them (they stay cached in memory)
config = RAGConfig()
with st.spinner("Loading embedding model and vector databases..."):
    embedding_model = get_embedding_model(config)
    missing = [p for p in config.vector_store_dirs if not os.path.isdir(p)]
    for path in config.vector_store_dirs:
        if path not in missing:
            load_vector_store(path, embedding_model)
if missing:
    st.warning(f"Missing vector DBs: {', '.join(missing)}. Build them with scripts/build_indexes.py.")
else:
    st.caption(f"✅ {len(config.vector_store_dirs)} vector DBs ready ({config.embedding_model_name})")

st.markdown(
    """
Welcome!  

Use the sidebar to navigate:

1. **Configuration** – define LLM, embedding model, JSON data folders, and agentic RAG options.  
2. **Vector DB Builder** – load your JSON data and create the FAISS vector database (via LangChain).  
3. **Chatbot Q&A** – talk with your agentic RAG chatbot (Task A single agent or Task B multi-agent).
4. **RAG Evaluation** – RAGAS metrics on the conversation logs and on the test set.
"""
)

st.info("➡️ Select a page from the sidebar to get started.")
