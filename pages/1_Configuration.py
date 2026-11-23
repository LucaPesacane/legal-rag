# ======================================================================
# pages/1_Configuration.py
# Global configuration for the Agentic RAG app (OpenAI + HuggingFace)
# ======================================================================
import streamlit as st

from backend.config import RAGConfig


def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


st.title("⚙️ Agent & RAG Configuration")

config = get_config()

# ---------------- LLM SETTINGS ----------------
st.subheader("LLM Settings")

col1, col2 = st.columns(2)

with col1:
    # [~] groq provider added
    config.llm_provider = st.selectbox(
        "LLM Provider",
        options=["groq", "openai", "huggingface"],
        index=["groq", "openai", "huggingface"].index(config.llm_provider)
        if config.llm_provider in ["groq", "openai", "huggingface"]
        else 0,
        help=(
            "- **groq**: Groq's OpenAI-compatible API (needs `GROQ_API_KEY` in `.env`).\n"
            "- **openai**: uses ChatOpenAI (needs `OPENAI_API_KEY` in `.env`).\n"
            "- **huggingface**: any HF generative model (hub id or local path)."
        ),
    )

with col2:
    config.llm_model_name = st.text_input(
        "LLM Model Name or Path",
        value=config.llm_model_name,
        help=(
            "For Groq: e.g. `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`.\n"
            "For OpenAI: e.g. `gpt-4o-mini`.\n"
            "For Hugging Face: model id (e.g. `meta-llama/Llama-3.1-8B-Instruct`) "
            "or a local folder path."
        ),
    )

# [+] Router model and temperature of the answers
col_l1, col_l2 = st.columns(2)

with col_l1:
    config.router_model_name = st.text_input(
        "Router / supervisor routing model",
        value=config.router_model_name,
        help="Smaller model used for the routing decision (Thought step of Task A, supervisor routing of Task B).",
    )

with col_l2:
    config.temperature = st.slider(
        "Temperature of the answers",
        min_value=0.0,
        max_value=1.0,
        value=float(config.temperature),
        step=0.05,
        help="Low values give answers faithful to the sources and reproducible.",
    )

# ---------------- EMBEDDING SETTINGS ----------------
st.subheader("Embedding Settings")

col3, col4 = st.columns(2)

with col3:
    config.embedding_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "openai"],
        index=["huggingface", "openai"].index(config.embedding_provider)
        if config.embedding_provider in ["huggingface", "openai"]
        else 0,
        help=(
            "huggingface → `HuggingFaceEmbeddings` (any HF model or local path).\n"
            "openai → `OpenAIEmbeddings` (e.g. `text-embedding-3-small`)."
        ),
    )

with col4:
    config.embedding_model_name = st.text_input(
        "Embedding Model Name or Path",
        value=config.embedding_model_name,
        help=(
            "For Hugging Face: e.g. `all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`, "
            "or a local path.\n"
            "For OpenAI: e.g. `text-embedding-3-small`."
        ),
    )

# ---------------- DATA FOLDERS ----------------
st.subheader("Data Folders (JSON with content + metadata)")

new_folder = st.text_input(
    "Add JSON folder path",
    value="",
    placeholder="data/folder1",
)

cols_add = st.columns([1, 1, 2])
with cols_add[0]:
    if st.button("➕ Add folder") and new_folder:
        if new_folder not in config.json_folders:
            config.json_folders.append(new_folder)

with cols_add[1]:
    if st.button("🧹 Clear folders"):
        config.json_folders = []

if config.json_folders:
    st.write("Current JSON folders:")
    for f in config.json_folders:
        st.code(f)
else:
    st.info("No JSON folders added yet. These are used by the Vector DB Builder page.")

# ---------------- VECTOR STORE SETTINGS ----------------
st.subheader("Vector Store Settings (FAISS)")

config.vector_store_base_dir = st.text_input(
    "Base directory for all vector stores",
    value=config.vector_store_base_dir,
    help=(
        "Root folder under which all FAISS vector DBs will be created.\n"
        "Example: `vector_store` → `vector_store/legal`, `vector_store/tax`, etc."
    ),
)

config.vector_store_dir = st.text_input(
    "Default / Legacy Vector Store Directory",
    value=config.vector_store_dir,
    help=(
        "Used as the default single vector DB path and for backward compatibility.\n"
        "For multi-DB setups, the Vector DB Builder will create subfolders under "
        "the base directory and update `vector_store_dirs`."
    ),
)

if config.vector_store_dirs:
    st.write("Registered vector DB directories (multi-DB mode):")
    for p in config.vector_store_dirs:
        st.code(p)
else:
    st.caption(
        "No explicit vector DB directories registered yet. "
    )

# ---------------- RETRIEVAL SETTINGS ----------------
st.subheader("Retrieval & Post-processing")

col_r1, col_r2 = st.columns(2)

with col_r1:
    config.top_k = st.slider(
        "Top-K documents after filtering",
        min_value=1,
        max_value=20,
        value=config.top_k,
        help="Number of documents kept after similarity-based filtering.",
    )

    # [+] candidates per DB and similarity threshold
    config.k_fetch = st.slider(
        "Candidates retrieved from each DB (before filtering)",
        min_value=5,
        max_value=50,
        value=config.k_fetch,
        help="More candidates than needed are retrieved, then filtered and reranked by cosine similarity.",
    )
    config.min_sim = st.slider(
        "Minimum cosine similarity",
        min_value=0.0,
        max_value=0.9,
        value=float(config.min_sim),
        step=0.05,
        help="Documents below this similarity are discarded.",
    )

with col_r2:
    # [+] search with the original question, metadata of past cases
    config.search_original_question = st.checkbox(
        "Search also with the original question (guide 8)",
        value=config.search_original_question,
        help="The DBs are searched with the English translation and with the question as written.",
    )
    config.add_case_details = st.checkbox(
        "Show the metadata of past cases to reranker and LLM",
        value=config.add_case_details,
        help="Duration, cost, proceeding, articles cited... (from the JSON metadata, files not changed).",
    )
    # [~] the reranker is now implemented (cross-encoder)
    config.use_rerank = st.checkbox(
        "Use a cross-encoder reranker after retrieval (guide 8)",
        value=config.use_rerank,
        help=(
            "After cosine filtering, a cross-encoder reads question and document together "
            "and re-ranks the candidates (more precise, slower). Model below."
        ),
    )
    config.reranker_model_name = st.text_input("Reranker model", value=config.reranker_model_name)

# ---------------- AGENTIC MODE (within each RAG agent) ----------------
st.subheader("Agentic RAG Reasoning Mode (per agent)")

config.agentic_mode = st.radio(
    "Agentic mode",
    options=["standard_rag", "react", "hybrid_legal"],
    index=["standard_rag", "react", "hybrid_legal"].index(config.agentic_mode)
    if config.agentic_mode in ["standard_rag", "react", "hybrid_legal"]
    else 0,
    help=(
        "- standard_rag / react: Task A single agent with ReAct-style logic "
        "(Thought / Action / Observation / Answer) and Task B multi-agent.\n"
        "- hybrid_legal: legal metadata extraction (Succession/Divorce + cost, "
        "duration, civil codes, etc.) + metadata-aware vector search."
    ),
)


# ---------------- MULTI-AGENT SUPERVISOR ----------------
# [+] Extension: filter past cases by legal area
st.subheader("Extensions (exam guide 8)")

config.filter_cases_by_law = st.checkbox(
    "Filter past cases by legal area (inheritance / divorce)",
    value=config.filter_cases_by_law,
    help=(
        "The router detects the legal area of the question; in the past cases DBs only the cases "
        "of that area are searched (metadata filter first, then vector similarity)."
    ),
)

st.subheader("Multi-agent Supervisor (tool-calling)")

config.use_multiagent = st.checkbox(
    "Enable multi-agent supervisor over specialized RAG agents",
    value=config.use_multiagent,
    help=(
        "If enabled, the chatbot will use a supervisor LLM that routes questions "
        "to specialized RAG agents (one per vector database) and synthesizes their "
        "answers. If disabled, a single agent handles everything."
    ),
)

# ---------------- SAVE ----------------
if st.button("💾 Save configuration"):
    st.session_state.config = config
    st.success("Configuration saved in session.")

st.caption("Next: go to **Vector DB Builder** to create the FAISS vector store(s).")
