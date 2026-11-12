# backend/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Role of this module:
# All other backend modules read from one shared configuration object.
# The Streamlit config page writes into RAGConfig, and the backend behaves accordingly


@dataclass
class RAGConfig:
    """
    Global configuration object stored in Streamlit session state.

    It controls:
      - LLM provider & model
      - Embedding provider & model
      - Data folders with JSON corpus
      - Vector store locations (single & multi-DB)
      - Retrieval behavior
      - Agentic behavior (standard RAG, ReAct, hybrid legal RAG)
      - Optional multi-agent supervisor
    """

    # ---------------- LLM ----------------
    # "groq"         -> ChatOpenAI on Groq's OpenAI-compatible API (needs GROQ_API_KEY)
    # "openai"       -> ChatOpenAI (needs OPENAI_API_KEY)
    # "huggingface"  -> HuggingFaceEndpoint / ChatHuggingFace (needs HF token for private models)
    llm_provider: str = "groq"
    llm_model_name: str = "openai/gpt-oss-120b"
    # smaller and faster model for the routing decision (Thought step)
    router_model_name: str = "openai/gpt-oss-20b"
    # RAGAS judge: a different model family from the generator (less self-preference bias)
    judge_model_name: str = "qwen/qwen3.8-27b"
    # low temperature: answers faithful to the sources and reproducible
    temperature: float = 0.1
    # gpt-oss models also spend tokens on reasoning before the answer
    max_tokens: int = 2000

    # ---------------- Embeddings ----------------
    # "huggingface" -> HuggingFaceEmbeddings (any HF model or local path)
    # "openai"      -> OpenAIEmbeddings
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "BAAI/bge-m3"

    # ---------------- Data (JSON corpus) ----------------
    # List of folders where JSON corpus lives
    json_folders: List[str] = field(default_factory=list)

    # ---------------- Vector stores (paths) ----------------
    # Root/base folder under which all FAISS vector DBs will be created.
    # Example: "vector_store" -> "vector_store/divorce_db", "vector_store/succession_db", etc.
    #
    # NOTE: `vector_store_base_dir` is kept for compatibility with the
    # Vector DB Builder / Config pages, which reference this attribute.
    vector_store_base_dir: str = "vector_store"

    # Alias used by some newer code; kept equal to base_dir by default.
    vector_store_root: str = "vector_store"

    # Default / active vector store directory (single-DB mode)
    # Example: "vector_store/default"
    vector_store_dir: str = "vector_store"

    # Optional: multiple DBs (multi-DB mode)
    # Default: the 9 DBs country x knowledge base built by scripts/build_indexes.py
    vector_store_dirs: List[str] = field(default_factory=lambda: [
        f"vector_store/{country}_{kb}"
        for country in ("italy", "estonia", "slovenia")
        for kb in ("inheritance_code", "divorce_code", "cases")
    ])

    # ---------------- Retrieval ----------------
    top_k: int = 5
    # documents retrieved from each DB before filtering and reranking
    k_fetch: int = 20
    # minimum cosine similarity to keep a document
    min_sim: float = 0.3
    # Reserved for future reranking strategies; currently not used in the pipeline.
    use_rerank: bool = False

    # ---------------- Agentic behavior ----------------
    # agentic_mode:
    #   - "standard_rag"  -> classic RAG (vector retrieval + answer)
    #   - "react"         -> ReAct-style agentic RAG (Thought / Action / Observation)
    #   - "hybrid_legal"  -> hybrid legal RAG (metadata extraction + metadata-aware vector search)
    agentic_mode: str = "react"

    # Multi-agent supervisor switch (used only in rag_pipeline for multi-DB agent routing)
    use_multiagent: bool = False
