from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# All other backend modules read from one shared configuration object.
# The Streamlit config page writes into RAGConfig, and the backend behaves accordingly

# [~] Groq models, 9 DBs, retrieval and extension options added.
@dataclass
class RAGConfig:

    # LLM
    # [+] "groq"     -> ChatOpenAI on Groq's OpenAI-compatible API (GROQ_API_KEY)
    # "openai"       -> ChatOpenAI (OPENAI_API_KEY)
    # "huggingface"  -> HuggingFaceEndpoint / ChatHuggingFace
    llm_provider: str = "groq"  # [~]
    llm_model_name: str = "openai/gpt-oss-120b"  # [~]
    # [+] smaller model for the routing decision (Thought step)
    router_model_name: str = "openai/gpt-oss-20b"
    # [+] RAGAS judge of another model family (less self-preference bias)
    judge_model_name: str = "qwen/qwen3.8-27b"
    # [+] low temperature: answers faithful to the sources
    temperature: float = 0.1
    # [+] gpt-oss also spends tokens on reasoning before the answer
    max_tokens: int = 2000
    # [+] reasoning effort of gpt-oss ("" = Groq default); "low" for the router keeps routing at 100%
    reasoning_effort: str = ""
    router_reasoning_effort: str = "low"

    # Embeddings
    embedding_provider: str = "huggingface"
    embedding_model_name: str = "BAAI/bge-m3"  # [~] multilingual

    # Data (JSON corpus)
    # List of folders where JSON corpus lives
    json_folders: List[str] = field(default_factory=lambda: ["data/Contest_Data"])  # [~]

    # Vector stores (paths)
    # Root/base folder under which all FAISS vector DBs will be created.
    vector_store_base_dir: str = "vector_store"

    # Alias used by some newer code; kept equal to base_dir by default.
    vector_store_root: str = "vector_store"

    # Default / active vector store directory (single-DB mode)
    vector_store_dir: str = "vector_store"

    # multiple DBs (multi-DB mode)
    # [~] default: the 9 DBs country x knowledge base
    vector_store_dirs: List[str] = field(default_factory=lambda: [
        f"vector_store/{country}_{kb}"
        for country in ("italy", "estonia", "slovenia")
        for kb in ("inheritance_code", "divorce_code", "cases")
    ])

    # Retrieval
    # [~] documents kept per country (Task A) / per agent (Task B)
    top_k: int = 3
    # [+] documents retrieved from each DB before filtering and reranking
    k_fetch: int = 20
    # [+] minimum cosine similarity to keep a document
    min_sim: float = 0.3
    # [+] guide 8: past cases DBs return only cases of the legal area of the question
    filter_cases_by_law: bool = False
    # [+] guide 8: search with the English translation and with the original question
    search_original_question: bool = True
    # [+] past cases: show their metadata (duration, cost, articles...) to reranker and LLM
    add_case_details: bool = True
    # [~] guide 8: second ranking of the retrieved documents with a cross-encoder
    use_rerank: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    # [+] only the first candidates by cosine are reranked (slow on CPU)
    rerank_candidates: int = 10

    # Multi-agent supervisor switch
    use_multiagent: bool = False
