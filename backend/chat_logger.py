# backend/chat_logger.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from .config import RAGConfig

# Role of this module:
# Structured logging of every conversation turn (one JSON object per line).
# The log is used by the RAGAS evaluation (question, answer, contexts) and to
# study how the system uses the multinational dataset (countries, types, sources).
# Used by the chat page and, later, by the API endpoint.

LOG_PATH = Path("logs/chat_log.jsonl")


def source_info(docs: List[Document]) -> List[dict]:
    """Key metadata of each retrieved document, in the order of the citations [S1], [S2], ..."""
    return [
        {
            "label": f"S{i}",
            "country": d.metadata.get("country"),
            "law": d.metadata.get("law"),
            "type": d.metadata.get("doc_type"),
            "article": d.metadata.get("article"),
            "case_id": d.metadata.get("case_id"),
            "articles_cited": d.metadata.get("articles_cited", []),
            "score": d.metadata.get("score"),
            "db": d.metadata.get("db_name"),
            "id": d.metadata.get("id"),
            "source": Path(str(d.metadata.get("source", ""))).name,
        }
        for i, d in enumerate(docs, start=1)
    ]


def log_turn(
    question: str,
    answer: str,
    docs: List[Document],
    config: RAGConfig,
    mode: str,
    latency_s: float,
    session_id: str = "",
    trace: Optional[str] = None,
    path: Path = LOG_PATH,
) -> dict:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "mode": mode,  # "single" (Task A) or "multi" (Task B)
        "question": question,
        "answer": answer,
        "sources": source_info(docs),
        # full text given to the LLM: RAGAS computes the context metrics on it
        "contexts": [d.page_content for d in docs],
        "trace": trace,
        "latency_s": round(latency_s, 2),
        "models": {
            "router": config.router_model_name,
            "generator": config.llm_model_name,
            "embedding": config.embedding_model_name,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
