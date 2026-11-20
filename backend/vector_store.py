from __future__ import annotations
from pathlib import Path
from typing import List
import os
from langchain_core.documents import Document  
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import shutil

from .document_loader import case_details  # [+]

# This is the vector database layer: which builds and loads FAISS vector stores using LangChain-Documents.

# During offline step, called by the Vector DB Builder page to create FAISS DBs.
# During online step, called by RAG pipelines to load the correct DB and create retrievers.


# backend/vector_store.py
# Simple in-memory cache: {path -> FAISS vector store}
_VECTOR_STORE_CACHE: dict[str, FAISS] = {}

# [~] FAISS DB with inner product on normalized vectors (= cosine similarity).
def build_vector_store(
    docs: List[Document],
    embedding_model,
    target_dir: str,
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    vs = FAISS.from_documents(
        docs, embedding_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    vs.save_local(target_dir)

    _VECTOR_STORE_CACHE[target_dir] = vs


# [~] Load a FAISS DB (cached), setting again the distance strategy.
def load_vector_store(
    path: str,
    embedding_model,
) -> FAISS:
    cached = _VECTOR_STORE_CACHE.get(path)
    if cached is not None:
        return cached

    # the distance strategy is not saved on disk
    vs = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    _VECTOR_STORE_CACHE[path] = vs
    return vs


# [+] One FAISS DB per country x knowledge base (9 DBs), e.g. vector_store/italy_cases.
def build_all_vector_stores(
    docs: List[Document],
    embedding_model,
    root: str,
) -> List[str]:
    groups: dict[str, List[Document]] = {}
    for d in docs:
        name = f"{d.metadata['country'].lower()}_{d.metadata['kb']}"
        groups.setdefault(name, []).append(d)

    paths = []
    for name, group in sorted(groups.items()):
        path = os.path.join(root, name)
        build_vector_store(group, embedding_model, path)
        paths.append(path)
    return paths


# [+] Search several DBs (one or more queries), keep the best chunk per document,
# optionally add case metadata and rerank; returns the top_k documents.
def search_vector_stores(
    question: str,
    db_paths: List[str],
    embedding_model,
    k_fetch: int = 20,
    top_k: int = 6,
    min_sim: float = 0.3,
    law: str | None = None,
    reranker=None,
    rerank_candidates: int = 10,
    extra_queries: List[str] = (),
    add_case_details: bool = False,
) -> List[Document]:
    # embed each query only once
    q_vecs = [embedding_model.embed_query(q) for q in [question, *extra_queries]]

    results: List[Document] = []
    for path in db_paths:
        vs = load_vector_store(path, embedding_model)
        # metadata filter first, then vector similarity (only the cases DBs mix the two areas)
        filters = {"filter": {"law": law}, "fetch_k": 200} if law and path.endswith("_cases") else {}
        for q_vec in q_vecs:
            for doc, score in vs.similarity_search_with_score_by_vector(q_vec, k=k_fetch, **filters):
                if score >= min_sim:
                    meta = {**doc.metadata, "score": round(float(score), 4), "db_name": os.path.basename(path)}
                    results.append(Document(page_content=doc.page_content, metadata=meta))

    results.sort(key=lambda d: d.metadata["score"], reverse=True)

    # keep only the best chunk of each document (and the best query for it)
    kept, seen = [], set()
    for d in results:
        if d.metadata.get("id") not in seen:
            seen.add(d.metadata.get("id"))
            kept.append(d)

    # case metadata right after the header, at reading time (indexes unchanged)
    if add_case_details:
        for d in kept:
            details = case_details(d.metadata) if d.metadata.get("doc_type") == "case" else ""
            if details:
                header, _, text = d.page_content.partition("\n")
                d.page_content = f"{header}\nCase details: {details}\n{text}"

    # optional second ranking: the cross-encoder reads question and document together
    if reranker and kept:
        kept = kept[:rerank_candidates]
        scores = reranker.predict([(question, d.page_content) for d in kept])
        for d, s in zip(kept, scores):
            d.metadata["rerank_score"] = round(float(s), 4)
        kept.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
    return kept[:top_k]


def clear_vector_store_cache(path: str) -> None:
    """Delete vector store from disk and from in-memory cache."""
    if path in _VECTOR_STORE_CACHE:
        del _VECTOR_STORE_CACHE[path]
    if os.path.isdir(path):
        shutil.rmtree(path)