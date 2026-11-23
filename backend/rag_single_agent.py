from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model, get_reranker
from .llm_provider import LLMBackend
from .rag_utils import KB_DESCRIPTIONS, _get_vector_db_dirs, expand_query, fix_citations
from .vector_store import search_vector_stores

# [+] Answer rules, shared with Task B
ANSWER_PROMPT = (
    "You are a legal assistant on Italian, Estonian and Slovenian civil law "
    "(inheritance and divorce). Answer ONLY with information from the numbered sources.\n"
    "- Answer only what is asked, in a few sentences, without extra notes or background.\n"
    "- Give the complete rule that is asked (all its cases, shares or conditions).\n"
    "- Do not confuse intestate succession shares (no will) with the reserved / statutory / "
    "compulsory portion guaranteed to forced heirs: use the provision that matches the question.\n"
    "- Cite the sources right after each statement using ONLY the labels written before each "
    "source (e.g. [S3]); never use an article number as a label.\n"
    "- Always say which country a rule or case belongs to and name the articles.\n"
    "- For a past case give its number, duration and outcome (and the amounts, if any).\n"
    "- If several countries are involved, compare them and point out the differences.\n"
    "- If the sources do not contain the answer, say so clearly; do not invent rules.\n"
    "- Write the answer in the same language as the question (an English question gets an "
    "English answer, even if it is about Italy), clearly and concisely."
)
# [+] Answer without documents, generic or out-of-domain questions
NO_SOURCES_PROMPT = (
    "You are a legal assistant specialized in inheritance and divorce law of Italy, Estonia "
    "and Slovenia. No document from the database was used for this answer.\n"
    "- Answer briefly from general knowledge.\n"
    "- Then say that the answer is not based on the sources of the system; if the question is "
    "outside inheritance and divorce law, also say that the system is specialized in those topics."
)

# [+] Thought: one JSON call -> retrieval needed?, which DBs, English search query, legal area.
def _decide(question: str, db_map: Dict[str, str], config: RAGConfig) -> Tuple[bool, List[str], str, Optional[str], str]:
    kb_block = "\n".join(f"- {name}: {KB_DESCRIPTIONS.get(name, name)}" for name in db_map)

    system_prompt = (
        "You route legal questions about Italy, Estonia and Slovenia (inheritance and divorce law).\n"
        "Decide if the question needs the document databases and which ones to search.\n"
        "Rules:\n"
        "- needs_retrieval = false ONLY for generic or definitional questions not tied to a country,\n"
        "  or for questions outside inheritance and divorce law (e.g. criminal law).\n"
        "- If no country is mentioned or countries are compared, use the databases of ALL the countries involved.\n"
        "- Questions about property of spouses, separation or divorce -> divorce code (+ cases if useful).\n"
        "- Questions about heirs, wills, estates -> inheritance code (+ cases if useful).\n"
        "- Questions about past cases, court decisions, durations or costs -> cases.\n"
        "- When in doubt, retrieve.\n"
        "Also write search_query: the question translated into English (the documents are in English),\n"
        'and law: "Inheritance" or "Divorce" (null if the question concerns both or neither).\n'
        'Reply with JSON only: {"needs_retrieval": true, "databases": ["name", ...], '
        '"search_query": "...", "law": "Inheritance", "reason": "short"}'
    )
    user_prompt = f"Databases:\n{kb_block}\n\nQuestion:\n{question}"

    # smaller model, low reasoning
    router = LLMBackend(replace(config, llm_model_name=config.router_model_name, temperature=0.0,
                                 reasoning_effort=config.router_reasoning_effort))
    resp = router.chat(system_prompt, user_prompt)

    try:
        data = json.loads(re.search(r"\{.*\}", resp, re.S).group())
        need = bool(data.get("needs_retrieval", True))
        dbs = [d for d in data.get("databases", []) if d in db_map]
        reason = str(data.get("reason", ""))
        # query in English (language of the documents) + synonyms of the three codes
        search_query = expand_query(str(data.get("search_query") or question))
        law = data.get("law") if data.get("law") in ("Inheritance", "Divorce") else None
    except Exception:
        # invalid JSON: search everything
        return True, list(db_map), expand_query(question), None, f"Router output not valid JSON ({resp[:80]!r}) -> all databases."

    if need and not dbs:
        return True, list(db_map), search_query, law, reason + " (no valid database named -> all databases)"
    return need, dbs, search_query, law, reason

# [+] Action: vector search on the chosen DBs, top_k per country.
def _retrieve(question: str, db_names: List[str], db_map: Dict[str, str], config: RAGConfig,
              law: Optional[str] = None, original: Optional[str] = None) -> List[Document]:
    embedding_model = get_embedding_model(config)

    # each country keeps its own top_k: Italy (largest corpus) cannot crowd out the others
    by_country: Dict[str, List[str]] = {}
    for name in db_names:
        by_country.setdefault(name.split("_")[0], []).append(db_map[name])

    docs: List[Document] = []
    for paths in by_country.values():
        docs += search_vector_stores(
            question, paths, embedding_model,
            k_fetch=config.k_fetch, top_k=config.top_k, min_sim=config.min_sim,
            law=law if config.filter_cases_by_law else None,
            reranker=get_reranker(config) if config.use_rerank else None,
            rerank_candidates=config.rerank_candidates,
            extra_queries=[original] if config.search_original_question and original else [],
            add_case_details=config.add_case_details,
        )
    return docs

# [+] Observation: summary of what was retrieved (computed, no LLM).
def _observation(db_names: List[str], docs: List[Document]) -> str:
    if not docs:
        return f"Searched {', '.join(db_names)}: no document above the similarity threshold."

    countries = Counter(d.metadata["country"] for d in docs)
    articles = Counter(
        f"{d.metadata['country']} art. {d.metadata['article']}" for d in docs if d.metadata.get("article")
    )
    cases = [f"{d.metadata['case_id']} ({d.metadata['score']:.2f})" for d in docs if d.metadata["doc_type"] == "case"]

    lines = [
        f"Searched: {', '.join(db_names)}",
        f"Documents kept: {len(docs)} ({', '.join(f'{c}: {n}' for c, n in countries.items())}), "
        f"similarity {docs[-1].metadata['score']:.2f}-{max(d.metadata['score'] for d in docs):.2f}",
    ]
    if articles:
        lines.append("Code articles: " + ", ".join(a for a, _ in articles.most_common(6)))
    if cases:
        lines.append("Most similar cases: " + "; ".join(cases[:3]))
    return "\n".join(lines)


# [~] Numbered sources [S1], [S2]... ("S" avoids confusion with article numbers).
def _build_context(docs: List[Document]) -> str:
    return "\n\n".join(f"[S{i}] {d.page_content}" for i, d in enumerate(docs, start=1))

# [~] Thought -> Action (+ fallback on all DBs) -> Observation -> Answer.
def single_agent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    db_map = _get_vector_db_dirs(config)  # {db_name -> path}
    llm_backend = LLMBackend(config)

    # Thought
    need_retrieval, db_names, search_query, law, reason = _decide(question, db_map, config)

    # Action
    docs: List[Document] = []
    action = "No retrieval: answer from general knowledge."
    if need_retrieval:
        docs = _retrieve(search_query, db_names, db_map, config, law, question)
        action = f"Vector retrieval on: {', '.join(db_names)}\nSearch query: {search_query}"
        if config.search_original_question:
            action += "\n(+ the original question)"
        if config.use_rerank:
            action += f"\nReranked by: {config.reranker_model_name}"
        if config.filter_cases_by_law and law:
            action += f"\nPast cases filtered by law: {law}"
        # nothing relevant in the chosen DBs -> search all DBs
        if not docs and len(db_names) < len(db_map):
            db_names = list(db_map)
            docs = _retrieve(search_query, db_names, db_map, config, law, question)
            action +="\n-> nothing relevant, retried on all databases"

    # Observation
    observation = _observation(db_names, docs) if need_retrieval else "No documents used."

    # Answer
    if docs:
        system_prompt = ANSWER_PROMPT
        user_prompt = f"Sources:\n{_build_context(docs)}\n\nSummary of the retrieval:\n{observation}\n\nQuestion:\n{question}"
    else:
        system_prompt = NO_SOURCES_PROMPT
        user_prompt = question

    answer = llm_backend.chat(system_prompt, user_prompt)
    answer = answer.replace("【", "[").replace("】", "]").replace("\u200b", "")  # odd brackets / invisible spaces of gpt-oss
    answer, n_invalid = fix_citations(answer, len(docs))

    reasoning_trace: Optional[str] = None
    if show_reasoning:
        reasoning_trace = (
            f"**Thought**: retrieval needed = {need_retrieval}. {reason}\n\n"
            f"**Action**: {action}\n\n"
            f"**Observation**:\n```text\n{observation}\n```\n\n"
            f"**Citation check**: {n_invalid} citation(s) to non-existing sources removed."
        )
    return answer, docs, reasoning_trace
