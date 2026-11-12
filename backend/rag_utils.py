# backend/rag_utils.py
from __future__ import annotations

import os
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document

from .config import RAGConfig
from .llm_provider import LLMBackend
from .vector_store import load_vector_store


# Hand-written description of each knowledge base (written from the EDA).
# The LLM reads them to decide which DBs to search (Task A) or which
# specialized agents to call (Task B).
# The same legal concept has different names in the three codes
# (e.g. Italy "reserved share", Slovenia "statutory portion", Estonia "compulsory portion").
# If a query contains one term of a group, the other terms are added (query expansion).
LEGAL_SYNONYMS: List[List[str]] = [
    ["reserved share", "forced share", "legitime", "compulsory portion", "statutory portion", "forced heirs"],
    ["intestate succession", "statutory heirs", "legal heirs", "succession by law", "order of succession"],
    ["community property", "community of property", "joint property", "matrimonial property", "marital property"],
    ["separate property", "personal property", "separation of property"],
    ["will", "testament", "testamentary disposition"],
]


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [t for group in LEGAL_SYNONYMS if any(term in q for term in group) for t in group if t not in q]
    return f"{query} ({', '.join(extra)})" if extra else query


KB_DESCRIPTIONS: Dict[str, str] = {
    "italy_inheritance_code": "Italian Civil Code, Book II arts. 456-768 (succession): opening of succession, "
        "acceptance and renunciation, inventory, legitimate (intestate) heirs, reserved share of forced heirs, "
        "wills and legacies, donations and their reduction, division of the estate.",
    "italy_divorce_code": "Italian Civil Code arts. 159-218 (property relations between spouses): statutory "
        "community of property, personal property, administration of community assets, matrimonial property "
        "fund, separation of property, dissolution and division of the community.",
    "italy_cases": "Past Italian cases (341 inheritance, 129 divorce; judicial and extrajudicial): division of "
        "inherited assets among heirs, co-ownership, reduction of donations, division of spouses' property, "
        "with duration and cost of the proceedings.",
    "estonia_inheritance_code": "Estonian Law of Succession Act: intestate succession and order of successors, "
        "wills and succession contracts, reciprocal wills of spouses, compulsory portion, acceptance and "
        "renunciation, notarial succession proceedings and certificate of succession.",
    "estonia_divorce_code": "Estonian Family Law Act (proprietary relationships of spouses): marital property "
        "contract, joint property and separate property regimes, administration of joint property, divorce "
        "and division of joint property, termination of the marriage.",
    "estonia_cases": "Past Estonian court cases (91 divorce, 31 inheritance): division of joint property and "
        "debts after divorce, compensation, ownership of apartments, inheritance disputes and compromises.",
    "slovenia_inheritance_code": "Slovenian Inheritance Act: title to inheritance (law or will), legal heirs and "
        "orders of succession, compulsory (reserved) portion, wills and witnesses, legacies, gifts counted in "
        "the hereditary portion, probate proceedings.",
    "slovenia_divorce_code": "Slovenian Family Code arts. 63-93 (property relations of spouses): marital "
        "property regime, prenuptial and marital agreements and their register, joint and separate property, "
        "obligations of spouses, effects of divorce on property.",
    "slovenia_cases": "Past Slovenian court cases (101 divorce, 100 inheritance; many appeals and Supreme Court "
        "decisions): division of marital property, inheritance proceedings, estate and pension disputes.",
}


def _get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    """
    Returns a mapping: {db_name -> folder_path}

    - If config.vector_store_dirs exists and is non-empty, use that.
    - Else fall back to single db: {<basename(vector_store_dir)>: vector_store_dir}.
    """
    dirs: List[str] = []

    if getattr(config, "vector_store_dirs", None):
        v = config.vector_store_dirs or []
        if isinstance(v, list) and len(v) > 0:
            dirs.extend(v)

    if not dirs:
        dirs.append(config.vector_store_dir)

    db_map: Dict[str, str] = {}
    for path in dirs:
        name = os.path.basename(os.path.normpath(path)) or path
        db_map[name] = path
    return db_map


def _describe_databases(
    db_map: Dict[str, str],
    embedding_model,
) -> Dict[str, str]:
    """
    Build a SHORT description for each DB based on a few documents' metadata.
    Used so the LLM (single agent or supervisor) can choose the right DB(s).
    """
    descriptions: Dict[str, str] = {}

    for db_name, path in db_map.items():
        try:
            vs = load_vector_store(path, embedding_model)
        except Exception:
            descriptions[db_name] = "Database could not be loaded."
            continue

# If the vector store vs has a docstore with an internal _dict,
# take all the Document objects from that internal dictionary
# and put them into a list called all_docs. Then take the first 20 documents
# from all_docs and put them into a list called docs. If any error occurs   

        docs: List[Document] = []
        try:
            if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
                all_docs = list(vs.docstore._dict.values())
                docs = all_docs[:20]
        except Exception:
            docs = []

        corpus_names = set()
        laws = set()
        types_ = set()
        other_tags = set()

        for d in docs:
            meta = d.metadata or {}
            if "law" in meta:
                laws.add(str(meta["law"]))
            if "type" in meta:
                types_.add(str(meta["type"]))
            for k in ("presence_of_children", "disputed_issues", "subject_of_succession", "financial_support", "subject_of_succession"):
                if k in meta:
                    other_tags.add(f"{k}={meta[k]}")

        parts = []
        if corpus_names:
            parts.append("corpus: " + ", ".join(sorted(corpus_names)))
        if laws:
            parts.append("law: " + ", ".join(sorted(laws)))
        if types_:
            parts.append("type: " + ", ".join(sorted(types_)))
        if other_tags:
            parts.append("tags: " + ", ".join(sorted(other_tags)))

        if parts:
            descriptions[db_name] = "; ".join(parts)
        else:
            descriptions[db_name] = "general corpus with mixed content."

    return descriptions


def _decide_which_dbs(
    question: str,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
    llm_backend: LLMBackend,
) -> Tuple[List[str], str]:
    db_names = list(db_map.keys())
    if len(db_names) == 1:
        return db_names, "Only one DB available → using it by default."

    lines = []
    for name in db_names:
        desc = db_descriptions.get(name, "no description")
        lines.append(f"- {name}: {desc}")
    db_descr_block = "\n".join(lines)

    system_prompt = (
        "You are selecting which knowledge databases are relevant for a user question.\n"
        "You are given a list of database names with short descriptions.\n"
        "Return a comma-separated list of the database names that should be used "
        "to answer the question. If only one database is relevant, return just that name. "
        "If multiple are relevant, include all of them. If none are relevant, return 'NONE'."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Available databases:\n{db_descr_block}\n\n"
        "Which database names should be used? Reply with names separated by commas, "
        "or 'NONE'."
    )

    resp = llm_backend.chat(system_prompt, user_prompt).strip()
    resp_lower = resp.lower()

    if "none" in resp_lower:
        return [], f"DB selection: model answered '{resp}' → NONE (no DB)."

    chosen = [name.strip() for name in resp.split(",") if name.strip()]
    chosen_valid = [c for c in chosen if c in db_map]

    if not chosen_valid:
        log = (
            f"DB selection: model answered '{resp}' but no valid DB name was parsed → "
            "falling back to ALL DBs."
        )
        return db_names, log

    log = (
        f"DB selection: model answered '{resp}' → using DBs: "
        + ", ".join(chosen_valid)
    )
    return chosen_valid, log


def _build_agent_config_log(
    config: RAGConfig,
    db_map: Dict[str, str],
    db_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a compact log describing the agent / DB settings:
      - LLM provider/model
      - Embedding provider/model
      - top_k, agentic_mode, use_multiagent
      - DB names, paths, and optional short descriptions
    """
    lines: List[str] = []

    lines.append(f"LLM provider: {config.llm_provider}")
    lines.append(f"LLM model: {config.llm_model_name}")
    lines.append(f"Embedding provider: {config.embedding_provider}")
    lines.append(f"Embedding model: {config.embedding_model_name}")
    lines.append(f"top_k: {config.top_k}")
    lines.append(f"agentic_mode: {config.agentic_mode}")
    use_multiagent = getattr(config, "use_multiagent", False)
    lines.append(f"use_multiagent: {use_multiagent}")

    lines.append("Vector DBs:")
    for name, path in db_map.items():
        if db_descriptions and name in db_descriptions:
            desc = db_descriptions[name]
            lines.append(f"  - {name}: path={path} | {desc}")
        else:
            lines.append(f"  - {name}: path={path}")

    return "\n".join(lines)
