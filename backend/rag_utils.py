# backend/rag_utils.py
from __future__ import annotations

import os
import re
from typing import Dict, List

from .config import RAGConfig


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


def fix_citations(answer: str, n_sources: int) -> tuple[str, int]:
    """Remove citations to sources that do not exist (e.g. [S26] when there are
    10 sources: the model confused an article number with a source label).
    Returns the cleaned answer and how many citations were removed."""
    invalid = [m for m in re.findall(r"\[S(\d+)\]", answer) if not 1 <= int(m) <= n_sources]
    for m in set(invalid):
        answer = answer.replace(f"[S{m}]", "")
    return answer, len(invalid)


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
