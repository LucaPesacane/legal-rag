# backend/rag_multiagent.py
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .rag_single_agent import ANSWER_PROMPT, NO_SOURCES_PROMPT
from .rag_utils import KB_DESCRIPTIONS, _get_vector_db_dirs, expand_query, fix_citations
from .vector_store import search_vector_stores

# Role of this module:
# Task B - multi-agent system with a supervisor.
#   - 9 specialized agents, one per DB (country x knowledge base), each described
#     by its KB_DESCRIPTIONS entry and able to retrieve + answer on its own DB only;
#   - the supervisor reads the descriptions, chooses the agents for the question,
#     runs them in parallel and merges their partial answers into one answer.

NO_INFO = "NO RELEVANT INFORMATION"


# =====================================================================
# Supervisor step 1: which specialized agents?
# =====================================================================
def _choose_agents(question: str, db_map: Dict[str, str], config: RAGConfig) -> Tuple[List[str], str, str]:
    agents_block = "\n".join(f"- {name}: {KB_DESCRIPTIONS.get(name, name)}" for name in db_map)

    system_prompt = (
        "You are the supervisor of specialized legal agents. Each agent knows one knowledge base "
        "of one country (Italy, Estonia, Slovenia).\n"
        "Choose the agents needed to answer the question:\n"
        "- one country mentioned -> only the agents of that country;\n"
        "- comparisons or no country mentioned -> the agents of ALL the countries involved;\n"
        "- choose code agents for rules, case agents for past cases, durations, costs;\n"
        "- no agent (empty list) only for generic definitions or questions outside inheritance and divorce law.\n"
        "Also write search_query: the question translated into English (the documents are in English).\n"
        'Reply with JSON only: {"agents": ["name", ...], "search_query": "...", "reason": "short"}'
    )
    user_prompt = f"Agents:\n{agents_block}\n\nQuestion:\n{question}"

    # same small routing model as Task A: the two systems differ only in the architecture
    router = LLMBackend(replace(config, llm_model_name=config.router_model_name, temperature=0.0))
    resp = router.chat(system_prompt, user_prompt)
    try:
        data = json.loads(re.search(r"\{.*\}", resp, re.S).group())
        agents = [a for a in data.get("agents", []) if a in db_map]
        return agents, expand_query(str(data.get("search_query") or question)), str(data.get("reason", ""))
    except Exception:
        return list(db_map), expand_query(question), f"Supervisor output not valid JSON ({resp[:80]!r}) -> all agents."


# =====================================================================
# Specialized agent: retrieval + partial answer on its own DB
# =====================================================================
def _run_agent(name: str, docs: List[Document], first_number: int, question: str, config: RAGConfig) -> str:
    if not docs:
        return NO_INFO
    # sources keep their global number, so the supervisor can merge citations as they are
    sources = "\n\n".join(f"[S{first_number + i}] {d.page_content}" for i, d in enumerate(docs))
    system_prompt = (
        f"You are the specialized agent for: {KB_DESCRIPTIONS.get(name, name)}\n"
        + ANSWER_PROMPT
        + f"\n- If your sources are not relevant to the question, reply only: {NO_INFO}"
    )
    return LLMBackend(config).chat(system_prompt, f"Sources:\n{sources}\n\nQuestion:\n{question}")


# =====================================================================
# MULTI-AGENT PIPELINE
# =====================================================================
def multiagent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    db_map = _get_vector_db_dirs(config)  # {agent/db name -> path}
    supervisor = LLMBackend(config)

    # ---- Supervisor: routing ----
    agents, search_query, reason = _choose_agents(question, db_map, config)
    if not agents:
        answer = supervisor.chat(NO_SOURCES_PROMPT, question)
        trace = f"**Supervisor**: no agent needed. {reason}" if show_reasoning else None
        return answer, [], trace

    # ---- Agents: retrieval on their own DB (sources numbered globally) ----
    embedding_model = get_embedding_model(config)
    agent_docs: Dict[str, List[Document]] = {}
    all_docs: List[Document] = []
    first_numbers: Dict[str, int] = {}
    for name in agents:
        docs = search_vector_stores(
            search_query, [db_map[name]], embedding_model,
            k_fetch=config.k_fetch, top_k=config.top_k, min_sim=config.min_sim,
        )
        first_numbers[name] = len(all_docs) + 1
        agent_docs[name] = docs
        all_docs += docs

    # ---- Agents: partial answers, in parallel (max 3 to respect the API rate limit) ----
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            name: pool.submit(_run_agent, name, agent_docs[name], first_numbers[name], question, config)
            for name in agents
        }
        partial = {name: f.result() for name, f in futures.items()}
    useful = {name: a for name, a in partial.items() if NO_INFO not in a}

    # ---- Supervisor: merge ----
    if not useful:
        answer = "The selected knowledge bases do not contain information to answer this question."
    elif len(useful) == 1:
        answer = next(iter(useful.values()))  # one agent: its answer is already final
    else:
        blocks = "\n\n".join(f"[Agent: {name}]\n{a}" for name, a in useful.items())
        system_prompt = (
            "You are a supervisor merging the partial answers of specialized legal agents "
            "into ONE coherent answer for the user.\n"
            "- Keep every source citation exactly as written ([S1], [S2], ...): never renumber them.\n"
            "- When several countries are involved, compare them and highlight divergences.\n"
            "- Do not add facts that are not in the partial answers; do not mention the agents.\n"
            "- Answer in the language of the question."
        )
        answer = supervisor.chat(system_prompt, f"Question:\n{question}\n\nPartial answers:\n{blocks}")
    answer = answer.replace("【", "[").replace("】", "]")
    answer, n_invalid = fix_citations(answer, len(all_docs))

    trace: Optional[str] = None
    if show_reasoning:
        lines = [f"- `{n}`: {len(agent_docs[n])} docs, "
                 f"{'answered' if n in useful else 'no relevant information'}" for n in agents]
        merge = ("partial answers combined by the supervisor" if len(useful) > 1
                 else "single useful answer returned as it is")
        trace = (
            f"**Supervisor routing**: {', '.join(agents)}. {reason}\n\n"
            f"**Search query**: {search_query}\n\n"
            f"**Specialized agents**:\n" + "\n".join(lines) + "\n\n"
            f"**Merge**: {len(useful)} useful answer(s), {merge}.\n\n"
            f"**Citation check**: {n_invalid} citation(s) to non-existing sources removed."
        )
    return answer, all_docs, trace
