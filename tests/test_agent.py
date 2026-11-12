from backend.rag_utils import KB_DESCRIPTIONS, expand_query, fix_citations
from backend.config import RAGConfig

# Checks that do not call the LLM. Run from the repo root: python -m pytest -q


def test_query_expansion_adds_terms_of_the_other_codes():
    q = expand_query("Reserved share of children in Slovenia")
    assert "statutory portion" in q and "forced heirs" in q


def test_query_without_legal_terms_is_unchanged():
    assert expand_query("How long do cases last?") == "How long do cases last?"


def test_citations_to_missing_sources_are_removed():
    answer, removed = fix_citations("Half of the share [S26] (art. 26) [S7].", n_sources=10)
    assert answer == "Half of the share  (art. 26) [S7]." and removed == 1


def test_every_database_has_a_description():
    names = {p.split("/")[-1] for p in RAGConfig().vector_store_dirs}
    assert names == set(KB_DESCRIPTIONS)
