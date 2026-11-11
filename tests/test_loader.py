"""Tests for backend/document_loader.py on the real corpus.

Run from the repo root:  python -m pytest -q
The tests are skipped if data/Contest_Data is not present (it is not in git).
"""
from collections import Counter
from pathlib import Path

import pytest

from backend.document_loader import (
    LegalDoc, chunk_docs, clean_text, load_corpus, load_documents_from_folders, to_langchain,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "Contest_Data"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not extracted in data/")


@pytest.fixture(scope="module")
def corpus():
    skipped = []
    docs = load_corpus(DATA, skipped=skipped)
    return docs, skipped


def test_every_file_is_loaded_or_explicitly_skipped(corpus):
    docs, skipped = corpus
    n_files = len(list(DATA.rglob("*.json")))
    assert n_files == 1472
    assert len(docs) + len(skipped) == n_files
    assert len(skipped) <= 5  # only empty headers such as "INHERITANCE"


def test_all_nine_knowledge_bases_are_populated(corpus):
    docs, _ = corpus
    kbs = Counter((d.country, d.kb) for d in docs)
    for country in ("Italy", "Estonia", "Slovenia"):
        for kb in ("inheritance_code", "divorce_code", "cases"):
            assert kbs[(country, kb)] > 0, (country, kb)


def test_country_comes_from_folder_not_metadata(corpus):
    docs, _ = corpus
    for d in docs:
        folder_country = Path(d.source).parts[-3].lower()
        assert d.country.lower() == folder_country


def test_cases_have_law_and_codes_have_consistent_law(corpus):
    docs, _ = corpus
    for d in docs:
        assert d.law in ("Inheritance", "Divorce")
        if d.kb == "inheritance_code":
            assert d.law == "Inheritance"
        if d.kb == "divorce_code":
            assert d.law == "Divorce"


def test_ids_are_unique_and_deterministic(corpus):
    docs, _ = corpus
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids))
    assert ids == [d.id for d in load_corpus(DATA)]


def test_articles_cited_is_always_a_list_and_header_number_is_used(corpus):
    docs, _ = corpus
    assert all(isinstance(d.articles_cited, list) for d in docs)
    art = next(d for d in docs if d.source.endswith("Article_279.json") and d.country == "Italy")
    assert art.article == "580"            # number in the text, not the metadata
    assert "Art. 279" in art.articles_cited  # metadata kept as a reference


def test_cleaning_removes_special_separators():
    assert clean_text("580 TITLE\xa0 here\n\n  body ") == "580\nTITLE here\nbody"


def test_chunking_keeps_metadata_and_overlaps(corpus):
    docs, _ = corpus
    long_doc = next(d for d in docs if len(d.text) > 2500)
    parts = chunk_docs([long_doc])
    assert len(parts) >= 2
    assert all(p.country == long_doc.country and p.kb == long_doc.kb for p in parts)
    assert parts[0].text.split()[-40:] == parts[1].text.split()[:40]  # 40-word overlap
    assert {p.id for p in parts} == {f"{long_doc.id}#c{i}" for i in range(len(parts))}


def test_langchain_documents_carry_header_and_metadata(corpus):
    docs, _ = corpus
    lc = to_langchain(docs[:3])
    assert lc[0].page_content.startswith(f"Country: {docs[0].country}")
    assert lc[0].metadata["source"] == docs[0].source
    assert "text" not in lc[0].metadata


def test_ui_entry_point_on_a_single_kb_folder():
    folder = DATA / "Estonia" / "Divorce_estonia"
    lc = load_documents_from_folders([str(folder), str(folder)])  # duplicates ignored
    assert len({d.metadata["parent_id"] for d in lc}) == 50
    assert all(d.metadata["kb"] == "divorce_code" for d in lc)


def test_invalid_country_is_rejected():
    with pytest.raises(ValueError):
        LegalDoc(id="x", country="France", law="Divorce", doc_type="code", kb="divorce_code",
                 title="t", source="s", text="x" * 40)
