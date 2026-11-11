from collections import Counter
from pathlib import Path

import pytest

from backend.document_loader import clean_text, load_documents_from_folders

# Checks on the real corpus. Run from the repo root: python -m pytest -q
DATA = Path("data/Contest_Data")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not in data/")


@pytest.fixture(scope="module")
def docs():
    return load_documents_from_folders([str(DATA)])


def test_all_files_loaded(docs):
    # 1472 files, only 1 skipped (Article_815-819.json is an empty header)
    assert len({d.metadata["id"] for d in docs}) == 1471


def test_nine_knowledge_bases(docs):
    kbs = Counter((d.metadata["country"], d.metadata["kb"]) for d in docs)
    assert len(kbs) == 9


def test_article_number_from_text(docs):
    # file Article_279.json contains article 580 (279 is only cited)
    d = next(d for d in docs if d.metadata["source"].endswith("Italy\\Inheritance_italy\\Article_279.json")
             or d.metadata["source"].endswith("Italy/Inheritance_italy/Article_279.json"))
    assert d.metadata["article"] == "580"
    assert d.metadata["articles_cited"] == ["Art. 279"]


def test_long_documents_are_split_with_header(docs):
    parts = [d for d in docs if d.metadata["n_chunks"] > 1]
    assert parts
    assert all(d.page_content.startswith("Country: ") for d in parts)


def test_clean_text():
    assert clean_text("580 TITLE\xa0 here\n\n  body ") == "580\nTITLE here\nbody"
