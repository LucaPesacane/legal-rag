from collections import Counter
from pathlib import Path

import pytest

from backend.document_loader import case_details, clean_text, load_documents_from_folders

# [+] Checks on the real corpus. Run from the repo root: python -m pytest -q
DATA = Path("data/Contest_Data")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not in data/")


# [+] All documents of the corpus, loaded once.
@pytest.fixture(scope="module")
def docs():
    return load_documents_from_folders([str(DATA)])


# [+] 1472 files, only 1 skipped (Article_815-819.json is an empty header).
def test_all_files_loaded(docs):
    assert len({d.metadata["id"] for d in docs}) == 1471


# [+] 3 countries x 3 knowledge bases.
def test_nine_knowledge_bases(docs):
    kbs = Counter((d.metadata["country"], d.metadata["kb"]) for d in docs)
    assert len(kbs) == 9


# [+] Article number from the text: Article_279.json contains article 580 (279 is only cited).
def test_article_number_from_text(docs):
    d = next(d for d in docs if d.metadata["source"].endswith("Italy\\Inheritance_italy\\Article_279.json")
             or d.metadata["source"].endswith("Italy/Inheritance_italy/Article_279.json"))
    assert d.metadata["article"] == "580"
    assert d.metadata["articles_cited"] == ["Art. 279"]


# [+] Almost every code article gets a number.
def test_every_code_article_has_a_number(docs):
    codes = [d for d in docs if d.metadata["doc_type"] == "code"]
    missing = [d.metadata["id"] for d in codes if d.metadata["article"] is None]
    assert len(missing) <= 2, missing


# [+] Long documents are split and every chunk keeps the header.
def test_long_documents_are_split_with_header(docs):
    parts = [d for d in docs if d.metadata["n_chunks"] > 1]
    assert parts
    assert all(d.page_content.startswith("Country: ") for d in parts)


# [+] Line separators, non-breaking and multiple spaces are cleaned.
def test_clean_text():
    assert clean_text("580\u2028TITLE\xa0 here\n\n  body ") == "580\nTITLE here\nbody"


# [+] Case metadata line: real values kept, "missing" values skipped.
def test_case_details_keeps_values_and_skips_missing():
    m = {"doc_type": "case", "case_id": "426/2019", "cost": "3'757,60 euro", "duration": "years",
         "disputed_issues": "No data", "articles_cited": ["Art. 536"], "proceeding": "extrajudicial"}
    assert case_details(m) == "cost: 3'757,60 euro | articles cited: Art. 536 | proceeding: extrajudicial"
