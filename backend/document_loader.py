"""
backend/document_loader.py

Ingestion layer: reads the raw JSON legal corpus (Italy / Estonia / Slovenia,
civil code articles + past cases) and turns it into *normalized* records and
LangChain Documents ready for embedding.

Why a dedicated normalization step (see EDA in eda/):
  * the meaning of `metadata.type` changes between sources
      - civil codes: type = country ("ITALY")
      - past cases:  type = "judicial" / "extrajudicial", country in `state`
    -> country and knowledge base (KB) are taken from the FOLDER PATH,
       which is the only field that is consistent for all 1,472 files;
  * the folder `slovenia` is lower case while the others are capitalized;
  * `civil_codes_used` is a string ("Art. 10"), a list (["Art. 718"]) or
    missing -> always a list (`articles_cited`);
  * in ~6% of code articles `civil_codes_used` is an article *referenced* in the
    text, not the article itself (e.g. file Article_279.json contains art. 580)
    -> the article number is parsed from the first line of the text;
  * texts contain U+2028 / non-breaking spaces and a few empty headers.

Public API
  load_corpus(root)                 -> list[LegalDoc]       (normalized records)
  chunk_docs(docs, ...)             -> list[LegalDoc]       (long docs split)
  to_langchain(docs)                -> list[Document]
  load_documents_from_folders(dirs) -> list[Document]       (used by the UI)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants: how folders map to countries and knowledge bases
# ---------------------------------------------------------------------------
COUNTRIES: Dict[str, str] = {"italy": "Italy", "estonia": "Estonia", "slovenia": "Slovenia"}
COUNTRY_CODES: Dict[str, str] = {"Italy": "IT", "Estonia": "EE", "Slovenia": "SI"}

# knowledge base = (doc_type, law-or-None); for cases the law comes from metadata
KB_NAMES = ("inheritance_code", "divorce_code", "cases")

# values that mean "no information" in the raw metadata
_MISSING = {"", "not specified", "unspecified", "none", "null", "n/a", "na", "unknown", "-"}

# first line of a code article: "580", "768-QUINQUIES", "10.", "64.1", "Art. 12"
_ARTICLE_HEAD = re.compile(r"^(?:art(?:icle)?\.?\s*)?(\d+[a-z]?(?:\.\d+)?(?:-[A-Z]+)?)\.?$", re.I)
_ARTICLE_REF = re.compile(r"\d+[a-z]?(?:\.\d+)?")

MIN_CHARS = 30  # shorter cleaned texts are headers without content (e.g. "INHERITANCE")


# ---------------------------------------------------------------------------
# Normalized record (validated with Pydantic)
# ---------------------------------------------------------------------------
class LegalDoc(BaseModel):
    id: str = Field(..., min_length=1)
    country: str
    law: str                      # "Inheritance" | "Divorce"
    doc_type: str                 # "code" | "case"
    kb: str                       # "inheritance_code" | "divorce_code" | "cases"
    title: str                    # article heading or case identifier
    article: Optional[str] = None # own article number (codes only)
    articles_cited: List[str] = Field(default_factory=list)
    proceeding: Optional[str] = None  # judicial / extrajudicial (cases only)
    extra: Dict[str, Any] = Field(default_factory=dict)  # other case metadata
    source: str                   # original file path
    text: str = Field(..., min_length=MIN_CHARS)
    chunk: int = 0                # chunk index (0 if the doc is not split)
    n_chunks: int = 1

    @field_validator("country")
    @classmethod
    def _country_ok(cls, v: str) -> str:
        if v not in COUNTRY_CODES:
            raise ValueError(f"unknown country {v!r}")
        return v

    @field_validator("law")
    @classmethod
    def _law_ok(cls, v: str) -> str:
        if v not in ("Inheritance", "Divorce"):
            raise ValueError(f"unknown law {v!r}")
        return v

    @field_validator("kb")
    @classmethod
    def _kb_ok(cls, v: str) -> str:
        if v not in KB_NAMES:
            raise ValueError(f"unknown kb {v!r}")
        return v

    # text actually embedded and shown to the LLM: a short metadata header lets
    # the embedding "know" country / area / type of the passage, and gives the
    # generator what it needs to cite sources correctly.
    @property
    def header(self) -> str:
        parts = [f"Country: {self.country}", f"Law: {self.law}"]
        if self.doc_type == "code":
            parts.append(f"Civil code article {self.article or '?'}: {self.title}")
        else:
            parts.append(f"Past case ({self.proceeding or 'n/a'}): {self.title}")
            if self.articles_cited:
                parts.append("Articles: " + ", ".join(self.articles_cited))
        if self.n_chunks > 1:
            parts.append(f"Part {self.chunk + 1}/{self.n_chunks}")
        return " | ".join(parts)

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    """Normalize line separators / nbsp, strip lines, drop empty lines."""
    t = raw.replace(" ", "\n").replace(" ", "\n").replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines())
    return "\n".join(ln for ln in lines if ln)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: List[str] = []
    for it in items:
        s = str(it).strip()
        if s and s.lower() not in _MISSING and s not in out:
            out.append(s)
    return out


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return None if value.strip().lower() in _MISSING else value.strip()
    if isinstance(value, list):
        vals = [_clean_value(v) for v in value]
        vals = [v for v in vals if v is not None]
        return vals or None
    return value


def _country_and_kb(path: Path) -> tuple[str, str, Optional[str]]:
    """Country, kb and (for codes) law, inferred from the folder names."""
    country = None
    for part in path.parts:
        if part.lower() in COUNTRIES:
            country = COUNTRIES[part.lower()]
    folder = path.parent.name.lower()
    if "cases" in folder:
        kb, law = "cases", None
    elif folder.startswith("inheritance"):
        kb, law = "inheritance_code", "Inheritance"
    elif folder.startswith("divorce"):
        kb, law = "divorce_code", "Divorce"
    else:
        raise ValueError(f"cannot infer knowledge base from folder {path.parent}")
    if country is None:
        raise ValueError(f"cannot infer country from path {path}")
    return country, kb, law


def _split_article_header(text: str) -> tuple[Optional[str], str, str]:
    """For a code article return (article_number, title, body).

    Typical layouts:  "580\\nRIGHTS OF ...\\nbody"  or  "ACT\\n10.\\nApplication of ...\\nbody"
    """
    lines = text.splitlines()
    for i, line in enumerate(lines[:3]):
        m = _ARTICLE_HEAD.match(line)
        if m:
            title = lines[i + 1] if i + 1 < len(lines) else ""
            body = "\n".join(lines[i + 2:]) if i + 2 < len(lines) else title
            return m.group(1).upper(), title.strip().rstrip("."), body
    return None, lines[0][:80] if lines else "", text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _records_from_file(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    return [it for it in items if isinstance(it, dict)]


def _normalize(item: dict, path: Path) -> Optional[LegalDoc]:
    raw = item.get("content") or item.get("text") or ""
    text = clean_text(str(raw))
    if len(text) < MIN_CHARS:
        return None

    meta = item.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    country, kb, law = _country_and_kb(path)
    cited = _as_list(meta.get("civil_codes_used"))

    if kb == "cases":
        law = str(meta.get("law", "")).strip().capitalize()
        case_id = _clean_value(meta.get("CASE_ID"))
        skip = {"law", "state", "type", "CASE_ID", "civil_codes_used"}
        extra = {k: v for k, v in ((k, _clean_value(v)) for k, v in meta.items() if k not in skip) if v is not None}
        return LegalDoc(
            id="pending", country=country, law=law, doc_type="case", kb=kb,
            title=str(case_id) if case_id else path.stem,
            articles_cited=cited, proceeding=_clean_value(meta.get("type")),
            extra=extra, source=str(path), text=text,
        )

    article, title, _body = _split_article_header(text)
    if article is None and cited:  # fallback: metadata reference
        nums = _ARTICLE_REF.findall(cited[0])
        article = nums[0] if nums else None
    return LegalDoc(
        id="pending", country=country, law=law, doc_type="code", kb=kb,
        title=title or f"Article {article}", article=article,
        articles_cited=[c for c in cited if article is None or article.split("-")[0] not in c],
        source=str(path), text=text,
    )


def load_corpus(root: str | Path, skipped: Optional[list] = None) -> List[LegalDoc]:
    """Load every JSON under `root` into validated LegalDoc records.

    Files are processed in sorted order so that ids are deterministic:
    <CC>_<kb>_<nnnn>, e.g. IT_cases_0042.
    `skipped`, if given, collects (path, reason) for dropped records.
    """
    root = Path(root)
    docs: List[LegalDoc] = []
    counters: Dict[str, int] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            items = _records_from_file(path)
        except Exception as e:  # malformed JSON: report, do not crash
            if skipped is not None:
                skipped.append((str(path), f"invalid json: {e}"))
            continue
        for item in items:
            try:
                doc = _normalize(item, path)
            except Exception as e:
                if skipped is not None:
                    skipped.append((str(path), f"invalid record: {e}"))
                continue
            if doc is None:
                if skipped is not None:
                    skipped.append((str(path), f"text shorter than {MIN_CHARS} chars"))
                continue
            key = f"{COUNTRY_CODES[doc.country]}_{doc.kb}"
            counters[key] = counters.get(key, 0) + 1
            doc.id = f"{key}_{counters[key]:04d}"
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Chunking: most documents fit in one chunk; only long ones are split
# ---------------------------------------------------------------------------
def chunk_docs(docs: Iterable[LegalDoc], max_chars: int = 1500,
               chunk_words: int = 200, overlap_words: int = 40) -> List[LegalDoc]:
    """Split documents longer than `max_chars` into overlapping word windows.

    Word windows (not sentences) because the case summaries have no
    punctuation. Every chunk keeps all metadata of its parent document.
    """
    out: List[LegalDoc] = []
    step = chunk_words - overlap_words
    for d in docs:
        if len(d.text) <= max_chars:
            out.append(d)
            continue
        words = d.text.split()
        starts = list(range(0, max(len(words) - overlap_words, 1), step))
        for i, s in enumerate(starts):
            piece = " ".join(words[s:s + chunk_words])
            out.append(d.model_copy(update={
                "id": f"{d.id}#c{i}", "text": piece, "chunk": i, "n_chunks": len(starts),
            }))
    return out


# ---------------------------------------------------------------------------
# Conversion to LangChain + backward-compatible entry point for the UI
# ---------------------------------------------------------------------------
def to_langchain(docs: Iterable[LegalDoc]) -> List[Document]:
    result = []
    for d in docs:
        meta = d.model_dump(exclude={"text"})
        meta["parent_id"] = d.id.split("#")[0]
        result.append(Document(page_content=d.embed_text, metadata=meta))
    return result


def load_documents_from_folders(folders: List[str], chunk: bool = True) -> List[Document]:
    """Used by pages/2_Vector_DB_Builder.py: any folder inside Contest_Data
    (the root, a country, or a single knowledge base) can be passed."""
    all_docs: List[LegalDoc] = []
    seen = set()
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[document_loader] Folder does not exist: {folder_path}")
            continue
        for d in load_corpus(folder_path):
            if d.id not in seen:  # the same folder selected twice
                seen.add(d.id)
                all_docs.append(d)
    if chunk:
        all_docs = chunk_docs(all_docs)
    return to_langchain(all_docs)
