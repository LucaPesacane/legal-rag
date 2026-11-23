import json
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Role of this module:
#This is the ingestion layer: it converts your raw JSON legal cases into
# LangChain Documents that can then be embedded and stored.
# Used mainly by vector_store.py and the Vector DB Builder page.

# [+] Folder name -> country name used in the metadata
COUNTRIES = {"italy": "Italy", "estonia": "Estonia", "slovenia": "Slovenia"}

# [+] Only documents longer than 1500 characters are split (overlap keeps concepts whole)
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)


# [+] Remove line separators, non-breaking spaces and empty lines.
def clean_text(text: str) -> str:
    text = text.replace(" ", "\n").replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# [+] Article number read from the first lines of the text ("580", "10.", "768-TER", "ARTICLE 160").
def article_number(text: str) -> Optional[str]:
    # civil_codes_used is not reliable: sometimes it is an article cited in the text
    for line in text.splitlines()[:3]:
        m = re.fullmatch(r"(?:art(?:icle)?\.?\s*)?(\d+[a-z]?(?:\.\d+)?(?:-[A-Z]+)?)\.?", line, re.I)
        if m:
            return m.group(1).upper()
    return None


# [+] Uniform metadata: country and knowledge base from the folder path, case or code fields.
def _build_metadata(meta: dict, path: Path, text: str) -> dict:
    # "type" means the country for codes but judicial/extrajudicial for cases
    country = COUNTRIES[path.parent.parent.name.lower()]
    folder = path.parent.name.lower()
    if "cases" in folder:
        kb = "cases"
    elif folder.startswith("inheritance"):
        kb = "inheritance_code"
    else:
        kb = "divorce_code"

    # civil_codes_used can be a string, a list or missing
    cited = meta.get("civil_codes_used") or []
    cited = [cited] if isinstance(cited, str) else list(cited)

    metadata = {
        "source": str(path),
        "country": country,
        "law": meta.get("law"),
        "kb": kb,
        "articles_cited": cited,
    }
    if kb == "cases":
        metadata["doc_type"] = "case"
        metadata["case_id"] = meta.get("CASE_ID") or path.stem
        metadata["proceeding"] = meta.get("type")
        skip = ("law", "state", "type", "CASE_ID", "civil_codes_used")
        metadata.update({k: v for k, v in meta.items() if k not in skip})
    else:
        metadata["doc_type"] = "code"
        metadata["article"] = article_number(text)
    return metadata


# [+] Header added to every chunk: country, law and article or case number.
def _header(m: dict) -> str:
    if m["doc_type"] == "code":
        return f"Country: {m['country']} | Law: {m['law']} | Civil code article {m['article']}"
    return f"Country: {m['country']} | Law: {m['law']} | Past case: {m['case_id']}"


# [+] Metadata already shown elsewhere, and values that only mean "missing"
_NOT_DETAILS = {"source", "country", "law", "kb", "doc_type", "case_id", "id", "chunk", "n_chunks", "score", "db_name"}
_MISSING = {"", "no data", "unspecified", "not specified", "none", "n/a", "[]"}


# [+] One line with the metadata of a past case (duration, cost, articles...).
# The case texts have no punctuation ("375760 euros" for 3'757,60 euro): the metadata keep the real values.
def case_details(m: dict) -> str:
    parts = []
    for k, v in m.items():
        if k in _NOT_DETAILS or k == "rerank_score":
            continue
        v = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
        if v.strip().lower() in _MISSING or (k == "duration" and not re.search(r"\d", v)):
            continue  # e.g. duration "years" without a number
        parts.append(f"{k.replace('_', ' ')}: {v}")
    return " | ".join(parts)


# [~] Load every JSON file: clean text, uniform metadata, chunks with header.
def load_documents_from_folders(folders: List[str]) -> List[Document]:
    all_docs: List[Document] = []
    seen = set()

    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"[document_loader] Folder does not exist: {folder_path}")
            continue

        for json_file in sorted(folder_path.rglob("*.json")):
            if json_file in seen:  # same folder selected twice
                continue
            seen.add(json_file)
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = clean_text(str(data.get("content", "")))
                if len(text) < 30:  # empty headers such as "INHERITANCE"
                    print(f"[document_loader] Skipped (no content): {json_file}")
                    continue
                metadata = _build_metadata(data.get("metadata") or {}, json_file, text)
            except Exception as e:
                print(f"[document_loader] Error loading {json_file}: {e}")
                continue

            # stable id (same file -> same id), used by the evaluation (Hit@k)
            metadata["id"] = f"{metadata['country']}/{metadata['kb']}/{json_file.stem}"
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                m = {**metadata, "chunk": i, "n_chunks": len(chunks)}
                all_docs.append(Document(page_content=f"{_header(m)}\n{chunk}", metadata=m))

    return all_docs
