"""Exploratory data analysis of the legal corpus.

Run from the repo root (venv active):
    python eda/eda_dataset.py            # default root: data/Contest_Data

Outputs in eda/output/: summary.md (tables + findings) and PNG charts to be
reused in the report and slides.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.document_loader import (  # noqa: E402
    _ARTICLE_REF, chunk_docs, load_corpus,
)

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "Contest_Data"
OUT = ROOT / "eda" / "output"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"Italy": "#2E7D32", "Estonia": "#1565C0", "Slovenia": "#C62828"}

skipped: list = []
docs = load_corpus(DATA, skipped=skipped)
chunks = chunk_docs(docs)
df = pd.DataFrame([{
    "id": d.id, "country": d.country, "law": d.law, "doc_type": d.doc_type, "kb": d.kb,
    "chars": len(d.text), "words": len(d.text.split()), "n_cited": len(d.articles_cited),
    "proceeding": d.proceeding, **{f"x_{k}": v for k, v in d.extra.items()},
} for d in docs])

md = ["# EDA – Contest_Data", ""]

# 1. counts -----------------------------------------------------------------
counts = df.pivot_table(index="country", columns=["kb", "law"], values="id", aggfunc="count", fill_value=0)
md += ["## 1. Documents per country × knowledge base × law", "",
       counts.to_markdown(), "",
       f"Total loaded: **{len(df)}** documents; skipped: **{len(skipped)}**; "
       f"after chunking: **{len(chunks)}** chunks.", ""]
for path, reason in skipped:
    md.append(f"- skipped `{Path(path).name}`: {reason}")
md.append("")

ax = df.groupby(["kb", "country"]).size().unstack().plot(
    kind="bar", color=[COLORS[c] for c in ["Estonia", "Italy", "Slovenia"]], figsize=(7, 4))
ax.set_title("Documents per knowledge base and country")
ax.set_xlabel(""); ax.set_ylabel("documents"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(OUT / "01_counts.png", dpi=150); plt.close()

# 2. lengths ----------------------------------------------------------------
lens = df.groupby(["doc_type", "country"])["chars"].describe()[["count", "min", "50%", "max"]]
long_docs = (df["chars"] > 1500).groupby(df["doc_type"]).sum()
md += ["## 2. Text length (characters, after cleaning)", "", lens.round(0).to_markdown(), "",
       f"Documents longer than 1,500 characters (split into chunks): "
       f"codes {int(long_docs.get('code', 0))}, cases {int(long_docs.get('case', 0))}.", ""]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, dt in zip(axes, ["code", "case"]):
    for c in COLORS:
        s = df[(df.doc_type == dt) & (df.country == c)]["chars"]
        ax.hist(s, bins=30, alpha=0.55, color=COLORS[c], label=c)
    ax.axvline(1500, ls="--", c="k", lw=1)
    ax.set_title(f"{'Civil code articles' if dt == 'code' else 'Past cases'}: length")
    ax.set_xlabel("characters"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "02_lengths.png", dpi=150); plt.close()

# 3. case metadata coverage -------------------------------------------------
cases = df[df.doc_type == "case"]
xcols = [c for c in cases.columns if c.startswith("x_")]
cov = (cases.groupby("country")[xcols].apply(lambda g: g.notna().mean() * 100)
       .T.round(0).sort_values("Italy", ascending=False))
cov.index = [i[2:] for i in cov.index]
proc = cases.groupby(["country", "law", "proceeding"]).size().unstack(fill_value=0)
md += ["## 3. Past cases: metadata coverage (% of cases with a value)", "", cov.to_markdown(), "",
       "Cases per law and proceeding type:", "", proc.to_markdown(), ""]

ax = cov.plot(kind="barh", color=[COLORS[c] for c in cov.columns], figsize=(7, 5))
ax.set_title("Past cases: % with metadata field"); ax.set_xlabel("%"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(OUT / "03_case_metadata.png", dpi=150); plt.close()

# 4. article numbers: text header vs civil_codes_used ------------------------
mismatch = []
for f in sorted(DATA.rglob("*.json")):
    if "cases" in f.parent.name.lower():
        continue
    raw = json.load(open(f, encoding="utf-8"))
    meta_nums = _ARTICLE_REF.findall(str(raw.get("metadata", {}).get("civil_codes_used", "")))
    d = next((x for x in docs if x.source == str(f)), None)
    if d and d.article and meta_nums and d.article.split("-")[0] != meta_nums[0]:
        mismatch.append((d.country, d.kb, f.name, d.article, meta_nums[0]))
codes_n = int((df.doc_type == "code").sum())
md += ["## 4. Article number: text header vs `civil_codes_used`", "",
       f"In **{len(mismatch)} / {codes_n}** code articles the number in the first line of the text "
       "differs from `civil_codes_used`: the metadata holds an article *referenced* in the text. "
       "The loader uses the header as `article` and keeps the metadata in `articles_cited`.", "",
       "| country | kb | file | article (text) | civil_codes_used |", "|---|---|---|---|---|"]
md += [f"| {c} | {k} | {fn} | {a} | {m} |" for c, k, fn, a, m in mismatch[:15]]
md.append("")

# 5. most cited articles in cases --------------------------------------------
md.append("## 5. Most cited articles in past cases")
md.append("")
for c in COLORS:
    cnt = Counter(a for d in docs if d.doc_type == "case" and d.country == c for a in d.articles_cited)
    md.append(f"- **{c}**: " + ", ".join(f"{a} ({n})" for a, n in cnt.most_common(8)))
md.append("")

(OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:12]))
print(f"\nSaved: {OUT / 'summary.md'} and 3 PNG charts")
