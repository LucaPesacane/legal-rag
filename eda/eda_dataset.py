import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.document_loader import article_number, clean_text, load_documents_from_folders

# Exploratory analysis of the corpus: counts, lengths, case metadata coverage
# and the article-number check. Run from the repo root:
#   python eda/eda_dataset.py
# Output (tables + charts for report and slides) in eda/output/

DATA = Path("data/Contest_Data")
OUT = Path("eda/output")
OUT.mkdir(parents=True, exist_ok=True)

docs = load_documents_from_folders([str(DATA)])
df = pd.DataFrame([d.metadata for d in docs])
first = df[df["chunk"] == 0]  # one row per original document
# length of the whole cleaned text, read from the original file
first = first.assign(chars=[len(clean_text(json.load(open(s, encoding="utf-8"))["content"]))
                            for s in first["source"]])

report = ["# EDA - Contest_Data", ""]

# 1. Documents per country and knowledge base
counts = pd.crosstab(first["country"], [first["kb"], first["law"]])
report += ["## Documents per country and knowledge base", counts.to_markdown(), "",
           f"Documents: {len(first)} - chunks after splitting: {len(df)}", ""]
pd.crosstab(first["kb"], first["country"]).plot(kind="bar", rot=0, title="Documents per knowledge base")
plt.tight_layout()
plt.savefig(OUT / "01_counts.png", dpi=150)
plt.close()

# 2. Text length (characters)
lengths = first.groupby(["doc_type", "country"])["chars"].describe()[["min", "50%", "max"]]
report += ["## Text length (characters)", lengths.round(0).to_markdown(), "",
           f"Documents longer than 1500 characters (split): {(first['chars'] > 1500).sum()}", ""]
first.boxplot(column="chars", by=["doc_type", "country"], rot=45)
plt.suptitle("")
plt.title("Text length by type and country")
plt.tight_layout()
plt.savefig(OUT / "02_lengths.png", dpi=150)
plt.close()

# 3. Past cases: % of cases with each metadata field
cases = first[first["doc_type"] == "case"]
fields = ["duration", "cost", "subject_of_succession", "disputed_issues", "succession_type",
          "number_of_persons_involved", "marital_regime", "financial_support", "presence_of_children"]
# values like "Not specified" count as missing
missing = ["not specified", "unspecified", "none", ""]
coverage = cases.groupby("country")[fields].apply(
    lambda g: (g.notna() & ~g.astype(str).apply(lambda c: c.str.lower().isin(missing))).mean() * 100).T.round(0)
report += ["## Past cases: % with metadata field", coverage.to_markdown(), ""]

# 4. Article number in the text vs civil_codes_used
mismatch = 0
codes = 0
for f in DATA.rglob("*.json"):
    if "cases" in f.parent.name.lower():
        continue
    data = json.load(open(f, encoding="utf-8"))
    own = article_number(clean_text(data["content"]))
    meta = str(data["metadata"].get("civil_codes_used", ""))
    codes += 1
    if own and own.split("-")[0].split(".")[0] not in meta:
        mismatch += 1
report += ["## Article number: text vs civil_codes_used",
           f"In {mismatch} of {codes} code articles civil_codes_used is an article cited in the text, "
           "not the article itself: the loader reads the number from the text.", ""]

(OUT / "summary.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
