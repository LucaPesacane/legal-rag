import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.embeddings import get_embedding_model
from backend.vector_store import search_vector_stores

# Quick retrieval check with hand-written questions (no LLM involved).
# For each question we know the right country, knowledge base and, when
# possible, the exact document (gold). We compare:
#   - routed:   search only the right DBs (what the agents will do)
#   - unrouted: search all 9 DBs
# Usage: python eval/sanity_retrieval.py [db_root]   (default: vector_store)

QUESTIONS = [
    ("When does intestate succession apply in Estonia?",
     ["estonia_inheritance_code"], ["Estonia/inheritance_code/Article_10"]),
    ("Can an Estonian spouse revoke a joint will with the other spouse while both are alive?",
     ["estonia_inheritance_code"], ["Estonia/inheritance_code/Article_§ 89"]),
    ("In Estonia, what may a spouse do when administering the joint property alone?",
     ["estonia_divorce_code"], ["Estonia/divorce_code/Article_§ 28"]),
    ("What are children born outside marriage who cannot be recognized entitled to in Italy?",
     ["italy_inheritance_code"], ["Italy/inheritance_code/Article_279"]),
    ("Can an Italian forced heir claim against third parties who bought real estate from a donee?",
     ["italy_inheritance_code"], ["Italy/inheritance_code/Article_561"]),
    ("How is community property divided between spouses under Italian law?",
     ["italy_divorce_code"], ["Italy/divorce_code/Article_194"]),
    ("What is the statutory matrimonial property regime in Italy?",
     ["italy_divorce_code"], ["Italy/divorce_code/Article_ARTICLE 159"]),
    ("On what legal grounds can a person inherit in Slovenia?",
     ["slovenia_inheritance_code"], ["Slovenia/inheritance_code/Article_7"]),
    ("Are small customary gifts counted in the compulsory share in Slovenia?",
     ["slovenia_inheritance_code"], ["Slovenia/inheritance_code/Article_55"]),
    ("Italian appeal court cases on dividing inherited assets among relatives with an expert appointed",
     ["italy_cases"], ["Italy/cases/1 - C.A. Salerno 29.12.2022, n. 1790"]),
    ("Estonian court cases on dividing joint property and debts after divorce",
     ["estonia_cases"], ["Estonia/cases/Processed 2-12-44594_modified",
                         "Estonia/cases/Processed 2-12-44594 (1)_modified"]),
    ("Compare the forced share of heirs in Italy and Slovenia",
     ["italy_inheritance_code", "slovenia_inheritance_code"], []),
]

root = sys.argv[1] if len(sys.argv) > 1 else "vector_store"
config = RAGConfig()
config.embedding_model_name = Path(root, "embedding_model.txt").read_text().strip()
embedding_model = get_embedding_model(config)
all_dbs = [str(p) for p in sorted(Path(root).iterdir()) if p.is_dir()]

rows = []
for question, right_dbs, gold in QUESTIONS:
    for mode, dbs in [("routed", [str(Path(root, n)) for n in right_dbs]), ("unrouted", all_dbs)]:
        docs = search_vector_stores(question, dbs, embedding_model, k_fetch=20, top_k=5, min_sim=0.0)
        found = [d.metadata["db_name"] for d in docs]
        ids = [d.metadata["id"] for d in docs]
        rank = next((i + 1 for i, x in enumerate(ids) if x in gold), None)
        rows.append({
            "question": question[:45], "mode": mode,
            "right_db@5": sum(f in right_dbs for f in found) / len(found),  # share of top-5 from the right DBs
            "hit@5": None if not gold else int(rank is not None),
            "rr": None if not gold else (1 / rank if rank else 0.0),         # reciprocal rank
            "top_score": docs[0].metadata["score"],
        })

df = pd.DataFrame(rows)
print(f"Embedding model: {config.embedding_model_name}\n")
print(df.pivot(index="question", columns="mode", values=["right_db@5", "hit@5"]).to_string())
summary = df.groupby("mode")[["right_db@5", "hit@5", "rr", "top_score"]].mean().round(3)
summary = summary.rename(columns={"rr": "MRR"})
print("\nAverages:\n", summary.to_string())

out = Path("eval/results")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"sanity_{Path(root).name}.csv", index=False)
