import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.rag_single_agent import _decide, _retrieve
from backend.rag_utils import _get_vector_db_dirs

# Free comparison (no judge LLM) of a retrieval extension: same routing decision,
# retrieval without and with the extension, hit@5 and MRR on the gold documents.
# Usage (from the repo root): python eval/compare_retrieval.py <config_flag> [dev|holdout] [field=value ...]
#   e.g. python eval/compare_retrieval.py filter_cases_by_law
#        python eval/compare_retrieval.py use_rerank dev reranker_model_name=BAAI/bge-reranker-v2-m3

load_dotenv(".env")
flag = sys.argv[1]
split = next((a for a in sys.argv[2:] if "=" not in a), "dev")
overrides = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)  # other config fields
testset = [q for q in json.load(open("eval/testset.json", encoding="utf-8"))
           if q["split"] == split and q["gold_ids"]]

base = RAGConfig(**overrides)
db_map = _get_vector_db_dirs(base)
# warm-up outside the timing: models are loaded once here
_retrieve("warm up", list(db_map)[:1], db_map, replace(base, **{flag: True}))

rows = []
for q in testset:
    need, dbs, query, law, _ = _decide(q["question"], db_map, base)  # one routing call per question
    for name, config in [("without", base), ("with", replace(base, **{flag: True}))]:
        start = time.time()
        docs = _retrieve(query, dbs, db_map, config, law)
        seconds = time.time() - start
        ids = [d.metadata["id"] for d in docs][:5]
        hit1 = int(bool(ids) and ids[0] in q["gold_ids"])
        rank = next((i + 1 for i, x in enumerate(ids) if x in q["gold_ids"]), None)
        # noise: share of retrieved past cases of the same legal area as the question
        cases = [d for d in docs if d.metadata["doc_type"] == "case"]
        same_law = sum(d.metadata["law"] == law for d in cases) / len(cases) if cases and law else None
        rows.append({"id": q["id"], "category": q["category"], "law": law, flag: name,
                     "hit@1": hit1, "hit@5": int(rank is not None), "mrr": 1 / rank if rank else 0.0,
                     "same_law_cases": same_law, "seconds": round(seconds, 2)})

df = pd.DataFrame(rows)
print(df.pivot_table(index=["category"], columns=flag, values=["hit@1", "hit@5", "mrr", "seconds"]).round(3))
print("\nall questions:\n", df.groupby(flag)[["hit@1", "hit@5", "mrr", "same_law_cases", "seconds"]].mean().round(3))
suffix = "".join(f"_{v.split('/')[-1]}" for v in overrides.values())
df.to_csv(f"eval/results/compare_{flag}{suffix}_{split}.csv", index=False)
