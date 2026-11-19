import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.rag_multiagent import _choose_agents
from backend.rag_single_agent import _decide
from backend.rag_utils import _get_vector_db_dirs

# Routing only (no retrieval, no answer): reasoning effort of the router model.
# Usage (from the repo root): python eval/compare_router.py [dev|holdout] [repeats]

load_dotenv(".env")
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 2
testset = [q for q in json.load(open("eval/testset.json", encoding="utf-8")) if q["split"] == split]
base = RAGConfig()
db_map = _get_vector_db_dirs(base)

rows = []
for effort in ["", "low"]:
    config = replace(base, router_reasoning_effort=effort)
    for r in range(repeats):
        for q in testset:
            for mode in ["single", "multi"]:
                start = time.time()
                if mode == "single":
                    need, dbs, *_ = _decide(q["question"], db_map, config)
                    dbs = set(dbs) if need else set()
                else:
                    dbs = set(_choose_agents(q["question"], db_map, config)[0])
                ok = set(q["expected_dbs"]) <= dbs if q["expected_dbs"] else not dbs
                rows.append({"id": q["id"], "category": q["category"], "mode": mode, "effort": effort or "default",
                             "routing_ok": ok, "n_dbs": len(dbs), "seconds": round(time.time() - start, 2)})

df = pd.DataFrame(rows)
print(df.groupby(["effort", "mode"])[["routing_ok", "n_dbs", "seconds"]].mean().round(3))
print(df[~df.routing_ok][["effort", "mode", "id", "category"]].value_counts().to_string())
df.to_csv(f"eval/results/compare_router_effort_{split}.csv", index=False)
