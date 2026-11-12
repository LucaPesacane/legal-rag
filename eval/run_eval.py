import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.chat_logger import log_turn
from backend.config import RAGConfig
from backend.rag_multiagent import multiagent_answer_question
from backend.rag_single_agent import single_agent_answer_question

# Runs Task A and Task B on the test set and saves one JSON line per answer
# (same format as the chat log) in eval/results/answers_<split>.jsonl.
# Then prints the metrics that need no LLM: routing, hit@5, MRR, citations, latency.
# Usage (from the repo root): python eval/run_eval.py [dev|holdout]

load_dotenv(".env")
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
testset = [q for q in json.load(open("eval/testset.json", encoding="utf-8")) if q["split"] == split]
out = Path(f"eval/results/answers_{split}.jsonl")
out.unlink(missing_ok=True)

config = RAGConfig()
systems = {"single": single_agent_answer_question, "multi": multiagent_answer_question}

rows = []
for q in testset:
    for mode, answer_fn in systems.items():
        start = time.time()
        answer, docs, trace = answer_fn(q["question"], config, show_reasoning=True)
        rec = log_turn(q["question"], answer, docs, config, mode, time.time() - start,
                       session_id=q["id"], trace=trace, path=out)

        ids = [s["id"] for s in rec["sources"]]
        dbs = {s["db"] for s in rec["sources"]}
        rank = next((i + 1 for i, x in enumerate(ids[:5]) if x in q["gold_ids"]), None)
        rows.append({
            "id": q["id"], "category": q["category"], "mode": mode,
            # routing is correct if all the expected DBs were searched (none expected -> no retrieval)
            "routing_ok": set(q["expected_dbs"]) <= dbs if q["expected_dbs"] else not dbs,
            "hit@5": None if not q["gold_ids"] else int(rank is not None),
            "mrr": None if not q["gold_ids"] else (1 / rank if rank else 0.0),
            "n_sources": len(ids),
            "cites": answer.count("[S"),
            "latency_s": rec["latency_s"],
        })
        print(f"{q['id']} {mode:6s} {rec['latency_s']:5.1f}s sources={len(ids)} rank={rank}")

df = pd.DataFrame(rows)
df.to_csv(f"eval/results/retrieval_{split}.csv", index=False)
print("\n", df.groupby("mode")[["routing_ok", "hit@5", "mrr", "n_sources", "latency_s"]].mean().round(3))
print("\n", df.pivot_table(index="category", columns="mode", values="routing_ok", aggfunc="mean").round(2))
