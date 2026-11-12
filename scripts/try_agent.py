import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.rag_single_agent import single_agent_answer_question

# Runs the single agent (Task A) on a few questions and prints
# Thought / Action / Observation, the answer and the sources.
# Usage: python scripts/try_agent.py ["your question"]

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

QUESTIONS = sys.argv[1:] or [
    "What is a will?",                                                       # generic: no retrieval
    "Who inherits when there is no will in Estonia?",                        # one country, code
    "Compare the reserved share of children in Italy and Slovenia.",         # comparative
    "Come si dividono i beni della comunione tra coniugi in Italia?",        # Italian question
    "How long do Italian court cases on the division of an estate last?",   # cases + metadata
    "What is the penalty for theft in Italy?",                               # out of domain
]

config = RAGConfig()
for q in QUESTIONS:
    start = time.time()
    answer, docs, trace = single_agent_answer_question(q, config, show_reasoning=True)
    print("=" * 100)
    print(f"Q: {q}   ({time.time() - start:.1f}s)\n")
    print(trace, "\n")
    print("ANSWER:", answer, "\n")
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}] {d.metadata['score']:.2f} {d.page_content.splitlines()[0]}")
