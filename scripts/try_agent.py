import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.rag_multiagent import multiagent_answer_question
from backend.rag_single_agent import single_agent_answer_question

# Runs the single agent (Task A) or the multi-agent system (Task B) on a few
# questions and prints the reasoning trace, the answer and the sources.
# Usage:
#   python scripts/try_agent.py ["question" ...]           # Task A
#   python scripts/try_agent.py --multi ["question" ...]   # Task B

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

args = sys.argv[1:]
multi = "--multi" in args
args = [a for a in args if a != "--multi"]
answer_fn = multiagent_answer_question if multi else single_agent_answer_question

QUESTIONS = args or [
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
    answer, docs, trace = answer_fn(q, config, show_reasoning=True)
    print("=" * 100)
    print(f"[{'Task B' if multi else 'Task A'}] Q: {q}   ({time.time() - start:.1f}s)\n")
    print(trace, "\n")
    print("ANSWER:", answer, "\n")
    for i, d in enumerate(docs, start=1):
        print(f"  [S{i}] {d.metadata['score']:.2f} {d.page_content.splitlines()[0]}")
