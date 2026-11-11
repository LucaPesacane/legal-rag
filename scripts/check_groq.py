"""Smoke test: verifies that the Groq provider answers through LLMBackend.

Usage (from the repo root, venv active):
    python scripts/check_groq.py
    python scripts/check_groq.py llama-3.1-8b-instant
"""
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig  # noqa: E402
from backend.llm_provider import LLMBackend  # noqa: E402

load_dotenv()

models = sys.argv[1:] or ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
for model in models:
    cfg = RAGConfig(llm_provider="groq", llm_model_name=model)
    t0 = time.time()
    answer = LLMBackend(cfg).chat(
        "You are a concise legal assistant.",
        "In one sentence: what is a forced share (legitime) in inheritance law?",
    )
    print(f"[{model}] {time.time() - t0:.1f}s -> {answer[:300]}")
