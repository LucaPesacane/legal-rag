import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.llm_provider import LLMBackend

# Checks that the Groq models answer through LLMBackend.
# Usage (from the repo root): python scripts/check_groq.py [model ...]

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

models = sys.argv[1:] or ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
for model in models:
    cfg = RAGConfig(llm_provider="groq", llm_model_name=model)
    t0 = time.time()
    answer = LLMBackend(cfg).chat(
        "You are a concise legal assistant.",
        "In one sentence: what is a forced share (legitime) in inheritance law?",
    )
    print(f"[{model}] {time.time() - t0:.1f}s -> {answer[:300]}")
