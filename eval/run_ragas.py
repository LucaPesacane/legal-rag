import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.ragas_eval import build_ragas_dataset_from_answers, compare_systems, run_ragas

# RAGAS evaluation of a test set run without the Streamlit interface.
# Usage (from the repo root): python eval/run_ragas.py [dev|holdout]
# Output: eval/results/ragas_<split>.csv (one row per answer) + Task A vs Task B table.

load_dotenv(".env")
split = sys.argv[1] if len(sys.argv) > 1 else "dev"

dataset = build_ragas_dataset_from_answers(Path(f"eval/results/answers_{split}.jsonl"))
df_scores = run_ragas(dataset, RAGConfig())
df_scores.to_csv(f"eval/results/ragas_{split}.csv", index=False)
print(compare_systems(df_scores))
