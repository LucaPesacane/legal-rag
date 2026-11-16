import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.ragas_eval import build_ragas_dataset_from_answers, compare_systems, run_ragas

# RAGAS evaluation of a test set run without the Streamlit interface.
# Usage (from the repo root):
#   python eval/run_ragas.py dev                           # all answers, key GROQ_API_KEY
#   python eval/run_ragas.py dev 1/2 GROQ_API_KEY          # first half with one key ...
#   python eval/run_ragas.py dev 2/2 GROQ_API_KEY_2        # ... second half with another account, in parallel
#   python eval/run_ragas.py dev merge                     # join the halves
# Output: eval/results/ragas_<split>.csv (one row per answer) + Task A vs Task B table.

load_dotenv(".env")
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
out = Path(f"eval/results/ragas_{split}.csv")

if len(sys.argv) > 2 and sys.argv[2] == "merge":
    df_scores = pd.concat(pd.read_csv(p) for p in sorted(out.parent.glob(f"ragas_{split}_part*.csv")))
else:
    part, n_parts = map(int, sys.argv[2].split("/")) if len(sys.argv) > 2 else (1, 1)
    if len(sys.argv) > 3:
        # the free tier limits are per account: another account's key doubles the throughput
        os.environ["GROQ_API_KEY"] = os.getenv(sys.argv[3]).strip()
    dataset = build_ragas_dataset_from_answers(Path(f"eval/results/answers_{split}.jsonl"))
    dataset = dataset.shard(n_parts, part - 1, contiguous=True)
    df_scores = run_ragas(dataset, RAGConfig())
    if n_parts > 1:
        out = out.with_name(f"ragas_{split}_part{part}.csv")

df_scores.to_csv(out, index=False)
print(compare_systems(df_scores))
