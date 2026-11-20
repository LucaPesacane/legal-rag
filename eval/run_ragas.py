import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.ragas_eval import METRICS, build_ragas_dataset_from_answers, compare_systems, run_ragas

# [+] RAGAS evaluation of a test set run, without the Streamlit interface.
# Usage (from the repo root):
#   python eval/run_ragas.py dev                           # all answers, key GROQ_API_KEY
#   python eval/run_ragas.py dev 1/2 GROQ_API_KEY          # first half with one key ...
#   python eval/run_ragas.py dev 2/2 GROQ_API_KEY_2        # ... second half with another account, in parallel
#   python eval/run_ragas.py dev merge                     # join the parts (and the recomputed cells)
#   python eval/run_ragas.py dev fill 1/3 GROQ_API_KEY     # recompute only the empty cells (part 1 of 3)
# Output: eval/results/ragas_<split>.csv (one row per answer) + Task A vs Task B table.

load_dotenv(".env")
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
args = sys.argv[2:]
part, n_parts = map(int, next((a for a in args if "/" in a), "1/1").split("/"))
key_var = next((a for a in args if a.startswith("GROQ_API_KEY")), None)
if key_var:
    # the free tier limits are per account: another account's key adds throughput and daily tokens
    os.environ["GROQ_API_KEY"] = os.getenv(key_var).strip()

out = Path(f"eval/results/ragas_{split}.csv")
answers = Path(f"eval/results/answers_{split}.jsonl")

if "merge" in args:
    df_scores = pd.concat([pd.read_csv(p) for p in sorted(out.parent.glob(f"ragas_{split}_part*.csv"))],
                          ignore_index=True)
    # cells recomputed by "fill" replace the empty ones
    for p in sorted(out.parent.glob(f"ragas_{split}_fill*.csv")):
        for r in pd.read_csv(p).dropna().itertuples():
            df_scores.at[r.row, r.metric] = r.value

elif "fill" in args:
    df_scores = pd.read_csv(out)
    dataset = build_ragas_dataset_from_answers(answers)  # same row order as ragas_<split>.csv
    filled = []
    for name in METRICS:
        # answers without sources (generic questions): context recall and faithfulness cannot be computed
        rows = [i for i in df_scores.index[df_scores[name].isna()]
                if dataset[i]["contexts"] or name not in ("context_recall", "faithfulness")]
        rows = rows[part - 1::n_parts]  # share of this part
        if rows:
            scores = run_ragas(dataset.select(rows), RAGConfig(), [name])
            filled.append(pd.DataFrame({"row": rows, "metric": name, "value": scores[name].values}))
    if filled:
        # one file per round, so cells recovered in earlier rounds are kept
        stamp = pd.Timestamp.now().strftime("%m%d_%H%M")
        pd.concat(filled).to_csv(out.with_name(f"ragas_{split}_fill_{stamp}_p{part}.csv"), index=False)
    sys.exit()

else:
    dataset = build_ragas_dataset_from_answers(answers).shard(n_parts, part - 1, contiguous=True)
    df_scores = run_ragas(dataset, RAGConfig())
    if n_parts > 1:
        out = out.with_name(f"ragas_{split}_part{part}.csv")

df_scores.to_csv(out, index=False)
print(compare_systems(df_scores))
