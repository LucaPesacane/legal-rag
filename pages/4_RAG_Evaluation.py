# pages/4_RAG_Evaluation.py

import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from datasets import Dataset

from backend.config import RAGConfig
from backend.ragas_eval import build_ragas_dataset_from_answers, compare_systems, run_ragas

CHAT_DB_PATH = Path("chat_sessions.json")
# answers of Task A and Task B on the test set, written by eval/run_eval.py
ANSWERS_PATH = Path("eval/results/answers_dev.jsonl")


# ---------------------------------------------------------------------
# Helpers: config + chat DB
# ---------------------------------------------------------------------
def get_config() -> RAGConfig:
    if "config" not in st.session_state:
        st.session_state.config = RAGConfig()
    return st.session_state.config


def load_chat_db() -> List[Dict[str, Any]]:
    if CHAT_DB_PATH.exists():
        try:
            with open(CHAT_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error reading `{CHAT_DB_PATH}`: {e}")
            return []
    return []


# ---------------------------------------------------------------------
# Build RAGAS dataset from chat_sessions.json
# ---------------------------------------------------------------------
def _extract_contexts_from_assistant_msg(msg: Dict[str, Any]) -> List[str]:
    """
    Extract 'contexts' from a stored assistant message.

    We assume that when answering, you stored retrieved sources like:

        {
          "role": "assistant",
          "content": "...",
          "sources": [
             {"page_content": "...", "metadata": {...}},
             ...
          ]
        }

    If no sources are present, we return [].
    """
    contexts: List[str] = []

    sources = msg.get("sources") or msg.get("contexts") or []
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict) and "page_content" in item:
                contexts.append(str(item["page_content"]))
            else:
                contexts.append(str(item))

    return contexts


def build_ragas_dataset_from_chats(chat_db: List[Dict[str, Any]]) -> Dataset:
    """
    Flatten chat_sessions.json into a RAGAS-compatible Dataset.

    We create rows with:
      - question
      - answer
      - contexts (list of strings)
      - ground_truth (string; initially empty, user can edit it in the UI)
      - chat_id (for traceability)
    """
    rows: List[Dict[str, Any]] = []

    for chat in chat_db:
        chat_id = chat.get("id")
        history = chat.get("history", [])

        last_user_msg: str | None = None

        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                last_user_msg = content

            elif role == "assistant" and last_user_msg:
                answer = content
                contexts = _extract_contexts_from_assistant_msg(msg)

                rows.append(
                    {
                        "question": last_user_msg,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": "",
                        "chat_id": chat_id,
                    }
                )

                last_user_msg = None

    if not rows:
        return Dataset.from_list([])

    return Dataset.from_list(rows)



# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("📊 RAG Evaluation (Ragas)")

config = get_config()

st.write("Evaluate your RAG chatbot using Ragas metrics based on `chat_sessions.json`.")

# ---- Metric help / explanations ----
with st.expander("ℹ️ What do these metrics mean?"):
    st.write(
        "- **context_precision**: Of all the retrieved context, how much is actually relevant to the answer? "
        "High precision = little noise in retrieved chunks."
    )
    st.write(
        "- **context_recall**: Of all the information needed to answer, how much is present in the retrieved "
        "context? High recall = the retriever brought in most of what was needed."
    )
    st.write(
        "- **faithfulness**: To what extent is the answer supported by the retrieved context (and not hallucinated)? "
        "High faithfulness = the answer sticks closely to the documents."
    )
    st.write(
        "- **answer_relevancy**: How well does the answer address the question itself (regardless of ground truth)? "
        "High relevancy = the answer is on-topic for the user query."
    )
    st.write(
        "- **answer_correctness**: How close is the answer to the reference `ground_truth` you provide? "
        "High correctness = the model’s answer matches your gold label."
    )
    st.caption(
        "All metrics are in [0, 1]. Higher is better. Use them to compare different RAG settings and pipelines."
    )

# Data to evaluate: answers of Task A and Task B on the test set, or saved chats
data_source = st.radio(
    "Data to evaluate",
    [f"Test set answers ({ANSWERS_PATH.name})", f"Saved chats ({CHAT_DB_PATH.name})"],
    horizontal=True,
)
if data_source.startswith("Test set"):
    if not ANSWERS_PATH.exists():
        st.warning(f"`{ANSWERS_PATH}` not found: run `python eval/run_eval.py` first.")
        st.stop()
    dataset = build_ragas_dataset_from_answers(ANSWERS_PATH)
else:
    chat_db = load_chat_db()
    if not chat_db:
        st.warning(
            f"No chat sessions found in `{CHAT_DB_PATH}`. "
            "Go to the Chatbot Q&A page, have some conversations, "
            "and make sure you save chats before evaluating."
        )
        st.stop()
    st.success(f"Loaded {len(chat_db)} saved chat session(s).")
    dataset = build_ragas_dataset_from_chats(chat_db)

num_rows = len(dataset)

if num_rows == 0:
    st.warning(
        "Chat sessions are present, but no (question, answer, contexts) rows "
        "could be built. Make sure your assistant messages store retrieved "
        "sources in the chat history (e.g. in a `sources` field)."
    )
    st.stop()

st.write(f"Built RAGAS dataset with {num_rows} Q&A rows.")

df_full = dataset.to_pandas()

st.write(
    "Edit ground truths in the table below (only the 'ground_truth' column is meant to be changed)."
)

# Editable table for ground_truth
df_edited = st.data_editor(
    df_full,
    num_rows="fixed",
    column_config={
        "question": st.column_config.TextColumn("Question", disabled=True),
        "answer": st.column_config.TextColumn("Answer", disabled=True),
        "contexts": st.column_config.ListColumn("Contexts", disabled=True),
        "ground_truth": st.column_config.TextColumn("Ground truth (optional label)"),
        "chat_id": st.column_config.TextColumn("Chat ID", disabled=True),
    },
    width="stretch",
    key="ragas_gt_editor",
)

# Allow sampling if the dataset is large
max_rows = st.slider(
    "Maximum number of rows to evaluate",
    min_value=5,
    max_value=max(10, len(df_edited)),
    value=min(50, len(df_edited)),
    step=5,
    help="For speed, you can subsample the dataset.",
)

if max_rows < len(df_edited):
    st.info(f"Sampling {max_rows} rows out of {len(df_edited)} total.")
    df_eval = df_edited.head(max_rows)
else:
    df_eval = df_edited

# --------- HARD FIXES: normalize types and drop bad rows ---------
# Ensure all core columns are proper strings
df_eval["question"] = df_eval["question"].fillna("").astype(str)
df_eval["answer"] = df_eval["answer"].fillna("").astype(str)
df_eval["ground_truth"] = df_eval["ground_truth"].fillna("").astype(str)


def _normalize_contexts(x):
    """
    Normalize the 'contexts' field so that each row is always a list[str].

    Handles:
      - list/tuple of items
      - numpy arrays (via .tolist())
      - scalars / strings
      - None / NaN
    """
    # Lists or tuples
    if isinstance(x, (list, tuple)):
        return [str(c) for c in x]

    # Numpy arrays or similar (have .tolist but are not strings)
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        try:
            arr = x.tolist()
            if isinstance(arr, (list, tuple)):
                return [str(c) for c in arr]
            return [str(arr)]
        except Exception:
            return [str(x)]

    # None
    if x is None:
        return []

    # Handle pandas NaN (float) or other scalars
    x_str = str(x).strip()
    if not x_str:
        return []

    return [x_str]


df_eval["contexts"] = df_eval["contexts"].apply(_normalize_contexts)

# Drop rows with empty question or answer (they break answer_relevancy)
df_eval = df_eval[
    (df_eval["question"].str.strip() != "") &
    (df_eval["answer"].str.strip() != "")
]

if df_eval.empty:
    st.warning("After cleaning, no rows with non-empty question & answer remain.")
    st.stop()

# Convert back to HF Dataset for Ragas
dataset_eval = Dataset.from_pandas(df_eval, preserve_index=False)

# ---------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------
if st.button("Run RAGAS evaluation"):
    with st.spinner("Running RAGAS metrics..."):
        try:
            df_scores = run_ragas(dataset_eval, config)
        except Exception as e:
            st.error(f"Error during RAGAS evaluation: {e}")
            st.stop()

    st.success("Evaluation completed.")
    st.subheader("Per-row metric scores")
    st.dataframe(df_scores, width="stretch")

    if df_scores["chat_id"].astype(str).str.contains("/").all():
        st.subheader("Task A vs Task B (mean scores)")
        st.dataframe(compare_systems(df_scores), width="stretch")
    else:
        st.subheader("Aggregated (mean) scores")
        st.dataframe(df_scores.mean(numeric_only=True).round(3), width="stretch")

    out = Path("eval/results/ragas_page.csv")
    df_scores.to_csv(out, index=False)
    st.caption(f"Scores saved in `{out}`")
