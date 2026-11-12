# backend/ragas_eval.py
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend

# Role of this module:
# RAGAS evaluation of the conversation logs (guide §6): context precision,
# context recall, faithfulness, answer relevancy, answer correctness.
# Used by the RAG Evaluation page and by eval/run_ragas.py.

TESTSET_PATH = Path("eval/testset.json")


def build_ragas_dataset_from_answers(answers_path: Path) -> Dataset:
    """Rows from a test set run (eval/run_eval.py): the ground truth is the reference
    answer of the test set and `chat_id` is "<question id>/<single|multi>"."""
    references = {q["question"]: q["reference"] for q in json.load(open(TESTSET_PATH, encoding="utf-8"))}
    rows = []
    with open(answers_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": references.get(r["question"], ""),
                "chat_id": f"{r['session_id']}/{r['mode']}",
            })
    return Dataset.from_list(rows)


def get_ragas_models(config: RAGConfig):
    """
    LangChain LLM + embeddings for Ragas:
    - Groq judge (config.judge_model_name, a different family from the generator)
      if GROQ_API_KEY is set, otherwise OpenAI gpt-4o-mini;
    - the same embedding model used by the RAG system.
    """
    if os.getenv("GROQ_API_KEY"):
        judge_config = RAGConfig(llm_provider="groq", llm_model_name=config.judge_model_name, temperature=0.0)
        return LLMBackend(judge_config).get_langchain_llm(), get_embedding_model(config)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY or OPENAI_API_KEY in .env before running RAGAS evaluation.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0), OpenAIEmbeddings(model="text-embedding-3-small")


def run_ragas(dataset: Dataset, config: RAGConfig) -> pd.DataFrame:
    """Compute the 5 metrics; returns one row per answer (chat_id kept)."""
    eval_llm, eval_embeddings = get_ragas_models(config)

    # Groq returns one completion per call: answer_relevancy must ask for 1, not 3
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness],
        llm=eval_llm,
        embeddings=eval_embeddings,
        show_progress=True,
        # few parallel calls: the free tier allows 8000 tokens per minute
        run_config=RunConfig(max_workers=2, timeout=300),
    )
    df_scores = result.to_pandas()
    df_scores["chat_id"] = dataset["chat_id"]  # same row order as the input
    return df_scores


def compare_systems(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Performance table Task A vs Task B (guide §7.2): mean of each metric."""
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    system = df_scores["chat_id"].str.split("/").str[1].map(
        {"single": "Task A - single agent", "multi": "Task B - multi-agent"})
    return df_scores.groupby(system)[[m for m in metrics if m in df_scores]].mean().round(3)
