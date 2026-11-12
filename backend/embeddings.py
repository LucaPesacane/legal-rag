from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .config import RAGConfig

# Simple in-memory cache: {model name -> embedding model}
# (loading bge-m3 takes ~10 s: it must happen once, not at every question)
_EMBEDDING_CACHE: dict[str, Embeddings] = {}


def get_embedding_model(config: RAGConfig) -> Embeddings:
    key = f"{config.embedding_provider}:{config.embedding_model_name}"
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = _build_embedding_model(config)
    return _EMBEDDING_CACHE[key]


def _build_embedding_model(config: RAGConfig) -> Embeddings:
    """
    Returns a LangChain Embeddings object based on config.

    - embedding_provider == "openai":
        Uses OpenAIEmbeddings.

    - embedding_provider == "huggingface":
        Uses HuggingFaceEmbeddings with device forced to CPU to avoid
        issues like "Cannot copy out of meta tensor; no data!" on some setups.
    """
    if config.embedding_provider == "openai":
        return OpenAIEmbeddings(model=config.embedding_model_name)

    # Default: Hugging Face embeddings on CPU
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": "cpu"},           # 🔴 force CPU
        encode_kwargs={"normalize_embeddings": True}, # 🔴 normalize embeddings
    )
