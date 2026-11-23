from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from .config import RAGConfig

# [+] In-memory cache {model name -> model}: bge-m3 is loaded once, not at every question
_EMBEDDING_CACHE: dict[str, Embeddings] = {}


# [+] Cross-encoder that re-ranks the documents retrieved by FAISS.
def get_reranker(config: RAGConfig) -> CrossEncoder:
    key = f"reranker:{config.reranker_model_name}"
    if key not in _EMBEDDING_CACHE:
        # documents truncated to 256 tokens: enough to judge relevance, faster on CPU
        _EMBEDDING_CACHE[key] = CrossEncoder(config.reranker_model_name, device="cpu", max_length=256)
    return _EMBEDDING_CACHE[key]


# [~] Embedding model from the cache.
def get_embedding_model(config: RAGConfig) -> Embeddings:
    key = f"{config.embedding_provider}:{config.embedding_model_name}"
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = _build_embedding_model(config)
    return _EMBEDDING_CACHE[key]


# [~] get_embedding_model where local files are tried first.
def _build_embedding_model(config: RAGConfig) -> Embeddings:

    # Returns a LangChain Embeddings object based on config.
    if config.embedding_provider == "openai":
        return OpenAIEmbeddings(model=config.embedding_model_name)

    # Default: Hugging Face embeddings on CPU
    # [+] local files first: online, a second copy of the weights (~2 GB) would be downloaded
    for local_only in (True, False):
        try:
            return HuggingFaceEmbeddings(
                model_name=config.embedding_model_name,
                model_kwargs={"device": "cpu", "local_files_only": local_only},
                encode_kwargs={"normalize_embeddings": True}, # normalize embeddings
            )
        except OSError:
            continue  # model not downloaded yet
