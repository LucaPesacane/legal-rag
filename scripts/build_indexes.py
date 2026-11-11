import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import RAGConfig
from backend.document_loader import load_documents_from_folders
from backend.embeddings import get_embedding_model
from backend.vector_store import build_all_vector_stores

# Builds the 9 FAISS DBs (country x knowledge base) from data/Contest_Data.
# Usage (from the repo root):
#   python scripts/build_indexes.py                       # model from config
#   python scripts/build_indexes.py BAAI/bge-m3 vector_store_bge

config = RAGConfig()
if len(sys.argv) > 1:
    config.embedding_model_name = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else config.vector_store_root

docs = load_documents_from_folders(["data/Contest_Data"])
start = time.time()
paths = build_all_vector_stores(docs, get_embedding_model(config), root)

# the same embedding model must be used for queries: write it next to the DBs
Path(root, "embedding_model.txt").write_text(config.embedding_model_name)

print(f"{len(docs)} chunks -> {len(paths)} DBs in {root}/ "
      f"({config.embedding_model_name}, {time.time() - start:.0f}s)")
for p in paths:
    print("  ", p)
