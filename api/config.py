"""
config.py — Central configuration for the FastAPI RAG gateway.

All paths are relative to this file's location (api/) so the server
can be started from any working directory.
"""

import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent.resolve()

# Directory containing the .txt corpus files.
DATA_DIRECTORY = str(_HERE.parent / "data")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# Model used by the Python engine for retrieval embeddings.
EMBED_MODEL = "nomic-embed-text"

# Model used by LangChain for answer generation (RAG step).
# Must be pulled: ollama pull llama3.2
LLM_MODEL = "llama3.2"

# ── Search defaults ───────────────────────────────────────────────────────────
DEFAULT_TOP_K = 3
DEFAULT_MODE  = "hybrid"
DEFAULT_ALPHA = 0.3
DEFAULT_BETA  = 0.7

# Maximum chunks passed as context to the LLM.
# Keeps prompt size manageable (each chunk ≤ 800 chars).
MAX_CONTEXT_CHUNKS = 5

# ── Chunker ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800   # Characters per chunk (matches C++ default)
CHUNK_OVERLAP = 200  # Overlap between adjacent chunks

# ── HNSW (faiss) ─────────────────────────────────────────────────────────────
HNSW_M              = 16   # Max bidirectional links per layer
HNSW_EF_CONSTRUCTION = 200  # Beam width during index construction
HNSW_EF_SEARCH      = 50   # Beam width during query

# ── LRU Cache ─────────────────────────────────────────────────────────────────
CACHE_CAPACITY = 100

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
