"""
main.py — FastAPI RAG Gateway backed by the pure-Python search engine.

Architecture overview:
  - The Python SearchEngine (BM25 + faiss HNSW hybrid) runs in-process.
    No C++ subprocess. No pipes. No engine_client.py.
  - The LangChain RAG pipeline (rag_pipeline.py) wraps the engine and
    a local Ollama LLM to generate grounded answers.
  - FastAPI handles routing, validation, and async concurrency.

Startup sequence:
  1. Instantiate SearchEngine (pure Python, in-memory)
  2. Attach OllamaEmbedder if Ollama is reachable (optional)
  3. Ingest all .txt files from data/ into memory
  4. Instantiate RAGPipeline (connects to Ollama LLM)
  5. Server begins accepting requests

Shutdown:
  - FastAPI lifespan context manager — nothing to close (no subprocess).

Run:
  cd api && uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import pathlib
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure api/ is on the Python path when imported from any CWD.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from engine.search_engine import SearchEngine
from engine.embedder      import OllamaEmbedder
from rag_pipeline          import RAGPipeline
from routers               import health, search, rag, ingest, documents
import config


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of long-lived resources:
      - Python SearchEngine (in-process, no subprocess)
      - RAG pipeline (LangChain + Ollama LLM)
    """
    # ── 1. Instantiate engine ──────────────────────────────────────────────
    print(f"[startup] Initialising Python search engine...")
    engine = SearchEngine(cache_capacity=config.CACHE_CAPACITY)

    # ── 2. Attach embedder if Ollama is running ────────────────────────────
    embedder = OllamaEmbedder(
        model    = config.EMBED_MODEL,
        base_url = config.OLLAMA_BASE_URL,
    )
    if embedder.is_available():
        engine.set_embedder(embedder)
        print(f"[startup] Ollama embedder ready ({config.EMBED_MODEL})")
    else:
        print("[startup] Ollama not reachable — running in BM25-only mode")

    # ── 3. Ingest corpus ───────────────────────────────────────────────────
    print(f"[startup] Ingesting documents from: {config.DATA_DIRECTORY}")
    n = engine.ingest_directory(config.DATA_DIRECTORY)
    print(
        f"[startup] Indexed {n} file(s) — "
        f"{engine.document_count} docs, "
        f"{engine.embedded_chunk_count} embedded chunks"
    )

    # ── 4. RAG pipeline ────────────────────────────────────────────────────
    rag_pipeline = RAGPipeline(engine_client=engine, llm_model=config.LLM_MODEL)
    print(f"[startup] RAG pipeline ready (LLM: {config.LLM_MODEL})")

    app.state.engine = engine
    app.state.rag    = rag_pipeline

    yield  # ← server is running

    print("[shutdown] Engine stopped (in-process — nothing to close).")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Hybrid Search Engine — RAG Gateway",
    description = (
        "REST API wrapping a pure-Python BM25 + faiss-HNSW hybrid search engine "
        "with a LangChain RAG pipeline powered by a local Ollama LLM."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)

# Allow all origins in development (restrict in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(search.router)
app.include_router(rag.router)
app.include_router(ingest.router)
app.include_router(documents.router)

# Serve the frontend — must be mounted AFTER all API routes.
_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")



# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
