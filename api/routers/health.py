"""
routers/health.py — Health and stats endpoints.

GET /health  → quick liveness + capability check
GET /stats   → engine metadata

Updated for the pure-Python engine:
  - engine_running is always True (in-process, no subprocess to die)
  - doc_count/chunk_count/semantic_available are read directly from the
    SearchEngine object — no dummy search needed
  - engine_pid reports the FastAPI process PID (not a subprocess PID)
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Request

from models import HealthResponse, StatsResponse
import config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Returns the health status of all system components.

    - engine_running: always True (Python engine runs in-process)
    - ollama_reachable: Ollama HTTP server responds to GET /api/tags
    - semantic_available: embedder is set and Ollama is reachable
    """
    engine = request.app.state.engine

    # Check Ollama reachability with a lightweight HTTP GET request.
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as http:
            resp = await http.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    semantic_available = ollama_ok and engine.embedder_available

    status = "ok" if ollama_ok else "degraded"
    return HealthResponse(
        status             = status,
        engine_running     = True,                        # always — in-process
        ollama_reachable   = ollama_ok,
        docs_indexed       = engine.document_count,
        chunks_indexed     = engine.embedded_chunk_count,
        semantic_available = semantic_available,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    """Returns static metadata about the running engine configuration."""
    engine = request.app.state.engine
    return StatsResponse(
        doc_count   = engine.document_count,
        chunk_count = engine.embedded_chunk_count,
        engine_pid  = os.getpid(),           # FastAPI process (no subprocess)
        llm_model   = config.LLM_MODEL,
        embed_model = config.EMBED_MODEL,
    )
