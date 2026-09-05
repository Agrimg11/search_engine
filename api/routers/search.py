"""
routers/search.py — Raw search endpoint (no LLM).

POST /search
  Body: SearchRequest
  Returns: SearchResponse (top-K results from Python engine)

This endpoint is useful for:
  - Testing the Python engine independently of the LLM
  - Building UIs that display search results without generating answers
  - Benchmarking retrieval quality

Updated for the pure-Python engine:
  - Calls engine.search() directly — no subprocess, no pipe, no thread executor
  - engine.search() returns typed SearchResult objects; we enrich them with
    document metadata (title, content snippet) from the DocumentStore
  - No is_alive check needed (engine is always alive in-process)
"""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from engine.search_engine import RankingMode
from models import SearchRequest, SearchResponse, DocumentResult
import config

router = APIRouter(tags=["search"])

_SNIPPET_LEN = 800   # characters — matches original C++ behaviour


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    """
    Run the Python hybrid search engine and return ranked document results.

    The Python engine is in-process and non-blocking for typical corpus sizes,
    so no thread-executor offload is needed.
    """
    engine = request.app.state.engine

    try:
        mode = RankingMode(body.mode)
        raw_results = engine.search(
            query = body.query,
            top_k = body.top_k,
            mode  = mode,
            alpha = body.alpha,
            beta  = body.beta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Enrich SearchResult(doc_id, score) with title + content snippet.
    doc_store = engine.document_store
    results: list[DocumentResult] = []
    for r in raw_results:
        try:
            doc = doc_store.get_document(r.doc_id)
            results.append(DocumentResult(
                doc_id  = r.doc_id,
                title   = doc.title,
                content = doc.content[:_SNIPPET_LEN],
                score   = r.score,
            ))
        except IndexError:
            continue

    return SearchResponse(
        results            = results,
        total_results      = len(results),
        semantic_available = engine.embedder_available,
        doc_count          = engine.document_count,
        chunk_count        = engine.embedded_chunk_count,
    )
