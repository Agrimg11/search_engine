"""
routers/documents.py — Document listing, autocomplete, and analytics endpoints.

GET /documents          → list all documents currently indexed in the engine
GET /suggest?prefix=... → Trie-based autocomplete using the engine's existing Trie
GET /analytics          → top queries by search count (engine-tracked analytics)

All three endpoints read from the in-process SearchEngine without modifying state.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Query

from models import (
    DocumentInfo, DocumentListResponse,
    SuggestItem, SuggestResponse,
    AnalyticsItem, AnalyticsResponse,
)

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    """
    Return metadata for every document currently indexed in the engine.

    The DocumentStore supports O(1) iteration; this is non-blocking
    for typical corpus sizes (hundreds of documents).
    """
    engine = request.app.state.engine
    docs = []
    for doc in engine.document_store:
        docs.append(DocumentInfo(
            doc_id      = doc.id,
            title       = doc.title,
            char_count  = len(doc.content),
            token_count = doc.token_count,
        ))
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/suggest", response_model=SuggestResponse)
async def suggest(
    request: Request,
    prefix: str = Query(..., min_length=1, max_length=100),
    limit:  int = Query(default=8, ge=1, le=20),
) -> SuggestResponse:
    """
    Return autocomplete suggestions for the given prefix using the engine's
    built-in Trie index. The Trie is built from indexed vocabulary terms,
    sorted by document frequency (most common terms first).
    """
    engine = request.app.state.engine
    raw = engine.suggest(prefix, limit)
    return SuggestResponse(
        suggestions=[SuggestItem(term=term, score=score) for term, score in raw]
    )


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics(request: Request) -> AnalyticsResponse:
    """
    Return per-query analytics tracked since the last server restart.
    Results are sorted by search_count descending (most-searched first).
    """
    engine = request.app.state.engine
    entries = engine.analytics()
    return AnalyticsResponse(
        queries=[
            AnalyticsItem(
                query         = e.query,
                search_count  = e.search_count,
                total_results = e.total_results,
            )
            for e in entries[:20]   # cap at top 20
        ]
    )
