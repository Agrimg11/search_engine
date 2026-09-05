"""
models.py — Pydantic request/response schemas for the RAG gateway API.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class DocumentResult(BaseModel):
    """One retrieved document (legacy / /search endpoint)."""
    doc_id:  int
    title:   str
    content: str     # up to 800 chars
    score:   float

class ChunkResult(BaseModel):
    """One retrieved chunk (RAG-optimised / /rag/query endpoint)."""
    doc_id:      int
    title:       str
    chunk_text:  str   # exact matched chunk ± window neighbours
    chunk_index: int   # position of the chunk within its document
    score:       float


# ── /search ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode:  Literal["bm25", "semantic", "hybrid"] = "hybrid"
    top_k: int   = Field(default=5,   ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta:  float = Field(default=0.5, ge=0.0, le=1.0)

class SearchResponse(BaseModel):
    results:            list[DocumentResult]
    total_results:      int
    semantic_available: bool
    doc_count:          int
    chunk_count:        int


# ── /rag/query ────────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    mode:  Literal["bm25", "semantic", "hybrid"] = "hybrid"
    top_k: int   = Field(default=3, ge=1, le=15,
                         description="Number of *chunks* to retrieve (not documents).")
    window: int  = Field(default=0, ge=0, le=3,
                         description="Number of neighbour chunks to include on each side.")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta:  float = Field(default=0.5, ge=0.0, le=1.0)

class RAGResponse(BaseModel):
    answer:      str
    sources:     list[ChunkResult]   # ← now chunks, not full docs
    model:       str
    latency_ms:  float


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id:      int
    title:       str
    doc_count:   int    # total docs now in the engine
    chunk_count: int    # total embedded chunks now in the engine


# ── /documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id:      int
    title:       str
    char_count:  int    # length of the document content in characters
    token_count: int    # number of tokens as tracked by BM25

class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total:     int


# ── /suggest ──────────────────────────────────────────────────────────────────

class SuggestItem(BaseModel):
    term:  str
    score: int   # document frequency used as relevance signal

class SuggestResponse(BaseModel):
    suggestions: list[SuggestItem]


# ── /analytics ────────────────────────────────────────────────────────────────

class AnalyticsItem(BaseModel):
    query:         str
    search_count:  int
    total_results: int

class AnalyticsResponse(BaseModel):
    queries: list[AnalyticsItem]



# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:             Literal["ok", "degraded"]
    engine_running:     bool
    ollama_reachable:   bool
    docs_indexed:       int
    chunks_indexed:     int
    semantic_available: bool


# ── /stats ────────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    doc_count:    int
    chunk_count:  int
    engine_pid:   Optional[int]
    llm_model:    str
    embed_model:  str
