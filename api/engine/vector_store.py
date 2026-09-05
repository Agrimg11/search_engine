"""
vector_store.py — Document embedding store backed by a faiss HNSW index.

Ports: src/core/vector_store.hpp / vector_store.cpp
       src/core/hnsw.hpp / hnsw.cpp  (replaced entirely by faiss.IndexHNSWFlat)

Phase 4 architecture preserved:
  - Documents may be added multiple times — once per text chunk.
  - Each add() call = one HNSW node.
  - search() aggregates by doc_id: only the best-scoring chunk from each
    document is returned (prevents one document with many chunks monopolising
    top-K results).
  - search_raw() returns raw chunk hits (for RAG pipeline which needs the
    exact chunk, not just the document).

HNSW parameters match C++ defaults:
  M              = 16   (max bidirectional links per layer)
  ef_construction= 200  (beam width during build)
  ef_search      = 50   (beam width during query)

All stored vectors are L2-normalised by the caller (SearchEngine).
Under L2 normalisation, cosine_similarity(a,b) = dot_product(a,b), so
faiss inner-product search gives cosine similarity directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import faiss   # faiss-cpu

from engine.bm25 import SearchResult
from engine.similarity import l2_normalize


# ── Chunk hit ─────────────────────────────────────────────────────────────────

@dataclass
class ChunkHit:
    """A raw chunk-level search hit used by the RAG pipeline."""
    node_idx: int    # HNSW node index (maps back to chunk_records_ in SearchEngine)
    doc_id:   int    # Parent document ID
    score:    float  # Cosine similarity in [-1, 1]


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    HNSW-backed embedding store using faiss.IndexHNSWFlat.

    Public API is backward-compatible with the C++ VectorStore:
      add(doc_id, embedding)  — one or more calls per doc are fine.
      search(query, top_k)    — returns one result per unique doc_id.
      search_raw(query, top_k)— returns raw chunk hits.
      clear()
      size / empty / dimension
    """

    # HNSW parameters — match C++ defaults.
    _M:              int = 16
    _EF_CONSTRUCTION: int = 200
    _EF_SEARCH:       int = 50

    def __init__(self) -> None:
        self._index:   faiss.IndexHNSWFlat | None = None
        self._doc_ids: list[int] = []   # doc_ids[node_idx] → doc_id
        self._dim:     int = 0

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add(self, doc_id: int, embedding: list[float]) -> bool:
        """
        Add an embedding for a document (or one of its chunks).

        The embedding is L2-normalised in-place before storage.
        Zero-vectors are silently discarded → returns False.

        Returns True if stored, False if zero-vector was rejected.
        """
        vec = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return False

        vec = vec / norm  # L2 normalise

        # Lazy index creation on first add.
        if self._index is None:
            self._dim   = len(vec)
            # IndexHNSWFlat with inner-product metric.
            # For unit vectors: IP == cosine similarity.
            self._index = faiss.IndexHNSWFlat(self._dim, self._M, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = self._EF_CONSTRUCTION

        self._index.add(vec.reshape(1, -1))
        self._doc_ids.append(doc_id)
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        """
        Return the top-k most similar *documents* for a query embedding.

        Fetches more chunk-level candidates than top_k, aggregates by doc_id
        keeping only the best-scoring chunk per document — same aggregation
        logic as C++ VectorStore::search().

        Results sorted by similarity descending.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        chunks = self.search_raw(query_embedding, top_k=top_k * 4)
        # Aggregate: best score per doc_id.
        best: dict[int, float] = {}
        for hit in chunks:
            if hit.doc_id not in best or hit.score > best[hit.doc_id]:
                best[hit.doc_id] = hit.score

        results = [SearchResult(doc_id=did, score=sc) for did, sc in best.items()]
        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    def search_raw(self, query_embedding: list[float], top_k: int) -> list[ChunkHit]:
        """
        Return raw top-k chunk hits WITHOUT document-level aggregation.

        Used by the RAG pipeline which needs the exact HNSW node index to
        map back to the original chunk text.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # L2-normalise query.
        qvec = np.array(query_embedding, dtype=np.float32)
        qnorm = float(np.linalg.norm(qvec))
        if qnorm == 0.0:
            return []
        qvec = qvec / qnorm

        k = min(top_k, self._index.ntotal)
        self._index.hnsw.efSearch = self._EF_SEARCH

        scores, indices = self._index.search(qvec.reshape(1, -1), k)
        hits: list[ChunkHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            hits.append(ChunkHit(
                node_idx = int(idx),
                doc_id   = self._doc_ids[int(idx)],
                score    = float(score),
            ))
        return hits

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        self._index   = None
        self._doc_ids = []
        self._dim     = 0

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def empty(self) -> bool:
        return self.size == 0

    @property
    def dimension(self) -> int:
        return self._dim
