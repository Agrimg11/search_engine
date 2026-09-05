"""
search_engine.py — High-level façade tying together all engine components.

Ports: src/engine/search_engine.hpp / search_engine.cpp

Components orchestrated (all in-process Python, no subprocess):
  DocumentStore   ← document storage + corpus statistics
  Tokenizer       ← text normalisation
  InvertedIndex   ← keyword posting lists
  BM25Scorer      ← Okapi BM25 keyword ranking (exact C++ math)
  Trie            ← prefix autocomplete
  LRUCache        ← O(1) query result caching
  Chunker         ← overlapping text chunk splitting
  VectorStore     ← faiss HNSW embedding index
  OllamaEmbedder  ← optional dense embeddings via local Ollama

Ranking modes:
  BM25     — keyword-only (Phase 1/2 behaviour)
  SEMANTIC — pure cosine similarity via VectorStore
  HYBRID   — α·NormBM25 + β·SemanticScore fusion

Hybrid fusion formula (exactly as in C++):
  FinalScore = α × minmax_normalize(BM25) + β × ((cosine_sim + 1) / 2)

QueryKey for LRU cache encodes (sorted_tokens, top_k, mode) so BM25 and
SEMANTIC queries for the same terms never share a cache slot.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from engine.bm25 import BM25Scorer, DocumentExplanation, SearchResult
from engine.chunker import split_into_chunks, TextChunk
from engine.document import Document, DocumentStore
from engine.embedder import EmbeddingProvider
from engine.inverted_index import InvertedIndex
from engine.lru_cache import LRUCache
from engine.tokenizer import Tokenizer
from engine.trie import Trie
from engine.vector_store import VectorStore


# ── RankingMode ───────────────────────────────────────────────────────────────

class RankingMode(str, Enum):
    BM25     = "bm25"
    SEMANTIC = "semantic"
    HYBRID   = "hybrid"


# ── Cache key ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryKey:
    """
    Encodes: sorted normalised query terms + top_k + RankingMode.

    Including mode prevents BM25-cached entries from being returned for
    SEMANTIC/HYBRID queries with identical tokens — same rationale as C++.
    """
    terms: tuple[str, ...]   # sorted, normalised tokens
    top_k: int
    mode:  RankingMode

    @staticmethod
    def build(tokens: list[str], top_k: int, mode: RankingMode) -> 'QueryKey':
        return QueryKey(terms=tuple(sorted(tokens)), top_k=top_k, mode=mode)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """Metadata stored per HNSW node (one record per embedded chunk)."""
    doc_id:      int
    chunk_index: int


@dataclass
class ChunkResult:
    """A retrieved chunk returned by SearchEngine.search_chunks()."""
    doc_id:      int
    title:       str
    chunk_text:  str
    chunk_index: int
    score:       float


@dataclass
class QueryAnalytics:
    """Per-query analytics entry tracked across a session."""
    query:         str
    search_count:  int = 0
    total_results: int = 0


@dataclass
class CacheStats:
    capacity: int
    size:     int
    hits:     int
    misses:   int


# ── SearchEngine ──────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Single in-process search engine — drop-in replacement for the C++
    subprocess + SearchEngineClient pipe architecture.

    Args:
        cache_capacity: Maximum LRU cache entries. 0 disables caching.
    """

    def __init__(self, cache_capacity: int = 100) -> None:
        # Core components — declaration order mirrors C++ (scorer refs index/store).
        self._store     = DocumentStore()
        self._tokenizer = Tokenizer()
        self._index     = InvertedIndex()
        self._scorer    = BM25Scorer(self._index, self._store)

        # Phase 3: semantic search.
        self._vector_store = VectorStore()
        self._embedder: Optional[EmbeddingProvider] = None
        self._embedded_docs: int = 0

        # Phase 4: chunk metadata — chunk_records[node_idx] = ChunkRecord.
        self._chunk_records: list[ChunkRecord] = []
        # Per-doc raw chunks for window retrieval — chunks_per_doc[doc_id] = [TextChunk]
        self._chunks_per_doc: dict[int, list[TextChunk]] = {}

        # LRU cache.
        self._cache: LRUCache[QueryKey, list[SearchResult]] = LRUCache(cache_capacity)

        # Trie autocomplete.
        self._trie = Trie()

        # Analytics.
        self._analytics: dict[str, QueryAnalytics] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, title: str, content: str) -> int:
        """
        Ingest a single document. Returns the assigned document ID.
        Clears the LRU cache (index changed; cached results are stale).
        If an embedder is attached, generates and stores chunk embeddings.
        """
        doc_id = self._store.add_document(title, content)

        # Tokenise and index.
        tokens = self._tokenizer.tokenize(content)
        self._index.add_document(doc_id, tokens)
        self._store.update_token_count(doc_id, len(tokens))

        # Update Trie with new vocabulary entries.
        for term in set(tokens):
            df = self._index.document_frequency(term)
            self._trie.insert(term, df)

        # Embed chunks if embedder is available.
        if self._embedder and self._embedder.is_available():
            chunks = split_into_chunks(content)
            self._chunks_per_doc[doc_id] = chunks
            doc_embedded = False
            for chunk in chunks:
                vec = self._embedder.embed(chunk.text)
                if vec:
                    node_idx_before = self._vector_store.size
                    ok = self._vector_store.add(doc_id, vec)
                    if ok:
                        self._chunk_records.append(
                            ChunkRecord(doc_id=doc_id, chunk_index=chunk.chunk_index)
                        )
                        doc_embedded = True
            if doc_embedded:
                self._embedded_docs += 1

        self._cache.clear()
        return doc_id

    def ingest_directory(self, dir_path: str | Path) -> int:
        """
        Ingest every .txt file in `dir_path`.
        Returns the number of files successfully loaded.
        """
        dir_path = Path(dir_path)
        count = 0
        for txt_file in sorted(dir_path.glob('*.txt')):
            try:
                content = txt_file.read_text(encoding='utf-8', errors='replace')
                title   = txt_file.stem
                self.ingest(title, content)
                count += 1
            except OSError:
                pass
        return count

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:  str,
        top_k:  int         = 10,
        mode:   RankingMode = RankingMode.HYBRID,
        alpha:  float       = 0.5,
        beta:   float       = 0.5,
    ) -> list[SearchResult]:
        """
        Run a search and return the top-k results.

        Cache behaviour:
          HIT  → O(K) copy returned, BM25/embedding skipped.
          MISS → full pipeline runs, result stored in cache.
        """
        tokens = self._tokenizer.tokenize(query)
        key    = QueryKey.build(tokens, top_k, mode)

        # Record analytics.
        q_lower = query.lower().strip()
        if q_lower not in self._analytics:
            self._analytics[q_lower] = QueryAnalytics(query=query)

        # Cache lookup.
        cached = self._cache.get(key)
        if cached is not None:
            self._analytics[q_lower].search_count += 1
            self._analytics[q_lower].total_results += len(cached)
            return list(cached)

        # Run the appropriate pipeline.
        if mode == RankingMode.BM25:
            results = self._run_bm25(tokens, top_k)
        elif mode == RankingMode.SEMANTIC:
            results = self._run_semantic(query, top_k)
        else:  # HYBRID
            results = self._run_hybrid(tokens, query, top_k, alpha, beta)

        self._cache.put(key, results)
        self._analytics[q_lower].search_count += 1
        self._analytics[q_lower].total_results += len(results)
        return results

    def search_chunks(
        self,
        query:  str,
        top_k:  int   = 5,
        window: int   = 1,
        alpha:  float = 0.5,
        beta:   float = 0.5,
    ) -> list[ChunkResult]:
        """
        RAG-optimised chunk-level search.

        Returns individual chunks (not whole documents) sorted by relevance.
        Multiple chunks from the same document can appear.

        Requires an embedder; returns [] if none is set.
        """
        if not self._embedder or self._vector_store.empty:
            return []

        vec = self._embedder.embed(query)
        if not vec:
            return []

        raw_hits = self._vector_store.search_raw(vec, top_k)
        results: list[ChunkResult] = []

        for hit in raw_hits:
            rec     = self._chunk_records[hit.node_idx]
            doc     = self._store.get_document(rec.doc_id)
            chunks  = self._chunks_per_doc.get(rec.doc_id, [])

            # Build windowed text: chunk ± window neighbours joined.
            ci      = rec.chunk_index
            lo      = max(0, ci - window)
            hi      = min(len(chunks) - 1, ci + window)
            text    = ' '.join(c.text for c in chunks[lo:hi + 1])

            results.append(ChunkResult(
                doc_id      = rec.doc_id,
                title       = doc.title,
                chunk_text  = text,
                chunk_index = ci,
                score       = hit.score,
            ))

        return results

    def explain(self, query: str, top_k: int = 10) -> list[DocumentExplanation]:
        """Run BM25 and return detailed term-level explanations."""
        tokens = self._tokenizer.tokenize(query)
        return self._scorer.explain(tokens, top_k)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def set_embedder(self, embedder: EmbeddingProvider) -> None:
        """
        Attach an embedding provider. Replaces the current embedder
        and clears the vector store + LRU cache (embeddings from the old
        provider are incompatible).
        """
        self._embedder = embedder
        self._vector_store.clear()
        self._chunk_records.clear()
        self._chunks_per_doc.clear()
        self._embedded_docs = 0
        self._cache.clear()

    @property
    def embedder_available(self) -> bool:
        return bool(self._embedder and self._embedder.is_available())

    @property
    def embedder_name(self) -> str:
        return self._embedder.name() if self._embedder else "(none)"

    @property
    def embedded_doc_count(self) -> int:
        return self._embedded_docs

    @property
    def embedded_chunk_count(self) -> int:
        return self._vector_store.size

    # ── Cache management ──────────────────────────────────────────────────────

    def cache_stats(self) -> CacheStats:
        return CacheStats(
            capacity = self._cache.capacity,
            size     = self._cache.size,
            hits     = self._cache.hits,
            misses   = self._cache.misses,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def set_cache_capacity(self, new_cap: int) -> None:
        self._cache.set_capacity(new_cap)

    # ── Suggest / Autocomplete ────────────────────────────────────────────────

    def suggest(self, prefix: str, limit: int = 10) -> list[tuple[str, int]]:
        return self._trie.autocomplete(prefix, limit)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def analytics(self) -> list[QueryAnalytics]:
        """Return query analytics sorted by search_count descending."""
        return sorted(
            self._analytics.values(),
            key=lambda a: -a.search_count,
        )

    def reset_analytics(self) -> None:
        self._analytics.clear()

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return self._store.size

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    @property
    def index(self) -> InvertedIndex:
        return self._index

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_bm25(self, tokens: list[str], top_k: int) -> list[SearchResult]:
        return self._scorer.search(tokens, top_k)

    def _run_semantic(self, raw_query: str, top_k: int) -> list[SearchResult]:
        if not self._embedder:
            return []
        vec = self._embedder.embed(raw_query)
        if not vec:
            return []
        return self._vector_store.search(vec, top_k)

    def _run_hybrid(
        self,
        tokens:    list[str],
        raw_query: str,
        top_k:     int,
        alpha:     float,
        beta:      float,
    ) -> list[SearchResult]:
        """
        Fuse BM25 and semantic results.

        FinalScore = α × minmax_normalize(BM25) + β × ((cosine_sim + 1) / 2)

        BM25 scores are min-max normalised to [0, 1] before fusion.
        Cosine similarity [-1, 1] is shifted to [0, 1].
        This ensures neither score dominates regardless of α/β values.
        """
        bm25_results = self._run_bm25(tokens, top_k * 2)
        sem_results  = self._run_semantic(raw_query, top_k * 2)

        # If semantic unavailable, fall back to BM25.
        if not sem_results:
            return bm25_results[:top_k]

        # Min-max normalise BM25 scores to [0, 1].
        bm25_scores  = {r.doc_id: r.score for r in bm25_results}
        bm25_vals    = list(bm25_scores.values())
        bm25_min     = min(bm25_vals) if bm25_vals else 0.0
        bm25_max     = max(bm25_vals) if bm25_vals else 1.0
        bm25_range   = bm25_max - bm25_min or 1.0

        def norm_bm25(doc_id: int) -> float:
            return (bm25_scores.get(doc_id, 0.0) - bm25_min) / bm25_range

        # Shift cosine similarity from [-1,1] → [0,1].
        sem_scores = {r.doc_id: (r.score + 1.0) / 2.0 for r in sem_results}

        # Union of candidates from both pipelines.
        all_ids = set(bm25_scores) | set(sem_scores)
        fused: list[SearchResult] = []
        for doc_id in all_ids:
            score = alpha * norm_bm25(doc_id) + beta * sem_scores.get(doc_id, 0.0)
            fused.append(SearchResult(doc_id=doc_id, score=score))

        fused.sort(key=lambda r: -r.score)
        return fused[:top_k]
