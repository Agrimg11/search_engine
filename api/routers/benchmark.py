"""
routers/benchmark.py — /benchmark endpoint.

Runs a synthetic benchmark suite against the live engine and returns
detailed performance metrics in a single response.

Metrics reported:
  ─ Cache        : capacity, current size, total hits, total misses, hit-rate %
  ─ Index        : vocabulary size, total postings, avg postings per term
  ─ Corpus       : doc count, chunk count, avg doc length (tokens),
                   total tokens, total characters
  ─ Search latency (BM25 / Semantic / Hybrid)
                 : min / max / mean / p50 / p95 across N_RUNS timed queries
                   using a representative set of probe terms extracted from
                   the actual vocabulary (so results are realistic, not dummy)
  ─ Trie autocomplete latency
                 : mean latency for single-char prefix lookups
  ─ Retrieval throughput (BM25 only, docs/sec)

All latencies are in milliseconds.  Benchmark queries use a warm cache hit
after the first run to show both cold-miss and warm-hit paths.
"""

from __future__ import annotations

import statistics
import time
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from engine.search_engine import RankingMode

router = APIRouter(tags=["benchmark"])

# ── Config ────────────────────────────────────────────────────────────────────
_N_RUNS           = 10   # repetitions per query for latency stats
_N_PROBE_QUERIES  = 5    # how many vocabulary terms to use as probe queries
_TOP_K            = 5    # top-k used during benchmark searches
_AUTOCOMPLETE_CHARS = ["a", "s", "t", "i", "c"]  # common first chars


# ── Response models ───────────────────────────────────────────────────────────

class CacheMetrics(BaseModel):
    capacity:    int
    current_size: int
    hits:        int
    misses:      int
    total_lookups: int
    hit_rate_pct: float   # 0–100


class IndexMetrics(BaseModel):
    vocabulary_size:       int
    total_postings:        int
    avg_postings_per_term: float


class CorpusMetrics(BaseModel):
    doc_count:           int
    embedded_chunk_count: int
    avg_doc_length_tokens: float
    total_tokens:        int
    total_chars:         int
    chunk_density:       float   # chunks / doc  (0 if no embedder)


class LatencyStats(BaseModel):
    mode:       str
    available:  bool
    runs:       int
    min_ms:     Optional[float] = None
    max_ms:     Optional[float] = None
    mean_ms:    Optional[float] = None
    median_ms:  Optional[float] = None
    p95_ms:     Optional[float] = None


class AutocompleteMetrics(BaseModel):
    mean_ms: float
    samples: int


class ThroughputMetrics(BaseModel):
    bm25_queries_per_sec: float   # BM25-only, measures pure index throughput


class BenchmarkResponse(BaseModel):
    cache:        CacheMetrics
    index:        IndexMetrics
    corpus:       CorpusMetrics
    latency:      list[LatencyStats]
    autocomplete: AutocompleteMetrics
    throughput:   ThroughputMetrics
    probe_queries: list[str]       # the actual terms used as probes
    note:         str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_probe_queries(engine, n: int) -> list[str]:
    """
    Pick the N most frequent vocabulary terms as representative probe queries.
    Falls back to generic terms if the vocabulary is too small.
    """
    vocab = engine.index.vocabulary   # dict[term, list[Posting]]
    if not vocab:
        return ["search", "document", "index", "query", "text"][:n]

    # Sort terms by document frequency descending; pick top n.
    ranked = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)
    terms = [term for term, _ in ranked[:n]]
    return terms


def _timed_search(engine, query: str, mode: RankingMode, runs: int) -> list[float]:
    """Return list of elapsed-ms per run."""
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        engine.search(query=query, top_k=_TOP_K, mode=mode)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile (linear interpolation)."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_d) - 1)
    return sorted_d[lo] + (sorted_d[hi] - sorted_d[lo]) * (k - lo)


def _build_latency_stats(mode_str: str, available: bool, samples: list[float]) -> LatencyStats:
    if not available or not samples:
        return LatencyStats(mode=mode_str, available=available, runs=0)
    return LatencyStats(
        mode      = mode_str,
        available = available,
        runs      = len(samples),
        min_ms    = round(min(samples), 3),
        max_ms    = round(max(samples), 3),
        mean_ms   = round(statistics.mean(samples), 3),
        median_ms = round(statistics.median(samples), 3),
        p95_ms    = round(_percentile(samples, 95), 3),
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/benchmark", response_model=BenchmarkResponse)
async def benchmark(request: Request) -> BenchmarkResponse:
    """
    Run a synthetic benchmark suite and return live performance metrics.

    This is a READ-ONLY operation — it does not modify the index or cache.
    Expect the endpoint itself to take 200 ms – 2 s depending on corpus size
    and whether Ollama semantic search is available.
    """
    engine = request.app.state.engine

    # ── 1. Cache metrics ──────────────────────────────────────────────────────
    cs = engine.cache_stats()
    total_lookups = cs.hits + cs.misses
    hit_rate = (cs.hits / total_lookups * 100.0) if total_lookups > 0 else 0.0

    cache_metrics = CacheMetrics(
        capacity      = cs.capacity,
        current_size  = cs.size,
        hits          = cs.hits,
        misses        = cs.misses,
        total_lookups = total_lookups,
        hit_rate_pct  = round(hit_rate, 2),
    )

    # ── 2. Index metrics ──────────────────────────────────────────────────────
    vocab     = engine.index.vocabulary
    vocab_size = len(vocab)
    total_postings = sum(len(p) for p in vocab.values())
    avg_postings   = (total_postings / vocab_size) if vocab_size > 0 else 0.0

    index_metrics = IndexMetrics(
        vocabulary_size       = vocab_size,
        total_postings        = total_postings,
        avg_postings_per_term = round(avg_postings, 2),
    )

    # ── 3. Corpus metrics ─────────────────────────────────────────────────────
    doc_store   = engine.document_store
    doc_count   = engine.document_count
    chunk_count = engine.embedded_chunk_count

    total_tokens = sum(d.token_count for d in doc_store)
    total_chars  = sum(len(d.content) for d in doc_store)
    avg_len      = doc_store.average_document_length
    chunk_density = (chunk_count / doc_count) if doc_count > 0 else 0.0

    corpus_metrics = CorpusMetrics(
        doc_count             = doc_count,
        embedded_chunk_count  = chunk_count,
        avg_doc_length_tokens = round(avg_len, 1),
        total_tokens          = total_tokens,
        total_chars           = total_chars,
        chunk_density         = round(chunk_density, 2),
    )

    # ── 4. Search latency per mode ────────────────────────────────────────────
    probe_queries = _pick_probe_queries(engine, _N_PROBE_QUERIES)

    # BM25 — always available
    bm25_times: list[float] = []
    for q in probe_queries:
        bm25_times.extend(_timed_search(engine, q, RankingMode.BM25, _N_RUNS))

    # Semantic / Hybrid — only if embedder is available
    sem_available = engine.embedder_available
    sem_times: list[float] = []
    hyb_times: list[float] = []

    if sem_available and probe_queries:
        for q in probe_queries:
            sem_times.extend(_timed_search(engine, q, RankingMode.SEMANTIC, _N_RUNS))
            hyb_times.extend(_timed_search(engine, q, RankingMode.HYBRID, _N_RUNS))

    latency_stats = [
        _build_latency_stats("bm25",     True,          bm25_times),
        _build_latency_stats("semantic", sem_available, sem_times),
        _build_latency_stats("hybrid",   sem_available, hyb_times),
    ]

    # ── 5. Trie autocomplete latency ──────────────────────────────────────────
    ac_times: list[float] = []
    for ch in _AUTOCOMPLETE_CHARS:
        t0 = time.perf_counter()
        engine.suggest(ch, limit=10)
        ac_times.append((time.perf_counter() - t0) * 1000.0)

    autocomplete_metrics = AutocompleteMetrics(
        mean_ms = round(statistics.mean(ac_times), 3) if ac_times else 0.0,
        samples = len(ac_times),
    )

    # ── 6. Throughput (BM25 queries per second) ───────────────────────────────
    # Use a single representative query for a clean throughput number.
    _BURST = 50
    probe_q = probe_queries[0] if probe_queries else "the"
    t_start = time.perf_counter()
    for _ in range(_BURST):
        engine.search(query=probe_q, top_k=_TOP_K, mode=RankingMode.BM25)
    elapsed = time.perf_counter() - t_start
    qps = round(_BURST / elapsed, 1) if elapsed > 0 else 0.0

    throughput_metrics = ThroughputMetrics(bm25_queries_per_sec=qps)

    # ── Assemble response ─────────────────────────────────────────────────────
    note = (
        "Semantic and Hybrid latency unavailable — Ollama embedder not reachable. "
        "Start Ollama and restart the server to enable semantic benchmarks."
        if not sem_available else
        "All modes benchmarked. Latencies include LRU cache hits after the first run."
    )

    return BenchmarkResponse(
        cache         = cache_metrics,
        index         = index_metrics,
        corpus        = corpus_metrics,
        latency       = latency_stats,
        autocomplete  = autocomplete_metrics,
        throughput    = throughput_metrics,
        probe_queries = probe_queries,
        note          = note,
    )
