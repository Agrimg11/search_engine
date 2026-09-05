"""
test_api.py — pytest test suite for the FastAPI RAG gateway.

Test layers:
  1. Unit tests — Python SearchEngine (replaces old SearchEngineClient tests)
  2. Unit tests — RAGPipeline with mocked engine + mocked LLM
  3. Integration tests — full FastAPI app via httpx.AsyncClient (pure Python,
                         no C++ binary required)

Run:
  cd api && pytest test_api.py -v
"""

from __future__ import annotations

import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest_asyncio

import pytest

# Ensure api/ is on the Python path.
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from models import ChunkResult, DocumentResult, RAGResponse


# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_CHUNK = ChunkResult(
    doc_id=0, title="algorithms", chunk_text="Sorting algorithms...",
    chunk_index=0, score=0.9
)

SAMPLE_DOC = DocumentResult(
    doc_id=0, title="algorithms", content="Sorting algorithms...", score=0.9
)


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Python SearchEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchEngine:
    """Unit tests for the pure-Python SearchEngine."""

    def _make_engine(self):
        from engine.search_engine import SearchEngine, RankingMode
        engine = SearchEngine(cache_capacity=10)
        engine.ingest("algorithms",      "binary search trees are efficient data structures for searching and sorting")
        engine.ingest("data_structures", "hash tables provide O(1) average lookup time in dictionaries and maps")
        engine.ingest("graphs",          "depth first search and breadth first search traverse graphs efficiently")
        return engine, RankingMode

    def test_ingest_increases_document_count(self):
        engine, _ = self._make_engine()
        assert engine.document_count == 3

    def test_bm25_search_returns_results(self):
        engine, RankingMode = self._make_engine()
        results = engine.search("binary search", top_k=3, mode=RankingMode.BM25)
        assert len(results) >= 1
        # "algorithms" doc contains "binary" and "search" — should be top result
        assert results[0].doc_id == 0

    def test_search_respects_top_k(self):
        engine, RankingMode = self._make_engine()
        results = engine.search("search", top_k=2, mode=RankingMode.BM25)
        assert len(results) <= 2

    def test_lru_cache_records_hits(self):
        engine, RankingMode = self._make_engine()
        engine.search("hash tables", top_k=3, mode=RankingMode.BM25)
        engine.search("hash tables", top_k=3, mode=RankingMode.BM25)  # second = cache hit
        assert engine.cache_stats().hits >= 1

    def test_cache_mode_isolation(self):
        """BM25 and SEMANTIC cache slots must not collide."""
        engine, RankingMode = self._make_engine()
        engine.search("search", top_k=3, mode=RankingMode.BM25)
        engine.search("search", top_k=3, mode=RankingMode.SEMANTIC)
        # SEMANTIC hit against BM25 slot would increment hits — verify they don't
        assert engine.cache_stats().misses >= 2

    def test_trie_suggest_returns_completions(self):
        engine, _ = self._make_engine()
        suggestions = engine.suggest("bin", limit=5)
        words = [s[0] for s in suggestions]
        assert "binary" in words

    def test_trie_suggest_sorted_by_df(self):
        engine, _ = self._make_engine()
        sugg = engine.suggest("", limit=50)
        # Sorted by document_frequency descending — all dfs are positive
        dfs = [s[1] for s in sugg]
        assert dfs == sorted(dfs, reverse=True)

    def test_explain_returns_term_breakdown(self):
        engine, _ = self._make_engine()
        explanations = engine.explain("binary search", top_k=3)
        assert len(explanations) >= 1
        assert explanations[0].final_score > 0
        assert len(explanations[0].terms) >= 1

    def test_explain_terms_sum_to_final_score(self):
        engine, _ = self._make_engine()
        exp = engine.explain("binary search", top_k=1)[0]
        total = sum(t.contribution for t in exp.terms)
        assert abs(total - exp.final_score) < 1e-9

    def test_analytics_tracks_queries(self):
        engine, RankingMode = self._make_engine()
        engine.search("hash tables", top_k=3, mode=RankingMode.BM25)
        engine.search("hash tables", top_k=3, mode=RankingMode.BM25)
        engine.search("graphs", top_k=3, mode=RankingMode.BM25)
        analytics = engine.analytics()
        counts = {a.query: a.search_count for a in analytics}
        assert counts.get("hash tables") == 2
        assert counts.get("graphs") == 1

    def test_clear_cache(self):
        engine, RankingMode = self._make_engine()
        engine.search("binary", top_k=3, mode=RankingMode.BM25)
        engine.clear_cache()
        assert engine.cache_stats().size == 0

    def test_search_no_results_for_unknown_term(self):
        engine, RankingMode = self._make_engine()
        results = engine.search("xyzzynotaword", top_k=5, mode=RankingMode.BM25)
        assert results == []

    def test_hybrid_falls_back_to_bm25_without_embedder(self):
        """Without an embedder, HYBRID should return BM25 results."""
        engine, RankingMode = self._make_engine()
        hybrid  = engine.search("binary search", top_k=3, mode=RankingMode.HYBRID)
        bm25    = engine.search("binary search", top_k=3, mode=RankingMode.BM25)
        assert [r.doc_id for r in hybrid] == [r.doc_id for r in bm25]

    def test_ingest_directory_loads_files(self, tmp_path):
        from engine.search_engine import SearchEngine
        (tmp_path / "doc1.txt").write_text("The quick brown fox")
        (tmp_path / "doc2.txt").write_text("Machine learning and neural networks")
        engine = SearchEngine()
        count = engine.ingest_directory(tmp_path)
        assert count == 2
        assert engine.document_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases_input(self):
        from engine.tokenizer import Tokenizer
        tok = Tokenizer()
        tokens = tok.tokenize("Hello WORLD")
        assert all(t == t.lower() for t in tokens)

    def test_removes_punctuation(self):
        from engine.tokenizer import Tokenizer
        tok = Tokenizer()
        tokens = tok.tokenize("Hello, world! How are you?")
        assert all(t.isalnum() for t in tokens)

    def test_removes_stop_words_by_default(self):
        from engine.tokenizer import Tokenizer
        tok = Tokenizer()
        tokens = tok.tokenize("the cat sat on the mat")
        assert "the" not in tokens
        assert "on" not in tokens

    def test_stop_word_removal_can_be_disabled(self):
        from engine.tokenizer import Tokenizer
        tok = Tokenizer()
        tok.enable_stop_word_removal(False)
        tokens = tok.tokenize("the cat")
        assert "the" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# Unit — Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_empty_text_returns_empty(self):
        from engine.chunker import split_into_chunks
        assert split_into_chunks("") == []

    def test_short_text_returns_one_chunk(self):
        from engine.chunker import split_into_chunks
        chunks = split_into_chunks("hello world", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_chunk_index_is_sequential(self):
        from engine.chunker import split_into_chunks
        chunks = split_into_chunks("a" * 200, chunk_size=50, overlap=10)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_overlap_respected(self):
        from engine.chunker import split_into_chunks
        text   = "a" * 1000
        chunks = split_into_chunks(text, chunk_size=100, overlap=20)
        # Step = 80, so chunk[1].start_char should be 80
        assert chunks[1].start_char == 80

    def test_overlap_clamped_when_too_large(self):
        from engine.chunker import split_into_chunks
        # overlap >= chunk_size should not raise, just get clamped
        chunks = split_into_chunks("x" * 200, chunk_size=50, overlap=60)
        assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Unit — LRU Cache
# ─────────────────────────────────────────────────────────────────────────────

class TestLRUCache:
    def test_get_miss_returns_none(self):
        from engine.lru_cache import LRUCache
        cache = LRUCache(capacity=5)
        assert cache.get("missing") is None
        assert cache.misses == 1

    def test_put_and_get(self):
        from engine.lru_cache import LRUCache
        cache = LRUCache(capacity=5)
        cache.put("key", [1, 2, 3])
        assert cache.get("key") == [1, 2, 3]
        assert cache.hits == 1

    def test_capacity_evicts_lru(self):
        from engine.lru_cache import LRUCache
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)   # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_capacity_zero_always_misses(self):
        from engine.lru_cache import LRUCache
        cache = LRUCache(capacity=0)
        cache.put("k", "v")
        assert cache.get("k") is None

    def test_get_keys_mru_first(self):
        from engine.lru_cache import LRUCache
        cache = LRUCache(capacity=3)
        cache.put("a", 1); cache.put("b", 2); cache.put("c", 3)
        cache.get("a")  # promote "a" to MRU
        keys = cache.get_keys()
        assert keys[0] == "a"


# ─────────────────────────────────────────────────────────────────────────────
# Unit — RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Unit tests for the RAG pipeline with mocked engine and LLM."""

    def _make_pipeline(
        self,
        llm_answer: str = "Binary trees store data hierarchically.",
        chunks: list | None = None,
    ):
        from engine.search_engine import ChunkResult as EngineChunkResult
        from rag_pipeline import RAGPipeline

        mock_engine = MagicMock()
        engine_chunks = chunks if chunks is not None else [
            EngineChunkResult(
                doc_id=0, title="algorithms",
                chunk_text="Sorting algorithms...",
                chunk_index=0, score=0.9,
            )
        ]
        mock_engine.search_chunks.return_value = engine_chunks

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._engine    = mock_engine
        pipeline._llm_model = "llama3.2"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = llm_answer
        pipeline._chain = mock_chain
        return pipeline

    def test_run_returns_rag_response(self):
        pipeline = self._make_pipeline()
        response = pipeline.run("What is a binary tree?")
        assert isinstance(response, RAGResponse)
        assert "Binary trees" in response.answer
        assert response.model == "llama3.2"
        assert response.latency_ms >= 0

    def test_run_includes_sources(self):
        pipeline = self._make_pipeline()
        response = pipeline.run("sorting algorithms")
        assert len(response.sources) == 1
        assert response.sources[0].title == "algorithms"

    def test_run_calls_engine_with_correct_params(self):
        pipeline = self._make_pipeline()
        pipeline.run("test query", mode="bm25", top_k=3, alpha=0.7, beta=0.3)
        pipeline._engine.search_chunks.assert_called_once_with(
            "test query", top_k=3, window=1, alpha=0.7, beta=0.3
        )

    def test_run_no_chunks_still_returns_response(self):
        pipeline = self._make_pipeline(chunks=[])
        response = pipeline.run("unknown topic")
        assert isinstance(response, RAGResponse)
        assert len(response.sources) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration — FastAPI endpoints (pure Python, no C++ binary needed)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_engine():
    """Mock SearchEngine for integration tests."""
    from engine.search_engine import SearchResult, CacheStats

    engine = MagicMock()
    engine.document_count        = 5
    engine.embedded_chunk_count  = 30
    engine.embedder_available    = True

    # Simulate document_store.get_document()
    mock_doc = MagicMock()
    mock_doc.title   = "algorithms"
    mock_doc.content = "Sorting algorithms..."
    engine.document_store.get_document.return_value = mock_doc

    def fake_search(query, top_k=5, mode=None, alpha=0.5, beta=0.5):
        return [SearchResult(doc_id=0, score=0.9)]

    engine.search.side_effect = fake_search
    engine.cache_stats.return_value = CacheStats(capacity=100, size=0, hits=0, misses=0)
    return engine


@pytest.fixture
def mock_rag():
    """Mock RAGPipeline for integration tests."""
    pipeline = MagicMock()
    pipeline.run.return_value = RAGResponse(
        answer     = "Binary trees are hierarchical data structures.",
        sources    = [SAMPLE_CHUNK],
        model      = "llama3.2",
        latency_ms = 123.4,
    )
    return pipeline


@pytest_asyncio.fixture
async def async_client(mock_engine, mock_rag):
    """Create an httpx.AsyncClient pointed at the test app."""
    import httpx
    from main import app

    app.state.engine = mock_engine
    app.state.rag    = mock_rag

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_root_returns_200(async_client):
    resp = await async_client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


@pytest.mark.asyncio
async def test_search_valid_query(async_client):
    resp = await async_client.post("/search", json={"query": "binary trees"})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert body["total_results"] == 1


@pytest.mark.asyncio
async def test_search_invalid_empty_query(async_client):
    """Empty query should be rejected by Pydantic (min_length=1)."""
    resp = await async_client.post("/search", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_invalid_top_k(async_client):
    resp = await async_client.post("/search", json={"query": "test", "top_k": 99})
    assert resp.status_code == 422   # Pydantic le=20 validator


@pytest.mark.asyncio
async def test_rag_query_valid(async_client):
    resp = await async_client.post("/rag/query", json={"question": "What is a B-tree?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "model" in body
    assert body["model"] == "llama3.2"


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Health endpoint should always return 200 with required fields."""
    with patch("routers.health.httpx.AsyncClient") as mock_http:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_ctx = MagicMock()
        mock_ctx.get = MagicMock(return_value=mock_resp)
        mock_http.return_value.__aenter__ = MagicMock(return_value=mock_ctx)
        mock_http.return_value.__aexit__  = MagicMock(return_value=False)

        resp = await async_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "engine_running" in body
    assert body["engine_running"] is True   # always True for in-process engine
