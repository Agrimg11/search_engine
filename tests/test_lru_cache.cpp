/// @file test_lru_cache.cpp
/// @brief Phase 2.2 — LRU Cache: correctness and regression tests.
///
/// ═══════════════════════════════════════════════════════════════════════════
/// TESTING STRATEGY
/// ═══════════════════════════════════════════════════════════════════════════
///
/// Four test suites:
///
///   Suite 1  LRUCacheBasic          (7 tests)  — insert, get, update, size, capacity
///   Suite 2  LRUCacheOrdering       (5 tests)  — MRU promotion, correct eviction
///   Suite 3  LRUCacheEdgeCases      (6 tests)  — cap=0, cap=1, repeated ops
///   Suite 4  LRUCacheSearchInteg    (12 tests) — SearchEngine HIT/MISS, K-disambiguation,
///                                                result equality, invalidation
///
/// ── LRU ordering invariant (spec example) ──────────────────────────────────
///   Capacity = 3
///   Insert A, B, C → [C(MRU), B, A(LRU)]
///   Access A       → [A(MRU), C, B(LRU)]
///   Insert D       → evict B → [D(MRU), A, C(LRU)]
///   Searching B    → MISS
/// (Tested in SpecExampleEvictionSequence)
///
/// ── Correctness invariant ───────────────────────────────────────────────────
///   For every query: uncached_result == cached_result
///   • same number of results
///   • same doc_ids
///   • same scores (within 1e-9 tolerance)
///   • same ordering
///
/// ── Thread safety ───────────────────────────────────────────────────────────
///   Project is single-threaded; no mutex tests required.

#include "core/lru_cache.hpp"
#include "engine/search_engine.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using namespace search;

// ═══════════════════════════════════════════════════════════════════════════
// Suite 1 — Basic Operations (7 tests)
// ═══════════════════════════════════════════════════════════════════════════

TEST(LRUCacheBasic, EmptyCacheReturnsMiss) {
    LRUCache<std::string, int> cache(10);

    auto result = cache.get("hello");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(cache.misses(), 1u);
    EXPECT_EQ(cache.hits(),   0u);
}

TEST(LRUCacheBasic, InsertAndRetrieve) {
    LRUCache<std::string, int> cache(10);
    cache.put("hello", 42);

    auto result = cache.get("hello");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
    EXPECT_EQ(cache.hits(),   1u);
    EXPECT_EQ(cache.misses(), 0u);
}

TEST(LRUCacheBasic, RetrieveExistingKeyMultipleTimes) {
    LRUCache<std::string, int> cache(10);
    cache.put("key", 100);

    (void)cache.get("key");
    auto result = cache.get("key");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 100);
    EXPECT_EQ(cache.hits(), 2u);
}

TEST(LRUCacheBasic, MissingKeyReturnsMiss) {
    LRUCache<std::string, int> cache(10);
    cache.put("a", 1);

    auto result = cache.get("b");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(cache.misses(), 1u);
}

TEST(LRUCacheBasic, UpdateExistingKeyOverwritesValue) {
    LRUCache<std::string, int> cache(10);
    cache.put("key", 1);
    cache.put("key", 99);   // update

    auto result = cache.get("key");

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 99);
    EXPECT_EQ(cache.size(), 1u);   // still only one entry
}

TEST(LRUCacheBasic, CacheSizeTrackedCorrectly) {
    LRUCache<std::string, int> cache(10);
    EXPECT_EQ(cache.size(), 0u);
    cache.put("a", 1); EXPECT_EQ(cache.size(), 1u);
    cache.put("b", 2); EXPECT_EQ(cache.size(), 2u);
    cache.put("c", 3); EXPECT_EQ(cache.size(), 3u);
}

TEST(LRUCacheBasic, ConfiguredCapacityIsReported) {
    LRUCache<std::string, int> c7(7);
    EXPECT_EQ(c7.capacity(), 7u);

    LRUCache<std::string, int> c1(1);
    EXPECT_EQ(c1.capacity(), 1u);

    LRUCache<std::string, int> c0(0);
    EXPECT_EQ(c0.capacity(), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite 2 — LRU Ordering (5 tests)
// ═══════════════════════════════════════════════════════════════════════════

TEST(LRUCacheOrdering, CorrectLRUItemEvictedOnOverflow) {
    // cap=3: insert A, B, C → [C(MRU), B, A(LRU)]; insert D → evict A
    LRUCache<std::string, int> cache(3);
    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);

    cache.put("D", 4);

    EXPECT_FALSE(cache.contains("A"));   // A was LRU → evicted
    EXPECT_TRUE(cache.contains("B"));
    EXPECT_TRUE(cache.contains("C"));
    EXPECT_TRUE(cache.contains("D"));
    EXPECT_EQ(cache.size(), 3u);
}

TEST(LRUCacheOrdering, SizeRemainsAtCapacityAfterEviction) {
    LRUCache<std::string, int> cache(3);
    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);
    cache.put("D", 4);   // triggers eviction
    EXPECT_EQ(cache.size(), 3u);   // still exactly capacity
}

TEST(LRUCacheOrdering, MultipleEvictionsRespectOrder) {
    // cap=2: A→[A]; B→[B,A]; C→evict A→[C,B]; D→evict B→[D,C]
    LRUCache<std::string, int> cache(2);
    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);
    cache.put("D", 4);

    EXPECT_FALSE(cache.contains("A"));
    EXPECT_FALSE(cache.contains("B"));
    EXPECT_TRUE(cache.contains("C"));
    EXPECT_TRUE(cache.contains("D"));
}

TEST(LRUCacheOrdering, AccessingItemPromotesToMRU) {
    // cap=3: insert A,B,C → [C,B,A]; get(A) → [A,C,B]; insert D → evict B
    LRUCache<std::string, int> cache(3);
    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);

    (void)cache.get("A");   // A becomes MRU → order: [A, C, B]

    cache.put("D", 4);   // B is LRU → evicted

    EXPECT_FALSE(cache.contains("B"));
    EXPECT_TRUE(cache.contains("A"));
    EXPECT_TRUE(cache.contains("C"));
    EXPECT_TRUE(cache.contains("D"));
}

TEST(LRUCacheOrdering, SpecExampleEvictionSequence) {
    // Exact scenario from the specification:
    //   Capacity = 3
    //   Searches:  A, B, C  → [C, B, A]
    //   Search A             → [A, C, B]
    //   Search D             → evict B → [D, A, C]
    //   Searching B must be a cache MISS.
    LRUCache<std::string, int> cache(3);
    cache.put("A", 1);
    cache.put("B", 2);
    cache.put("C", 3);

    (void)cache.get("A");   // A → MRU; order: [A, C, B]

    cache.put("D", 4);   // evict LRU (B)

    EXPECT_FALSE(cache.contains("B"));   // B was evicted
    EXPECT_TRUE(cache.contains("A"));
    EXPECT_TRUE(cache.contains("C"));
    EXPECT_TRUE(cache.contains("D"));

    // Searching B is a MISS.
    EXPECT_FALSE(cache.get("B").has_value());
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite 3 — Edge Cases (6 tests)
// ═══════════════════════════════════════════════════════════════════════════

TEST(LRUCacheEdgeCases, CapacityZeroNeverCrashes) {
    LRUCache<std::string, int> cache(0);

    cache.put("key", 42);   // no-op
    EXPECT_EQ(cache.size(), 0u);

    auto result = cache.get("key");
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(cache.misses(), 1u);
    EXPECT_EQ(cache.hits(),   0u);
    EXPECT_FALSE(cache.contains("key"));
}

TEST(LRUCacheEdgeCases, CapacityOneEvictsOnSecondInsert) {
    LRUCache<std::string, int> cache(1);
    cache.put("A", 1);
    EXPECT_TRUE(cache.contains("A"));

    cache.put("B", 2);   // evicts A

    EXPECT_FALSE(cache.contains("A"));
    EXPECT_TRUE(cache.contains("B"));
    EXPECT_EQ(cache.size(), 1u);
}

TEST(LRUCacheEdgeCases, RepeatedInsertionSameKeyKeepsSizeOne) {
    LRUCache<std::string, int> cache(10);
    for (int i = 0; i < 100; ++i) {
        cache.put("key", i);
    }
    EXPECT_EQ(cache.size(), 1u);

    auto result = cache.get("key");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 99);   // last value stored
}

TEST(LRUCacheEdgeCases, RepeatedAccessAlwaysHits) {
    LRUCache<std::string, int> cache(5);
    cache.put("key", 42);

    for (int i = 0; i < 50; ++i) {
        auto result = cache.get("key");
        ASSERT_TRUE(result.has_value()) << "failed on iteration " << i;
        EXPECT_EQ(*result, 42);
    }
    EXPECT_EQ(cache.hits(), 50u);
}

TEST(LRUCacheEdgeCases, EmptyResultVectorCanBeCached) {
    LRUCache<std::string, std::vector<SearchResult>> cache(5);
    cache.put("empty_query", {});

    auto result = cache.get("empty_query");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->empty());
}

TEST(LRUCacheEdgeCases, ClearRemovesEntriesPreservesStats) {
    LRUCache<std::string, int> cache(10);
    cache.put("a", 1);
    cache.put("b", 2);
    (void)cache.get("a");         // 1 hit

    cache.clear();

    EXPECT_EQ(cache.size(),  0u);
    EXPECT_EQ(cache.hits(),  1u);    // stats not reset by clear
    EXPECT_FALSE(cache.contains("a"));
    EXPECT_FALSE(cache.contains("b"));
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite 4 — Search Integration (12 tests)
// ═══════════════════════════════════════════════════════════════════════════
//
// Corpus (same as test_topk_heap):
//   Doc 0 "alpha"   : algorithm(×3) data(×2) structure(×1)
//   Doc 1 "beta"    : algorithm(×1) sorting(×2) complexity(×1)
//   Doc 2 "gamma"   : data(×3) science(×2) statistics(×1)
//   Doc 3 "delta"   : graph(×2) traversal(×2) algorithm(×1)
//   Doc 4 "epsilon" : cooking(×1) pasta(×1) italian(×1)
//
// Engine is created via unique_ptr so BM25Scorer's references remain valid.

class LRUCacheSearchInteg : public ::testing::Test {
protected:
    void SetUp() override {
        engine_ = std::make_unique<SearchEngine>();
        engine_->ingest("alpha",   "algorithm algorithm algorithm data data structure");
        engine_->ingest("beta",    "algorithm sorting sorting complexity");
        engine_->ingest("gamma",   "data data data science science statistics");
        engine_->ingest("delta",   "graph graph traversal traversal algorithm");
        engine_->ingest("epsilon", "cooking pasta italian");
        // After all ingests the cache is empty (ingest() calls clear()).
    }

    std::unique_ptr<SearchEngine> engine_;
};

TEST_F(LRUCacheSearchInteg, FirstQueryIsCacheMiss) {
    EXPECT_EQ(engine_->cache_hits(),   0u);
    EXPECT_EQ(engine_->cache_misses(), 0u);

    (void)engine_->search("algorithm", 5);

    EXPECT_EQ(engine_->cache_hits(),   0u);
    EXPECT_EQ(engine_->cache_misses(), 1u);
}

TEST_F(LRUCacheSearchInteg, RepeatedIdenticalQueryIsCacheHit) {
    (void)engine_->search("algorithm", 5);   // MISS
    (void)engine_->search("algorithm", 5);   // HIT

    EXPECT_EQ(engine_->cache_hits(),   1u);
    EXPECT_EQ(engine_->cache_misses(), 1u);
}

TEST_F(LRUCacheSearchInteg, SameQueryDifferentKProducesSeparateKeys) {
    // ":top 1 algorithm" and ":top 10 algorithm" must never share a cache entry.
    (void)engine_->search("algorithm", 1);    // MISS (key: ["algorithm"], K=1)
    (void)engine_->search("algorithm", 10);   // MISS (key: ["algorithm"], K=10)

    EXPECT_EQ(engine_->cache_hits(),   0u);
    EXPECT_EQ(engine_->cache_misses(), 2u);

    // Second round — both should be HITs.
    (void)engine_->search("algorithm", 1);
    (void)engine_->search("algorithm", 10);

    EXPECT_EQ(engine_->cache_hits(), 2u);
}

TEST_F(LRUCacheSearchInteg, DifferentQueryIsCacheMiss) {
    (void)engine_->search("algorithm", 5);   // MISS
    (void)engine_->search("data",      5);   // MISS (different terms)

    EXPECT_EQ(engine_->cache_hits(),   0u);
    EXPECT_EQ(engine_->cache_misses(), 2u);
}

TEST_F(LRUCacheSearchInteg, CachedResultEqualsOriginalResult) {
    auto first  = engine_->search("algorithm data", 5);   // MISS → compute
    auto second = engine_->search("algorithm data", 5);   // HIT  → cached copy

    EXPECT_EQ(engine_->cache_hits(), 1u);

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i].doc_id, second[i].doc_id)
            << "doc_id mismatch at rank " << i;
        EXPECT_NEAR(first[i].score, second[i].score, 1e-9)
            << "score mismatch at rank " << i;
    }
}

TEST_F(LRUCacheSearchInteg, CacheHitReturnsSameDocIdsAndScores) {
    auto first  = engine_->search("graph traversal", 3);   // MISS
    auto second = engine_->search("graph traversal", 3);   // HIT

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i].doc_id, second[i].doc_id)
            << "doc_id differs at index " << i;
        EXPECT_NEAR(first[i].score, second[i].score, 1e-9)
            << "score differs at index " << i;
    }
}

TEST_F(LRUCacheSearchInteg, BM25NotRunOnCacheHit) {
    // Indirect proof: hit counter increases and result is identical, meaning
    // the cache path was taken (BM25 skipped).
    (void)engine_->search("sorting", 5);         // MISS: BM25 runs
    auto cached = engine_->search("sorting", 5);   // HIT: BM25 skipped

    EXPECT_EQ(engine_->cache_hits(), 1u);
    EXPECT_FALSE(cached.empty());   // valid result from cache
}

TEST_F(LRUCacheSearchInteg, ClearCacheForcesFreshComputation) {
    (void)engine_->search("algorithm", 5);   // MISS
    (void)engine_->search("algorithm", 5);   // HIT
    EXPECT_EQ(engine_->cache_hits(), 1u);

    engine_->clear_cache();            // evict all entries
    EXPECT_EQ(engine_->cache_size(), 0u);

    (void)engine_->search("algorithm", 5);   // MISS again

    // Hits = 1 (clear does not reset stats); misses = 2 total.
    EXPECT_EQ(engine_->cache_hits(),   1u);
    EXPECT_EQ(engine_->cache_misses(), 2u);
    EXPECT_EQ(engine_->cache_size(),   1u);   // one entry re-cached
}

TEST_F(LRUCacheSearchInteg, IngestInvalidatesCachedResults) {
    (void)engine_->search("algorithm", 5);   // MISS → cached
    (void)engine_->search("algorithm", 5);   // HIT
    EXPECT_EQ(engine_->cache_hits(), 1u);

    // Adding a new document changes the index → cached results are stale.
    engine_->ingest("zeta", "algorithm algorithm brand new document");

    EXPECT_EQ(engine_->cache_size(), 0u);   // invalidated

    (void)engine_->search("algorithm", 5);   // MISS (index changed)
    EXPECT_EQ(engine_->cache_misses(), 2u);
}

TEST_F(LRUCacheSearchInteg, EmptyQueryReturnsEmptyResultConsistently) {
    // An all-stop-word query tokenizes to [] → BM25 returns {}.
    auto first  = engine_->search("the a an", 5);
    auto second = engine_->search("the a an", 5);

    EXPECT_TRUE(first.empty());
    EXPECT_TRUE(second.empty());
}

TEST_F(LRUCacheSearchInteg, CacheSizeGrowsWithDistinctQueries) {
    (void)engine_->search("algorithm", 5);
    (void)engine_->search("data",      5);
    (void)engine_->search("graph",     5);

    EXPECT_EQ(engine_->cache_size(),   3u);
    EXPECT_EQ(engine_->cache_misses(), 3u);
    EXPECT_EQ(engine_->cache_hits(),   0u);
}

TEST_F(LRUCacheSearchInteg, QueryKeyNormalisesTermOrder) {
    // BM25 is bag-of-words: "algorithm data" and "data algorithm" give
    // identical results.  With sorted QueryKey they share one cache entry
    // so the second query is a HIT rather than a redundant MISS.
    auto first  = engine_->search("algorithm data", 5);   // MISS
    auto second = engine_->search("data algorithm", 5);   // HIT (same sorted key)

    EXPECT_EQ(engine_->cache_hits(), 1u);   // second was a cache hit

    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_EQ(first[i].doc_id, second[i].doc_id);
        EXPECT_NEAR(first[i].score, second[i].score, 1e-9);
    }
}
