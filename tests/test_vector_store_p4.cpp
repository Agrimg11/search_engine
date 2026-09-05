/// @file test_vector_store_p4.cpp
/// @brief Phase 4 additions to VectorStore tests.
///
/// Tests the new multi-chunk aggregation behaviour: multiple embeddings
/// for the same doc_id should collapse into one result, returning only
/// the best-scoring chunk.

#include "core/vector_store.hpp"
#include "core/similarity.hpp"

#include <gtest/gtest.h>

using namespace search;

// ── Multi-chunk aggregation ───────────────────────────────────────────────────

TEST(VectorStoreP4, TwoChunksSameDocReturnOneResult) {
    VectorStore vs;
    // Add two chunks for doc 0.
    vs.add(0, {1.0f, 0.0f});   // chunk 0 — points east
    vs.add(0, {0.0f, 1.0f});   // chunk 1 — points north

    // Query pointing east — chunk 0 should win, but we expect only ONE result
    // with doc_id == 0, not two separate entries.
    auto results = vs.search({1.0f, 0.0f}, 5);
    EXPECT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0u);
}

TEST(VectorStoreP4, BestChunkScoreReturned) {
    VectorStore vs;
    // Doc 0 has two chunks: one pointing east, one pointing north.
    vs.add(0, {1.0f, 0.0f});   // chunk 0 — cos sim with east query = 1.0
    vs.add(0, {0.0f, 1.0f});   // chunk 1 — cos sim with east query = 0.0

    // Query pointing east — best chunk for doc 0 should score ~1.0.
    auto results = vs.search({1.0f, 0.0f}, 5);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_NEAR(results[0].score, 1.0, 1e-4);
}

TEST(VectorStoreP4, ChunksFromDifferentDocsKeepSeparate) {
    VectorStore vs;
    // Doc 0: two chunks, both pointing east.
    vs.add(0, {1.0f, 0.0f});
    vs.add(0, {0.9f, 0.1f});
    // Doc 1: one chunk pointing north.
    vs.add(1, {0.0f, 1.0f});

    // Expect 2 unique results (one per doc_id), not 3 chunk-level results.
    auto results = vs.search({1.0f, 0.0f}, 10);
    EXPECT_EQ(results.size(), 2u);

    // The first result should be doc 0 (closer to east query).
    EXPECT_EQ(results[0].doc_id, 0u);
    EXPECT_EQ(results[1].doc_id, 1u);
}

TEST(VectorStoreP4, ManyChunksSameDocStillOneResult) {
    VectorStore vs;
    // Add 10 chunks for doc 42, each pointing in a slightly different direction.
    for (int i = 0; i < 10; ++i) {
        float angle = static_cast<float>(i) * 0.1f;
        vs.add(42, {std::cos(angle), std::sin(angle)});
    }
    // Query pointing along angle = 0 (east).
    auto results = vs.search({1.0f, 0.0f}, 10);
    // Must return exactly 1 result for doc_id 42, not 10.
    EXPECT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 42u);
    // Best chunk (angle = 0.0) has similarity ~1.0.
    EXPECT_NEAR(results[0].score, 1.0, 1e-3);
}

// ── Backward-compat: size() counts chunks not docs ───────────────────────────

TEST(VectorStoreP4, SizeCountsChunksNotDocs) {
    VectorStore vs;
    vs.add(0, {1.0f, 0.0f});
    vs.add(0, {0.0f, 1.0f});  // second chunk of same doc
    vs.add(1, {0.5f, 0.5f});
    // 3 chunks total across 2 docs.
    EXPECT_EQ(vs.size(), 3u);
}
