/// @file test_vector_store.cpp
/// @brief Unit tests for VectorStore — brute-force KNN correctness.
///
/// Tests use simple hand-crafted 2-D and 3-D vectors where the expected
/// nearest-neighbour is obvious, making the tests self-verifying without
/// a reference implementation.
///
/// Phase 3.1 tests.

#include "core/vector_store.hpp"
#include "core/similarity.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace search;

// ── Helper: build a normalised unit vector in a given direction ───────────────
static std::vector<float> unit(float x, float y) {
    std::vector<float> v = {x, y};
    l2_normalize(v);
    return v;
}

// ── Basic API ─────────────────────────────────────────────────────────────────

TEST(VectorStore, EmptyStoreReturnsEmpty) {
    VectorStore vs;
    auto results = vs.search({1.0f, 0.0f}, 5);
    EXPECT_TRUE(results.empty());
}

TEST(VectorStore, SizeAfterAdd) {
    VectorStore vs;
    vs.add(0, {1.0f, 0.0f});
    vs.add(1, {0.0f, 1.0f});
    EXPECT_EQ(vs.size(), 2u);
}

TEST(VectorStore, ClearResetsSize) {
    VectorStore vs;
    vs.add(0, {1.0f, 0.0f});
    vs.clear();
    EXPECT_EQ(vs.size(), 0u);
    EXPECT_TRUE(vs.empty());
}

TEST(VectorStore, ZeroVectorRejected) {
    VectorStore vs;
    bool added = vs.add(0, {0.0f, 0.0f});
    EXPECT_FALSE(added);
    EXPECT_EQ(vs.size(), 0u);
}

// ── KNN correctness ───────────────────────────────────────────────────────────

TEST(VectorStore, ClosestDocReturnsFirst) {
    VectorStore vs;
    // Doc 0 points up (north), doc 1 points right (east).
    vs.add(0, {0.0f, 1.0f});   // north
    vs.add(1, {1.0f, 0.0f});   // east

    // Query pointing north — doc 0 should be nearest.
    auto results = vs.search({0.1f, 0.9f}, 2);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].doc_id, 0u);
    EXPECT_GT(results[0].score, results[1].score);
}

TEST(VectorStore, TopKLimitsResults) {
    VectorStore vs;
    for (uint32_t i = 0; i < 10; ++i) {
        vs.add(i, {static_cast<float>(i), 1.0f});
    }
    auto results = vs.search({9.0f, 1.0f}, 3);
    EXPECT_EQ(results.size(), 3u);
}

TEST(VectorStore, IdenticalVectorScoreIsOne) {
    VectorStore vs;
    vs.add(42, {1.0f, 0.0f, 0.0f});
    auto results = vs.search({1.0f, 0.0f, 0.0f}, 1);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].doc_id, 42u);
    EXPECT_NEAR(results[0].score, 1.0, 1e-5);
}

TEST(VectorStore, ResultsDescendingOrder) {
    VectorStore vs;
    vs.add(0, {1.0f, 0.0f});   // cosine with {0.7, 0.7} ≈ 0.707
    vs.add(1, {0.0f, 1.0f});   // cosine with {0.7, 0.7} ≈ 0.707
    vs.add(2, {1.0f, 1.0f});   // cosine with {0.7, 0.7} = 1.0 (same dir)

    auto results = vs.search({1.0f, 1.0f}, 3);
    ASSERT_EQ(results.size(), 3u);
    // Scores should be non-increasing.
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].score, results[i].score);
    }
    // The document pointing in the same direction (doc 2) should be first.
    EXPECT_EQ(results[0].doc_id, 2u);
}

TEST(VectorStore, TopKBiggerThanStoreReturnsAll) {
    VectorStore vs;
    vs.add(0, {1.0f, 0.0f});
    vs.add(1, {0.0f, 1.0f});
    // Requesting K=100 but only 2 docs stored.
    auto results = vs.search({1.0f, 0.5f}, 100);
    EXPECT_EQ(results.size(), 2u);
}

TEST(VectorStore, DimensionAccessor) {
    VectorStore vs;
    EXPECT_EQ(vs.dimension(), 0u);
    vs.add(0, {1.0f, 2.0f, 3.0f});
    EXPECT_EQ(vs.dimension(), 3u);
}
