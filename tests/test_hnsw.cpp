/// @file test_hnsw.cpp
/// @brief Unit tests for HNSWIndex.
///
/// Correctness tests use small, hand-crafted 2-D and 3-D vectors where
/// the expected nearest neighbour is geometrically obvious.
///
/// The recall test compares HNSW results against brute-force on a
/// larger dataset to verify the ANN approximation quality.
///
/// Phase 4 tests.

#include "core/hnsw.hpp"
#include "core/similarity.hpp"  // l2_normalize, dot_product

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace search;

// ── Helper ────────────────────────────────────────────────────────────────────

/// Build a unit vector pointing in direction θ (2-D).
static std::vector<float> unit2d(float theta) {
    return {std::cos(theta), std::sin(theta)};
}

// ── Basic API ─────────────────────────────────────────────────────────────────

TEST(HNSW, EmptySearchReturnsEmpty) {
    HNSWIndex idx;
    auto results = idx.search({1.0f, 0.0f}, 5);
    EXPECT_TRUE(results.empty());
}

TEST(HNSW, SizeAfterInsert) {
    HNSWIndex idx;
    EXPECT_EQ(idx.size(), 0u);
    idx.insert({1.0f, 0.0f});
    EXPECT_EQ(idx.size(), 1u);
    idx.insert({0.0f, 1.0f});
    EXPECT_EQ(idx.size(), 2u);
}

TEST(HNSW, ClearResets) {
    HNSWIndex idx;
    idx.insert({1.0f, 0.0f});
    idx.insert({0.0f, 1.0f});
    idx.clear();
    EXPECT_EQ(idx.size(), 0u);
    EXPECT_TRUE(idx.empty());
    // Search on cleared index must not crash.
    EXPECT_TRUE(idx.search({1.0f, 0.0f}, 3).empty());
}

TEST(HNSW, InsertReturnIncreasesMonotonically) {
    HNSWIndex idx;
    for (std::size_t i = 0; i < 5; ++i) {
        std::vector<float> v = {static_cast<float>(i), 1.0f};
        l2_normalize(v);
        EXPECT_EQ(idx.insert(v), i);
    }
}

// ── Single-node ───────────────────────────────────────────────────────────────

TEST(HNSW, SingleNodeSearch) {
    HNSWIndex idx;
    idx.insert({1.0f, 0.0f});
    auto results = idx.search({1.0f, 0.0f}, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first, 0u);
    EXPECT_NEAR(results[0].second, 1.0f, 1e-5f);
}

// ── Nearest-neighbour correctness ─────────────────────────────────────────────

TEST(HNSW, NearestNeighbourCorrect) {
    // Three vectors pointing in different directions (unit vectors).
    // Query points almost exactly north — node 1 (north) should be #1.
    HNSWIndex idx;
    idx.insert(unit2d(0.0f));          // node 0: east
    idx.insert(unit2d(3.14159f / 2));  // node 1: north
    idx.insert(unit2d(3.14159f));      // node 2: west

    // Query close to north.
    std::vector<float> q = unit2d(3.14159f / 2 + 0.05f);
    auto results = idx.search(q, 3);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].first, 1u);  // north should be closest
}

TEST(HNSW, IdenticalVectorScoreIsOne) {
    HNSWIndex idx;
    idx.insert({1.0f, 0.0f, 0.0f});
    auto results = idx.search({1.0f, 0.0f, 0.0f}, 1);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].first, 0u);
    EXPECT_NEAR(results[0].second, 1.0f, 1e-5f);
}

// ── Top-K behaviour ───────────────────────────────────────────────────────────

TEST(HNSW, TopKLimitsResults) {
    HNSWIndex idx;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> v = {static_cast<float>(i), 1.0f};
        l2_normalize(v);
        idx.insert(v);
    }
    auto results = idx.search({1.0f, 0.0f}, 5);
    EXPECT_LE(results.size(), 5u);
}

TEST(HNSW, TopKBiggerThanStoreReturnsAll) {
    HNSWIndex idx;
    idx.insert({1.0f, 0.0f});
    idx.insert({0.0f, 1.0f});
    auto results = idx.search({1.0f, 0.0f}, 100);
    EXPECT_EQ(results.size(), 2u);
}

// ── Result ordering ───────────────────────────────────────────────────────────

TEST(HNSW, ResultsSortedBySimilarityDescending) {
    HNSWIndex idx;
    idx.insert({1.0f, 0.0f});   // node 0: east
    idx.insert({0.0f, 1.0f});   // node 1: north
    idx.insert({1.0f, 1.0f});   // (not unit, but insert normalises)

    std::vector<float> q = {1.0f, 1.0f};
    l2_normalize(q);
    auto results = idx.search(q, 3);
    ASSERT_GE(results.size(), 1u);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].second, results[i].second);
    }
}

// ── Recall test ───────────────────────────────────────────────────────────────
// Verify HNSW finds the *exact* nearest neighbour on a small dataset where
// brute-force is feasible as a reference.

TEST(HNSW, ExactNearestNeighbourRecall) {
    // Build a dataset of 50 random unit vectors in 4-D.
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int N = 50, D = 4;
    HNSWIndex idx;
    std::vector<std::vector<float>> raw_vecs;

    for (int i = 0; i < N; ++i) {
        std::vector<float> v(D);
        for (auto& x : v) x = dist(gen);
        l2_normalize(v);
        raw_vecs.push_back(v);
        idx.insert(v);
    }

    // Query vector.
    std::vector<float> q(D);
    for (auto& x : q) x = dist(gen);
    l2_normalize(q);

    // Brute-force: compute dot product to every vector.
    std::vector<std::pair<int, float>> bf;
    for (int i = 0; i < N; ++i) {
        float sim = dot_product(q, raw_vecs[i]);
        bf.push_back({i, sim});
    }
    std::sort(bf.begin(), bf.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // HNSW search.
    auto hnsw_results = idx.search(q, 3, 50);
    ASSERT_GE(hnsw_results.size(), 1u);

    // The true nearest neighbour (bf[0]) should appear in HNSW's top-3.
    int true_nn = bf[0].first;
    bool found  = false;
    for (const auto& [node_idx, sim] : hnsw_results) {
        if (static_cast<int>(node_idx) == true_nn) { found = true; break; }
    }
    EXPECT_TRUE(found) << "HNSW missed the true nearest neighbour (node " << true_nn << ")";
}
