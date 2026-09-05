/// @file test_similarity.cpp
/// @brief Unit tests for similarity.hpp — l2_normalize, dot_product, cosine_similarity.
///
/// Phase 3.1 tests.

#include "core/similarity.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace search;

// ── l2_normalize ──────────────────────────────────────────────────────────────

TEST(L2Normalize, UnitVectorAfterNorm) {
    std::vector<float> v = {3.0f, 4.0f};   // |v| = 5
    bool ok = l2_normalize(v);
    ASSERT_TRUE(ok);
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    EXPECT_NEAR(len, 1.0f, 1e-6f);
}

TEST(L2Normalize, AlreadyUnit) {
    std::vector<float> v = {1.0f, 0.0f, 0.0f};
    bool ok = l2_normalize(v);
    ASSERT_TRUE(ok);
    EXPECT_NEAR(v[0], 1.0f, 1e-6f);
    EXPECT_NEAR(v[1], 0.0f, 1e-6f);
}

TEST(L2Normalize, ZeroVectorReturnsFalse) {
    std::vector<float> v = {0.0f, 0.0f, 0.0f};
    bool ok = l2_normalize(v);
    EXPECT_FALSE(ok);
    // Vector should be left unchanged.
    EXPECT_FLOAT_EQ(v[0], 0.0f);
}

TEST(L2Normalize, SingleElement) {
    std::vector<float> v = {5.0f};
    bool ok = l2_normalize(v);
    ASSERT_TRUE(ok);
    EXPECT_NEAR(v[0], 1.0f, 1e-6f);
}

TEST(L2Normalize, NegativeValues) {
    std::vector<float> v = {-3.0f, 4.0f};
    bool ok = l2_normalize(v);
    ASSERT_TRUE(ok);
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    EXPECT_NEAR(len, 1.0f, 1e-6f);
}

// ── dot_product ───────────────────────────────────────────────────────────────

TEST(DotProduct, BasicDot) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_NEAR(dot_product(a, b), 32.0f, 1e-5f);
}

TEST(DotProduct, ZeroDot) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    EXPECT_NEAR(dot_product(a, b), 0.0f, 1e-6f);
}

TEST(DotProduct, UnitVectorSelfDot) {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    EXPECT_NEAR(dot_product(a, a), 1.0f, 1e-6f);
}

TEST(DotProduct, DimensionMismatchThrows) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    EXPECT_THROW(dot_product(a, b), std::invalid_argument);
}

// ── cosine_similarity ─────────────────────────────────────────────────────────

TEST(CosineSimilarity, IdenticalVectors) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    EXPECT_NEAR(cosine_similarity(a, a), 1.0f, 1e-5f);
}

TEST(CosineSimilarity, OrthogonalVectors) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    EXPECT_NEAR(cosine_similarity(a, b), 0.0f, 1e-6f);
}

TEST(CosineSimilarity, OppositeVectors) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    EXPECT_NEAR(cosine_similarity(a, b), -1.0f, 1e-5f);
}

TEST(CosineSimilarity, ZeroVectorReturnsZero) {
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> b = {1.0f, 2.0f};
    EXPECT_FLOAT_EQ(cosine_similarity(a, b), 0.0f);
}

TEST(CosineSimilarity, ScaleInvariant) {
    // Scaling a vector should not change its cosine similarity.
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {2.0f, 4.0f, 6.0f};   // b = 2*a
    EXPECT_NEAR(cosine_similarity(a, b), 1.0f, 1e-5f);
}

TEST(CosineSimilarity, DimensionMismatchThrows) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    EXPECT_THROW(cosine_similarity(a, b), std::invalid_argument);
}

// ── Dot product on normalised vectors == cosine similarity ────────────────────

TEST(NormalisedDotEquivalence, DotEqualsCosineForUnitVectors) {
    std::vector<float> a = {3.0f, 1.0f, 4.0f};
    std::vector<float> b = {1.0f, 5.0f, 9.0f};

    std::vector<float> na = a, nb = b;
    l2_normalize(na);
    l2_normalize(nb);

    float dot_result    = dot_product(na, nb);
    float cosine_result = cosine_similarity(a, b);

    EXPECT_NEAR(dot_result, cosine_result, 1e-5f);
}
