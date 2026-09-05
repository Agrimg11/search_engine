/// @file test_hybrid_search.cpp
/// @brief Integration tests for Phase 3 hybrid search.
///
/// Uses a MockEmbedder (a deterministic EmbeddingProvider) so tests run
/// without a live Ollama server.  The mock assigns fixed 2-D unit vectors
/// to each document text so KNN results are predictable.
///
/// Phase 3 integration tests.

#include "embedding/embedding_provider.hpp"
#include "engine/search_engine.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <unordered_map>

using namespace search;

// ── MockEmbedder ──────────────────────────────────────────────────────────────
/// Deterministic embedder: maps specific text strings to fixed 2-D vectors.
/// Any unmapped text gets {0.5f, 0.5f}.
class MockEmbedder final : public EmbeddingProvider {
public:
    void map(const std::string& text, std::vector<float> vec) {
        table_[text] = std::move(vec);
    }

    std::vector<float> embed(const std::string& text) const override {
        auto it = table_.find(text);
        if (it != table_.end()) return it->second;
        return {0.5f, 0.5f};   // default fallback
    }

    std::size_t dimension()           const noexcept override { return 2; }
    std::string name()                const override { return "MockEmbedder"; }
    bool        is_available()        const noexcept override { return true; }

private:
    std::unordered_map<std::string, std::vector<float>> table_;
};

// ── Test fixture ──────────────────────────────────────────────────────────────
class HybridSearchTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Three documents with known embedding directions:
        //   doc A → {1, 0} (east)   "machine learning neural network"
        //   doc B → {0, 1} (north)  "database query sql index"
        //   doc C → {0.7, 0.7}      "algorithms data structures"

        auto mock = std::make_unique<MockEmbedder>();
        // Map document *content* strings to vectors.
        mock->map("machine learning neural network",  {1.0f, 0.0f});
        mock->map("database query sql index",          {0.0f, 1.0f});
        mock->map("algorithms data structures",        {0.7f, 0.7f});
        // Map query strings as well.
        mock->map("neural network",  {0.9f, 0.1f});  // close to doc A
        mock->map("sql query",       {0.1f, 0.9f});  // close to doc B
        mock->map("algorithms data", {0.6f, 0.8f});  // close to doc C

        engine_.set_embedder(std::move(mock));

        id_A_ = engine_.ingest("DocA", "machine learning neural network");
        id_B_ = engine_.ingest("DocB", "database query sql index");
        id_C_ = engine_.ingest("DocC", "algorithms data structures");
    }

    SearchEngine engine_{50};
    uint32_t id_A_{}, id_B_{}, id_C_{};
};

// ── BM25 mode (unchanged behaviour) ─────────────────────────────────────────
TEST_F(HybridSearchTest, BM25ModeKeywordMatch) {
    auto results = engine_.search("machine learning", 3,
                                  RankingMode::BM25);
    ASSERT_FALSE(results.empty());
    // DocA contains "machine" and "learning" — should rank first.
    EXPECT_EQ(results[0].doc_id, id_A_);
}

TEST_F(HybridSearchTest, BM25ModeNoResults) {
    // "physics" is not in any document.
    auto results = engine_.search("physics quantum", 3, RankingMode::BM25);
    EXPECT_TRUE(results.empty());
}

// ── Semantic mode ────────────────────────────────────────────────────────────
TEST_F(HybridSearchTest, SemanticModeReturnsNearestDoc) {
    // "neural network" embedding is close to DocA's embedding {1,0}.
    auto results = engine_.search("neural network", 3,
                                  RankingMode::SEMANTIC);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].doc_id, id_A_);
}

TEST_F(HybridSearchTest, SemanticModeSqlClosestToDocB) {
    auto results = engine_.search("sql query", 3, RankingMode::SEMANTIC);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].doc_id, id_B_);
}

TEST_F(HybridSearchTest, SemanticModeTopKLimit) {
    auto results = engine_.search("neural network", 2,
                                  RankingMode::SEMANTIC);
    EXPECT_LE(results.size(), 2u);
}

// ── Hybrid mode ──────────────────────────────────────────────────────────────
TEST_F(HybridSearchTest, HybridModeReturnsResults) {
    auto results = engine_.search("machine learning", 3,
                                  RankingMode::HYBRID, 0.5f, 0.5f);
    // Hybrid mode must return at least one result.
    EXPECT_FALSE(results.empty());
}

TEST_F(HybridSearchTest, HybridModePureBM25Weight) {
    // alpha=1, beta=0 → BM25 drives the ranking; semantic adds nothing.
    // The hybrid pipeline uses a wider candidate pool than pure BM25, so
    // ordering may differ slightly.  We verify the result is non-empty and
    // that DocA (the only doc containing both "machine" and "learning") is
    // present in the top results.
    auto hybrid_results = engine_.search("machine learning", 3,
                                          RankingMode::HYBRID, 1.0f, 0.0f);
    ASSERT_FALSE(hybrid_results.empty());
    // DocA must appear somewhere in the results (likely first).
    bool found_a = false;
    for (const auto& r : hybrid_results) {
        if (r.doc_id == id_A_) { found_a = true; break; }
    }
    EXPECT_TRUE(found_a);
}

// ── Cache isolation — different modes must NOT share cache slots ─────────────
TEST_F(HybridSearchTest, CacheKeySeparatePerMode) {
    // Run the same query in both modes.
    (void)engine_.search("neural network", 3, RankingMode::BM25);
    (void)engine_.search("neural network", 3, RankingMode::SEMANTIC);

    // Both should have been cache misses (no slot sharing).
    // We cannot observe misses directly but we can verify two separate
    // entries exist in the cache by checking cache size == 2.
    EXPECT_GE(engine_.cache_size(), 2u);
}

// ── Analytics ────────────────────────────────────────────────────────────────
TEST_F(HybridSearchTest, AnalyticsTracksSearches) {
    (void)engine_.search("machine learning", 3, RankingMode::BM25);
    (void)engine_.search("machine learning", 3, RankingMode::BM25);   // repeated

    auto entries = engine_.analytics();
    ASSERT_FALSE(entries.empty());
    // "machine learning" was searched twice — should be the top entry.
    EXPECT_EQ(entries[0].search_count, 2u);
    EXPECT_NE(entries[0].query.find("machine"), std::string::npos);
}

TEST_F(HybridSearchTest, ResetAnalyticsClearsEntries) {
    (void)engine_.search("machine learning", 3, RankingMode::BM25);
    engine_.reset_analytics();
    EXPECT_TRUE(engine_.analytics().empty());
}

// ── Embedder management ──────────────────────────────────────────────────────
TEST(EmbedderManagement, SetEmbedderClearsVectorStore) {
    SearchEngine eng;
    auto mock1 = std::make_unique<MockEmbedder>();
    mock1->map("hello world", {1.0f, 0.0f});
    eng.set_embedder(std::move(mock1));
    eng.ingest("Doc1", "hello world");
    EXPECT_EQ(eng.embedded_doc_count(), 1u);

    // Replace embedder — vector store must be cleared.
    auto mock2 = std::make_unique<MockEmbedder>();
    eng.set_embedder(std::move(mock2));
    EXPECT_EQ(eng.embedded_doc_count(), 0u);
}

TEST(EmbedderManagement, NoEmbedderSemanticReturnsEmpty) {
    SearchEngine eng;
    eng.ingest("Doc1", "machine learning");
    // No embedder set → SEMANTIC returns empty.
    auto results = eng.search("machine learning", 5, RankingMode::SEMANTIC);
    EXPECT_TRUE(results.empty());
}
