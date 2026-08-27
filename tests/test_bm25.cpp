/// @file test_bm25.cpp
/// @brief Unit tests for the BM25Scorer class.

#include "core/bm25.hpp"
#include "core/document.hpp"
#include "core/inverted_index.hpp"
#include <gtest/gtest.h>

using namespace search;

// ── Fixture ─────────────────────────────────────────────────────────────────
// Three-document micro-corpus designed so relevance ranking is predictable.
//
//   Doc 0 ("search_heavy"):  search(×2) engine(×2) optimization(×1)   → 5 tokens
//   Doc 1 ("search_light"):  web(×1) search(×1) basics(×1) intro(×1)  → 4 tokens
//   Doc 2 ("unrelated"):     cooking(×1) recipes(×1) pasta(×1) italian(×1) → 4 tokens
//
// avgdl = (5+4+4)/3 ≈ 4.33

class BM25Test : public ::testing::Test {
protected:
    DocumentStore store;
    InvertedIndex index;

    void SetUp() override {
        std::vector<std::string> t0 = {"search", "engine", "search", "engine", "optimization"};
        auto id0 = store.add_document("search_heavy", "search engine search engine optimization");
        store.update_token_count(id0, static_cast<uint32_t>(t0.size()));
        index.add_document(id0, t0);

        std::vector<std::string> t1 = {"web", "search", "basics", "intro"};
        auto id1 = store.add_document("search_light", "web search basics intro");
        store.update_token_count(id1, static_cast<uint32_t>(t1.size()));
        index.add_document(id1, t1);

        std::vector<std::string> t2 = {"cooking", "recipes", "pasta", "italian"};
        auto id2 = store.add_document("unrelated", "cooking recipes pasta italian");
        store.update_token_count(id2, static_cast<uint32_t>(t2.size()));
        index.add_document(id2, t2);
    }
};

// ── Ranking tests ───────────────────────────────────────────────────────────

TEST_F(BM25Test, HigherTFScoresHigher) {
    BM25Scorer scorer(index, store);

    double s0 = scorer.score_document({"search", "engine"}, 0);
    double s1 = scorer.score_document({"search", "engine"}, 1);
    // Doc 0 has "search"×2 and "engine"×2 → must outscore Doc 1.
    EXPECT_GT(s0, s1);
}

TEST_F(BM25Test, IrrelevantDocScoresZero) {
    BM25Scorer scorer(index, store);
    double s2 = scorer.score_document({"search", "engine"}, 2);
    EXPECT_DOUBLE_EQ(s2, 0.0);
}

TEST_F(BM25Test, SearchReturnsSortedResults) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"search"}, 10);

    ASSERT_GE(results.size(), 2u);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].score, results[i].score);
    }
}

TEST_F(BM25Test, TopKTruncatesResults) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"search"}, 1);
    EXPECT_EQ(results.size(), 1u);
}

// ── Edge cases ──────────────────────────────────────────────────────────────

TEST_F(BM25Test, EmptyQueryReturnsNothing) {
    BM25Scorer scorer(index, store);
    EXPECT_TRUE(scorer.search({}, 10).empty());
}

TEST_F(BM25Test, NonexistentTermReturnsNothing) {
    BM25Scorer scorer(index, store);
    EXPECT_TRUE(scorer.search({"xyzzy"}, 10).empty());
}

TEST_F(BM25Test, ScoresArePositive) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"search", "engine"}, 10);
    for (const auto& r : results) {
        EXPECT_GT(r.score, 0.0);
    }
}
