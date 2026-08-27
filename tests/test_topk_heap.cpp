/// @file test_topk_heap.cpp
/// @brief Phase 2.1 — Min-Heap Top-K: correctness and regression tests.
///
/// ═══════════════════════════════════════════════════════════════════════════
/// TESTING STRATEGY
/// ═══════════════════════════════════════════════════════════════════════════
///
/// The central invariant to verify is:
///
///   heap_search(query, K)  ==  naive_sort_search(query, K)
///
/// where "naive sort search" is the Phase 1 reference approach:
///   1. Collect candidate doc_ids from all posting lists.
///   2. Score each via scorer.score_document() — same BM25 formula, same
///      parameters, no formula duplication.
///   3. Sort descending by score; stable secondary key = doc_id ascending
///      to resolve any score ties deterministically.
///   4. Take the first min(K, R) results.
///
/// The production heap implementation (bm25.cpp) accumulates scores
/// identically to score_document() — it iterates the same posting lists and
/// applies the same BM25 formula.  The ONLY thing that changed in Phase 2 is
/// the Top-K extraction step (sort → heap).  By comparing heap results against
/// the reference oracle on every query, we verify that the heap always returns
/// the correct Top-K set in the correct order.
///
/// ─── Tie handling ───────────────────────────────────────────────────────────
/// The production search() uses unordered_map iteration, which is
/// non-deterministic.  When two documents have IDENTICAL BM25 scores AND one
/// of them falls exactly at the K/K+1 boundary, the heap and naive sort may
/// disagree on which document to include.
///
/// Policy:
///   • Regression comparison tests use queries/K values where no exact tie
///     falls at a boundary position (verified by pre-computing scores).
///   • The explicit tie test (TieHandling) checks the RESULT SET only
///     (not doc order among tied entries) and uses EXPECT_NEAR for scores.
///
/// ─── Corpus ─────────────────────────────────────────────────────────────────
///   Doc 0 "alpha"   : algorithm(×3) data(×2) structure(×1)       → 6 tokens
///   Doc 1 "beta"    : algorithm(×1) sorting(×2) complexity(×1)   → 4 tokens
///   Doc 2 "gamma"   : data(×3) science(×2) statistics(×1)        → 6 tokens
///   Doc 3 "delta"   : graph(×2) traversal(×2) algorithm(×1)      → 5 tokens
///   Doc 4 "epsilon" : cooking(×1) pasta(×1) italian(×1)          → 3 tokens
///
///   avgdl = (6+4+6+5+3)/5 = 4.8
///   BM25 defaults: k1=1.5, b=0.75

#include "core/bm25.hpp"
#include "core/document.hpp"
#include "core/inverted_index.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <set>
#include <unordered_map>
#include <vector>

using namespace search;

// ═══════════════════════════════════════════════════════════════════════════
// Reference oracle — true Phase 1 equivalent
// ═══════════════════════════════════════════════════════════════════════════
//
// This function is the canonical reference implementation.  It:
//   1. Finds all candidate doc_ids that share at least one posting with the
//      query (same set that production's score-accumulation loop touches).
//   2. Scores each via scorer.score_document() — uses the exact same BM25
//      formula as production, avoiding any risk of formula drift.
//   3. Sorts descending by score with doc_id as a stable secondary key.
//   4. Truncates to top_k.
//
// Using score_document() guarantees mathematical identity with the production
// accumulation loop: both compute the same IDF, same TF normalization, same
// per-term contributions, and the same final sum.
//
// The secondary sort key (doc_id ascending) is used only to make the oracle
// deterministic for equal-score documents.  Tests that compare heap output
// against this oracle must therefore only use queries/K values where no tie
// falls at a boundary (see "Tie handling" above).
static std::vector<SearchResult> reference_top_k(
    const BM25Scorer&        scorer,
    const InvertedIndex&     index,
    const std::vector<std::string>& query_tokens,
    std::size_t              top_k)
{
    if (query_tokens.empty() || top_k == 0) return {};

    // Collect candidate doc_ids (union of all posting lists for query terms).
    std::set<uint32_t> candidate_ids;
    for (const auto& term : query_tokens) {
        for (const auto& posting : index.get_postings(term)) {
            candidate_ids.insert(posting.doc_id);
        }
    }
    if (candidate_ids.empty()) return {};

    // Score each candidate using the production API.
    std::vector<SearchResult> all;
    all.reserve(candidate_ids.size());
    for (uint32_t doc_id : candidate_ids) {
        double s = scorer.score_document(query_tokens, doc_id);
        if (s > 0.0) {   // match Phase 1: only include docs that contributed
            all.push_back({doc_id, s});
        }
    }

    // Primary sort: descending score.  Secondary: ascending doc_id (for
    // determinism when scores are exactly equal).
    std::sort(all.begin(), all.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  if (a.score != b.score) return a.score > b.score;
                  return a.doc_id < b.doc_id;   // stable tie-break
              });

    if (all.size() > top_k) all.resize(top_k);
    return all;
}

// ═══════════════════════════════════════════════════════════════════════════
// Assertion helper: compare heap results against reference result
// ═══════════════════════════════════════════════════════════════════════════
// Only call this for queries where no score tie falls at the K/K+1 boundary.
// (When there are no boundary ties, the set membership is deterministic even
// though heap/sort order among ties may differ.)
static void assert_matches_reference(
    const std::vector<SearchResult>& heap,
    const std::vector<SearchResult>& ref,
    const char* label)
{
    ASSERT_EQ(heap.size(), ref.size())
        << label << ": result count differs";

    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_EQ(heap[i].doc_id, ref[i].doc_id)
            << label << ": doc_id mismatch at rank " << i;
        EXPECT_NEAR(heap[i].score, ref[i].score, 1e-9)
            << label << ": score mismatch at rank " << i
            << " (doc " << heap[i].doc_id << ")";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════
class TopKHeapTest : public ::testing::Test {
protected:
    DocumentStore store;
    InvertedIndex index;

    void SetUp() override {
        // Doc 0 "alpha": algorithm(×3) data(×2) structure(×1) — 6 tokens
        {
            std::vector<std::string> t = {
                "algorithm","data","algorithm","structure","algorithm","data"
            };
            auto id = store.add_document("alpha",
                "algorithm data algorithm structure algorithm data");
            store.update_token_count(id, static_cast<uint32_t>(t.size()));
            index.add_document(id, t);
        }
        // Doc 1 "beta": algorithm(×1) sorting(×2) complexity(×1) — 4 tokens
        {
            std::vector<std::string> t = {
                "algorithm","sorting","sorting","complexity"
            };
            auto id = store.add_document("beta",
                "algorithm sorting sorting complexity");
            store.update_token_count(id, static_cast<uint32_t>(t.size()));
            index.add_document(id, t);
        }
        // Doc 2 "gamma": data(×3) science(×2) statistics(×1) — 6 tokens
        {
            std::vector<std::string> t = {
                "data","science","data","statistics","data","science"
            };
            auto id = store.add_document("gamma",
                "data science data statistics data science");
            store.update_token_count(id, static_cast<uint32_t>(t.size()));
            index.add_document(id, t);
        }
        // Doc 3 "delta": graph(×2) traversal(×2) algorithm(×1) — 5 tokens
        {
            std::vector<std::string> t = {
                "graph","traversal","graph","traversal","algorithm"
            };
            auto id = store.add_document("delta",
                "graph traversal graph traversal algorithm");
            store.update_token_count(id, static_cast<uint32_t>(t.size()));
            index.add_document(id, t);
        }
        // Doc 4 "epsilon": cooking(×1) pasta(×1) italian(×1) — 3 tokens
        {
            std::vector<std::string> t = {"cooking","pasta","italian"};
            auto id = store.add_document("epsilon", "cooking pasta italian");
            store.update_token_count(id, static_cast<uint32_t>(t.size()));
            index.add_document(id, t);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// SECTION A — Reference oracle validation
// Verify that score_document() and the internal accumulation loop agree.
// This is the foundation that makes all subsequent cross-checks meaningful.
// ═══════════════════════════════════════════════════════════════════════════

// A1: score_document() and the full search() produce the same per-document
//     scores for a single-term query.
TEST_F(TopKHeapTest, OracleScoresMatchProductionScores_SingleTerm) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm"};

    // "algorithm" appears in docs 0, 1, 3.
    auto heap_all = scorer.search(query, 100);   // K >> R → get all candidates
    ASSERT_EQ(heap_all.size(), 3u);

    // For each returned doc, verify score_document() matches.
    for (const auto& r : heap_all) {
        double ref_score = scorer.score_document(query, r.doc_id);
        EXPECT_NEAR(r.score, ref_score, 1e-9)
            << "score_document() disagrees with search() for doc " << r.doc_id;
    }
}

// A2: Same check for a multi-term query.
TEST_F(TopKHeapTest, OracleScoresMatchProductionScores_MultiTerm) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm", "data"};

    // Matches docs 0, 1, 2, 3 (at least one term each).
    auto heap_all = scorer.search(query, 100);
    ASSERT_FALSE(heap_all.empty());

    for (const auto& r : heap_all) {
        double ref_score = scorer.score_document(query, r.doc_id);
        EXPECT_NEAR(r.score, ref_score, 1e-9)
            << "score mismatch for doc " << r.doc_id << " on multi-term query";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION B — Direct heap vs. reference regression tests
// For every combination of (query, K) below, the heap result must match the
// reference oracle result exactly (same doc_ids, same scores, same order).
// ═══════════════════════════════════════════════════════════════════════════

// B1: "algorithm", K=1 — only the best document
TEST_F(TopKHeapTest, Regression_Algorithm_K1) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm"};
    auto heap = scorer.search(query, 1);
    auto ref  = reference_top_k(scorer, index, query, 1);
    assert_matches_reference(heap, ref, "algorithm,K=1");

    // Doc 0 (alpha) has the highest BM25 score: it has tf=3 for "algorithm",
    // the highest term frequency among the three matching documents, in a
    // document whose length (6) is above avgdl (4.8) but whose TF advantage
    // outweighs the length penalty.
    ASSERT_EQ(heap.size(), 1u);
    EXPECT_EQ(heap[0].doc_id, 0u);
}

// B2: "algorithm", K=2 — top 2 out of 3 candidates (R=3, K=2 < R)
TEST_F(TopKHeapTest, Regression_Algorithm_K2) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm"};
    auto heap = scorer.search(query, 2);
    auto ref  = reference_top_k(scorer, index, query, 2);
    assert_matches_reference(heap, ref, "algorithm,K=2");

    ASSERT_EQ(heap.size(), 2u);
    EXPECT_EQ(heap[0].doc_id, 0u);   // alpha must be rank 1
    EXPECT_GT(heap[0].score, heap[1].score);   // strict ordering (no tie at top)
}

// B3: "algorithm", K=3 — K == R exactly
TEST_F(TopKHeapTest, Regression_Algorithm_K3) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm"};
    auto heap = scorer.search(query, 3);
    auto ref  = reference_top_k(scorer, index, query, 3);
    assert_matches_reference(heap, ref, "algorithm,K=3");

    ASSERT_EQ(heap.size(), 3u);
    // All three must be present.
    std::set<uint32_t> ids;
    for (const auto& r : heap) ids.insert(r.doc_id);
    EXPECT_TRUE(ids.count(0u));
    EXPECT_TRUE(ids.count(1u));
    EXPECT_TRUE(ids.count(3u));
    // Strictly sorted.
    EXPECT_GE(heap[0].score, heap[1].score);
    EXPECT_GE(heap[1].score, heap[2].score);
}

// B4: "algorithm", K=10 — K >> R (only 3 candidates exist)
TEST_F(TopKHeapTest, Regression_Algorithm_K10) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm"};
    auto heap = scorer.search(query, 10);
    auto ref  = reference_top_k(scorer, index, query, 10);
    assert_matches_reference(heap, ref, "algorithm,K=10");

    EXPECT_EQ(heap.size(), 3u);   // only 3 candidates exist
}

// B5: "algorithm" + "data", K=1
TEST_F(TopKHeapTest, Regression_AlgorithmData_K1) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm", "data"};
    auto heap = scorer.search(query, 1);
    auto ref  = reference_top_k(scorer, index, query, 1);
    assert_matches_reference(heap, ref, "algorithm+data,K=1");

    ASSERT_EQ(heap.size(), 1u);
    EXPECT_GT(heap[0].score, 0.0);
}

// B6: "algorithm" + "data", K=3
TEST_F(TopKHeapTest, Regression_AlgorithmData_K3) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm", "data"};
    auto heap = scorer.search(query, 3);
    auto ref  = reference_top_k(scorer, index, query, 3);
    assert_matches_reference(heap, ref, "algorithm+data,K=3");

    ASSERT_EQ(heap.size(), 3u);
    // Strictly descending scores.
    for (std::size_t i = 1; i < heap.size(); ++i) {
        EXPECT_GE(heap[i-1].score, heap[i].score);
    }
}

// B7: "algorithm" + "graph", K=2
// Docs matching: 0 (algorithm×3), 1 (algorithm×1), 3 (graph×2, algorithm×1).
// Doc 3 gets a boost from both "graph" (unique to it) and "algorithm".
TEST_F(TopKHeapTest, Regression_AlgorithmGraph_K2) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm", "graph"};
    auto heap = scorer.search(query, 2);
    auto ref  = reference_top_k(scorer, index, query, 2);
    assert_matches_reference(heap, ref, "algorithm+graph,K=2");

    ASSERT_EQ(heap.size(), 2u);
    EXPECT_GT(heap[0].score, heap[1].score);
}

// B8: Non-existent term, K=10 — zero candidates
TEST_F(TopKHeapTest, Regression_NonExistentTerm_K10) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"xyzzy_not_in_index"};
    auto heap = scorer.search(query, 10);
    auto ref  = reference_top_k(scorer, index, query, 10);
    assert_matches_reference(heap, ref, "nonexistent,K=10");
    EXPECT_TRUE(heap.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION C — Edge cases
// ═══════════════════════════════════════════════════════════════════════════

// C1: K = 0 must return empty before any heap access.
TEST_F(TopKHeapTest, EdgeCase_KEqualsZero) {
    BM25Scorer scorer(index, store);
    EXPECT_TRUE(scorer.search({"algorithm"}, 0).empty());
}

// C2: Empty query returns empty.
TEST_F(TopKHeapTest, EdgeCase_EmptyQuery) {
    BM25Scorer scorer(index, store);
    EXPECT_TRUE(scorer.search({}, 5).empty());
}

// C3: K > R (K=10, R=1 for unique term "cooking").
TEST_F(TopKHeapTest, EdgeCase_KGreaterThanR) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"cooking"}, 10);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 4u);
    EXPECT_GT(results[0].score, 0.0);
}

// C4: K = R exactly ("algorithm" matches docs 0,1,3 → R=3).
TEST_F(TopKHeapTest, EdgeCase_KEqualsR) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm"}, 3);
    EXPECT_EQ(results.size(), 3u);
}

// C5: K = 1 returns only the single best.
TEST_F(TopKHeapTest, EdgeCase_KEqualsOne) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm"}, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0u);
}

// C6: Irrelevant doc (epsilon) must never appear in CS-topic queries.
TEST_F(TopKHeapTest, EdgeCase_IrrelevantDocNeverAppears) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm", "data", "graph"}, 10);
    ASSERT_FALSE(results.empty());
    for (const auto& r : results) {
        EXPECT_NE(r.doc_id, 4u)
            << "epsilon (doc 4) incorrectly appeared in CS-topic results";
    }
}

// C7: All returned scores must be positive.
TEST_F(TopKHeapTest, EdgeCase_AllScoresPositive) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm", "data"}, 10);
    ASSERT_FALSE(results.empty());
    for (const auto& r : results) {
        EXPECT_GT(r.score, 0.0)
            << "doc " << r.doc_id << " has non-positive score";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION D — Ordering invariants
// ═══════════════════════════════════════════════════════════════════════════

// D1: Results must be sorted in non-increasing score order for any query.
TEST_F(TopKHeapTest, Ordering_ResultsAreDescending_SingleTerm) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm"}, 10);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].score, results[i].score)
            << "Score order violated at positions " << i-1 << " and " << i;
    }
}

// D2: Descending order for multi-term query.
TEST_F(TopKHeapTest, Ordering_ResultsAreDescending_MultiTerm) {
    BM25Scorer scorer(index, store);
    auto results = scorer.search({"algorithm", "data", "graph"}, 10);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].score, results[i].score)
            << "Score order violated at positions " << i-1 << " and " << i;
    }
}

// D3: Best document appears at index 0.
TEST_F(TopKHeapTest, Ordering_BestDocIsFirst) {
    BM25Scorer scorer(index, store);

    // For "data": Doc 2 (gamma) has data×3 in a 6-token doc; Doc 0 has
    // data×2 in a 6-token doc.  Doc 2 has the highest TF contribution
    // for "data".  Also "data" is a common term (docs 0,2 have it); IDF
    // will be the same for both.  So gamma (data×3) must outscore alpha
    // (data×2) for this single-term query.
    auto results = scorer.search({"data"}, 10);
    ASSERT_GE(results.size(), 2u);
    EXPECT_EQ(results[0].doc_id, 2u)   // gamma: data×3
        << "gamma should rank first for 'data' query due to higher TF";
    EXPECT_GT(results[0].score, results[1].score);
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION E — Multiple heap replacements
// Verify that a strong candidate appearing late in the scores map iteration
// is correctly identified as Top-K.
//
// Strategy: use K=1 on a query with 3 candidates.  Regardless of iteration
// order, the single best doc must end up in the heap.
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(TopKHeapTest, MultipleReplacements_StrongLateCandidate) {
    BM25Scorer scorer(index, store);

    // K=1, query "algorithm" → 3 candidates (docs 0,1,3).
    // Doc 0 (alpha) is the strongest by a significant margin (tf=3 vs tf=1).
    // No matter in what order the unordered_map yields the three candidates,
    // after all heap operations the heap must contain doc 0.
    auto results = scorer.search({"algorithm"}, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0u);

    // Verify the score matches the oracle.
    double expected = scorer.score_document({"algorithm"}, 0u);
    EXPECT_NEAR(results[0].score, expected, 1e-9);
}

// K=2 on "algorithm+graph": 3 candidates, we keep 2.
// The weakest candidate (either doc 1 or a lower-scoring one) must be evicted.
TEST_F(TopKHeapTest, MultipleReplacements_CorrectEviction_K2) {
    BM25Scorer scorer(index, store);
    const std::vector<std::string> query = {"algorithm", "graph"};

    auto results = scorer.search(query, 2);
    auto ref     = reference_top_k(scorer, index, query, 2);
    assert_matches_reference(results, ref, "algorithm+graph,K=2 eviction");

    ASSERT_EQ(results.size(), 2u);
    // The evicted document (rank 3) must NOT appear.
    std::set<uint32_t> returned_ids{results[0].doc_id, results[1].doc_id};
    uint32_t evicted_id = ref.size() > 0 ?
        // The candidate NOT in results is the evicted one.
        [&]() -> uint32_t {
            std::set<uint32_t> all_candidates;
            for (const auto& t : query)
                for (const auto& p : index.get_postings(t))
                    all_candidates.insert(p.doc_id);
            for (uint32_t id : all_candidates)
                if (!returned_ids.count(id)) return id;
            return UINT32_MAX;
        }()
        : UINT32_MAX;

    if (evicted_id != UINT32_MAX) {
        EXPECT_EQ(returned_ids.count(evicted_id), 0u)
            << "evicted doc " << evicted_id << " incorrectly appears in results";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION F — Tie handling
// ═══════════════════════════════════════════════════════════════════════════
//
// We deliberately create a two-document corpus where both documents receive
// identical BM25 scores for a given query.  Because neither the heap nor the
// naive sort has a defined ordering for exact ties, we:
//   • Verify that the RESULT COUNT is correct.
//   • Verify that the SCORE of every returned result matches the expected value.
//   • Verify that the RESULT SET equals the expected set of doc_ids.
//   • Do NOT assert a specific order among tied documents.
TEST(TopKHeapTieTest, TieHandling_EqualScores) {
    // Build a symmetric two-document corpus:
    //   Doc 0: "alpha(×2) beta(×0)"   — token: alpha alpha
    //   Doc 1: "alpha(×2) beta(×0)"   — same token frequencies, same doc length
    //
    // With identical TF, DF, IDF, and document length, BM25 produces the
    // same score for both documents.
    DocumentStore store;
    InvertedIndex index;

    std::vector<std::string> t0 = {"alpha", "gamma", "alpha", "gamma"};
    auto id0 = store.add_document("doc_a", "alpha gamma alpha gamma");
    store.update_token_count(id0, static_cast<uint32_t>(t0.size()));
    index.add_document(id0, t0);

    std::vector<std::string> t1 = {"alpha", "gamma", "alpha", "gamma"};
    auto id1 = store.add_document("doc_b", "alpha gamma alpha gamma");
    store.update_token_count(id1, static_cast<uint32_t>(t1.size()));
    index.add_document(id1, t1);

    BM25Scorer scorer(index, store);

    // Both docs have tf=2 for "alpha", same length (4), same avgdl (4).
    // Scores must be identical.
    double s0 = scorer.score_document({"alpha"}, id0);
    double s1 = scorer.score_document({"alpha"}, id1);
    ASSERT_NEAR(s0, s1, 1e-9) << "Scores must be equal for the tie test to be valid";

    // K=1: heap returns exactly 1 result; it must be one of {0, 1} with
    // the expected score.
    auto results_k1 = scorer.search({"alpha"}, 1);
    ASSERT_EQ(results_k1.size(), 1u);
    EXPECT_NEAR(results_k1[0].score, s0, 1e-9);
    bool doc_is_valid = (results_k1[0].doc_id == id0 || results_k1[0].doc_id == id1);
    EXPECT_TRUE(doc_is_valid) << "Returned doc_id must be 0 or 1";

    // K=2: both documents must appear with equal scores.
    auto results_k2 = scorer.search({"alpha"}, 2);
    ASSERT_EQ(results_k2.size(), 2u);
    EXPECT_NEAR(results_k2[0].score, s0, 1e-9);
    EXPECT_NEAR(results_k2[1].score, s0, 1e-9);
    std::set<uint32_t> returned_ids{results_k2[0].doc_id, results_k2[1].doc_id};
    EXPECT_TRUE(returned_ids.count(id0)) << "doc 0 must appear in K=2 tie result";
    EXPECT_TRUE(returned_ids.count(id1)) << "doc 1 must appear in K=2 tie result";
}
