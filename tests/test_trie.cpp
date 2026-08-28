/// @file test_trie.cpp
/// @brief Phase 2.3 — Trie-based Autocomplete: correctness and regression tests.
///
/// ═══════════════════════════════════════════════════════════════════════════
/// TESTING STRATEGY
/// ═══════════════════════════════════════════════════════════════════════════
///
/// Three test suites:
///
///   Suite 1  TrieBasic           (12 tests) — insert, lookup, ranking, edge cases
///   Suite 2  TrieFuzz            (3 tests)  — property-based / adversarial inputs
///   Suite 3  TrieSearchInteg     (5 tests)  — SearchEngine.suggest() integration
///
/// Ranking invariant:
///   Suggestions are ordered by:
///     1. document_frequency descending
///     2. lexicographic order ascending (tie-break)
///
/// ── Coverage checklist ──────────────────────────────────────────────────────
///   1.  Single insertion
///   2.  Multiple words
///   3.  Prefix matching
///   4.  Multiple words sharing a prefix
///   5.  Duplicate insertion updates DF (replaces, not adds)
///   6.  DF ranking
///   7.  Lexical tie-breaking
///   8.  Limit is respected
///   9.  Prefix not found
///  10.  Empty prefix
///  11.  Exact word that is also a prefix
///  12.  Invalid/punctuation input
///  13.  SearchEngine integration

#include "core/trie.hpp"
#include "engine/search_engine.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace search;

// ═══════════════════════════════════════════════════════════════════════════
// Suite 1 — Basic Operations
// ═══════════════════════════════════════════════════════════════════════════

// ── Test 1: Single insertion ─────────────────────────────────────────────────
TEST(TrieBasic, SingleInsertion) {
    Trie trie;
    trie.insert("algorithm", 5);

    auto results = trie.autocomplete("algorithm");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].first,  "algorithm");
    EXPECT_EQ(results[0].second, 5u);
}

// ── Test 2: Multiple words ────────────────────────────────────────────────────
TEST(TrieBasic, MultipleWords) {
    Trie trie;
    trie.insert("cat",  3);
    trie.insert("car",  7);
    trie.insert("card", 2);

    // All words should be retrievable by their full name.
    auto r1 = trie.autocomplete("cat");
    ASSERT_EQ(r1.size(), 1u);
    EXPECT_EQ(r1[0].first, "cat");

    auto r2 = trie.autocomplete("card");
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_EQ(r2[0].first, "card");
}

// ── Test 3: Prefix matching ───────────────────────────────────────────────────
TEST(TrieBasic, PrefixMatching) {
    Trie trie;
    trie.insert("apple",  4);
    trie.insert("apply",  2);
    trie.insert("apt",    1);
    trie.insert("banana", 9);

    auto results = trie.autocomplete("ap");
    ASSERT_EQ(results.size(), 3u);

    // Verify all returned terms start with "ap".
    for (const auto& [term, df] : results) {
        EXPECT_EQ(term.substr(0, 2), "ap") << "Term '" << term << "' does not start with 'ap'";
    }
}

// ── Test 4: Multiple words sharing a prefix ───────────────────────────────────
TEST(TrieBasic, MultipleWordsSharedPrefix) {
    Trie trie;
    trie.insert("algo",        12);
    trie.insert("algorithm",    8);
    trie.insert("algorithmic",  5);
    trie.insert("algebra",      3);

    auto results = trie.autocomplete("alg");
    ASSERT_EQ(results.size(), 4u);

    // Verify DF ordering (highest first).
    EXPECT_EQ(results[0].first,  "algo");
    EXPECT_EQ(results[0].second, 12u);
    EXPECT_EQ(results[1].first,  "algorithm");
    EXPECT_EQ(results[1].second, 8u);
    EXPECT_EQ(results[2].first,  "algorithmic");
    EXPECT_EQ(results[2].second, 5u);
    EXPECT_EQ(results[3].first,  "algebra");
    EXPECT_EQ(results[3].second, 3u);
}

// ── Test 5: Duplicate insertion REPLACES DF (does not accumulate) ─────────────
TEST(TrieBasic, DuplicateInsertionReplacesDF) {
    Trie trie;
    trie.insert("algorithm", 5);
    trie.insert("algorithm", 8);  // replace

    auto results = trie.autocomplete("algorithm");
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].second, 8u);  // must be 8, not 13
}

// ── Test 6: DF ranking ────────────────────────────────────────────────────────
TEST(TrieBasic, DFRanking) {
    Trie trie;
    trie.insert("bat", 1);
    trie.insert("ball", 10);
    trie.insert("band", 5);

    auto results = trie.autocomplete("ba");
    ASSERT_EQ(results.size(), 3u);

    // Highest DF first.
    EXPECT_EQ(results[0].first,  "ball");
    EXPECT_EQ(results[0].second, 10u);
    EXPECT_EQ(results[1].first,  "band");
    EXPECT_EQ(results[1].second, 5u);
    EXPECT_EQ(results[2].first,  "bat");
    EXPECT_EQ(results[2].second, 1u);
}

// ── Test 7: Lexical tie-breaking ──────────────────────────────────────────────
TEST(TrieBasic, LexicalTieBreaking) {
    Trie trie;
    trie.insert("algorithm", 8);   // same DF
    trie.insert("algebra",   8);   // same DF

    auto results = trie.autocomplete("alg");
    ASSERT_EQ(results.size(), 2u);
    // Both have DF=8 → lexicographic order: "algebra" < "algorithm"
    EXPECT_EQ(results[0].first, "algebra");
    EXPECT_EQ(results[1].first, "algorithm");
}

// ── Test 8: Limit is respected ────────────────────────────────────────────────
TEST(TrieBasic, LimitRespected) {
    Trie trie;
    for (int i = 0; i < 20; ++i) {
        trie.insert("word" + std::to_string(i), static_cast<std::size_t>(i + 1));
    }

    auto results = trie.autocomplete("word", 5);
    EXPECT_EQ(results.size(), 5u);

    // Verify limit=0 returns nothing.
    auto empty = trie.autocomplete("word", 0);
    EXPECT_TRUE(empty.empty());
}

// ── Test 9: Prefix not found ──────────────────────────────────────────────────
TEST(TrieBasic, PrefixNotFound) {
    Trie trie;
    trie.insert("hello", 3);

    auto results = trie.autocomplete("xyz");
    EXPECT_TRUE(results.empty());
}

// ── Test 10: Empty prefix returns words from entire vocabulary ─────────────────
TEST(TrieBasic, EmptyPrefixReturnsAll) {
    Trie trie;
    trie.insert("alpha", 5);
    trie.insert("beta",  2);
    trie.insert("gamma", 9);

    auto results = trie.autocomplete("", 10);
    EXPECT_EQ(results.size(), 3u);

    // Should still be DF-ranked.
    EXPECT_EQ(results[0].first, "gamma");   // DF=9
    EXPECT_EQ(results[1].first, "alpha");   // DF=5
    EXPECT_EQ(results[2].first, "beta");    // DF=2
}

// ── Test 11: Exact word that is also a prefix ──────────────────────────────────
TEST(TrieBasic, ExactWordAlsoPrefix) {
    Trie trie;
    trie.insert("algo",      12);
    trie.insert("algorithm",  8);

    // Searching for the exact word "algo" should return "algo" itself too.
    auto results = trie.autocomplete("algo");
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].first,  "algo");        // DF=12, higher
    EXPECT_EQ(results[1].first,  "algorithm");   // DF=8
}

// ── Test 12: Invalid / punctuation input ──────────────────────────────────────
TEST(TrieBasic, InvalidCharacterInPrefix) {
    Trie trie;
    trie.insert("hello", 5);

    // A prefix containing a non-alphanumeric character cannot match anything.
    auto r1 = trie.autocomplete("hel!");
    EXPECT_TRUE(r1.empty());

    auto r2 = trie.autocomplete("hel lo");
    EXPECT_TRUE(r2.empty());

    auto r3 = trie.autocomplete("hel-lo");
    EXPECT_TRUE(r3.empty());
}

TEST(TrieBasic, InvalidCharacterTokenNotInserted) {
    Trie trie;
    // Tokens with invalid characters should be silently rejected.
    trie.insert("bad token", 5);        // space
    trie.insert("hyphen-word", 3);      // hyphen
    trie.insert("exclaim!", 2);         // exclamation

    // None of these should appear in the Trie.
    auto r1 = trie.autocomplete("bad");
    EXPECT_TRUE(r1.empty());

    auto r2 = trie.autocomplete("hyphen");
    EXPECT_TRUE(r2.empty());

    auto r3 = trie.autocomplete("exclaim");
    EXPECT_TRUE(r3.empty());
}

// ── Additional: clear() resets the Trie ──────────────────────────────────────
TEST(TrieBasic, ClearResetsState) {
    Trie trie;
    trie.insert("hello", 5);
    trie.clear();

    auto results = trie.autocomplete("hello");
    EXPECT_TRUE(results.empty());

    EXPECT_TRUE(trie.empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite 2 — Fuzz / Property Tests
// ═══════════════════════════════════════════════════════════════════════════

// Property: results never exceed the requested limit.
TEST(TrieFuzz, ResultsNeverExceedLimit) {
    Trie trie;
    std::vector<std::string> words = {
        "apple", "application", "appetizer", "apply", "apt",
        "apricot", "ape", "apex", "aperture", "apathy"
    };
    for (std::size_t i = 0; i < words.size(); ++i) {
        trie.insert(words[i], i + 1);
    }

    for (std::size_t limit = 0; limit <= words.size() + 2; ++limit) {
        auto results = trie.autocomplete("ap", limit);
        EXPECT_LE(results.size(), limit)
            << "Limit=" << limit << " but got " << results.size() << " results";
    }
}

// Property: every returned term starts with the requested prefix.
TEST(TrieFuzz, AllResultsStartWithPrefix) {
    Trie trie;
    std::vector<std::string> words = {
        "search", "searcher", "searching", "sea", "season",
        "seat", "seam", "seal", "sear", "second"
    };
    for (std::size_t i = 0; i < words.size(); ++i) {
        trie.insert(words[i], i + 1);
    }

    std::string prefix = "sea";
    auto results = trie.autocomplete(prefix, 20);
    for (const auto& [term, df] : results) {
        EXPECT_EQ(term.substr(0, prefix.size()), prefix)
            << "Term '" << term << "' does not start with prefix '" << prefix << "'";
    }
}

// Property: arbitrary queries never crash.
TEST(TrieFuzz, ArbitraryPrefixesNeverCrash) {
    Trie trie;
    trie.insert("test", 1);
    trie.insert("testing", 2);
    trie.insert("tester", 3);

    // These should not crash regardless of content.
    std::vector<std::string> adversarial_prefixes = {
        "", "t", "te", "tes", "test", "testy",
        "!!", "test!", " ", "\t", "UPPER", "123", "t e s t"
    };
    for (const auto& p : adversarial_prefixes) {
        EXPECT_NO_THROW({
            auto results = trie.autocomplete(p, 10);
            (void)results;
        }) << "Prefix '" << p << "' caused an exception";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite 3 — SearchEngine Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

// Helper: build a SearchEngine with a deterministic in-memory corpus.
// The engine uses tokens that survive the Tokenizer (stop-words are filtered).
static SearchEngine build_test_engine() {
    SearchEngine engine;

    // Documents are chosen so specific prefixes have predictable results.
    // The tokenizer lowercases and removes stop-words and punctuation.
    engine.ingest("Doc1", "algorithm data structures algo");            // algo, algorithm, data, structures
    engine.ingest("Doc2", "algorithm efficient search algorithm");       // algorithm (TF=2), efficient, search
    engine.ingest("Doc3", "algorithmic complexity analysis");           // algorithmic, complexity, analysis
    engine.ingest("Doc4", "algebra linear equations");                   // algebra, linear, equations

    return engine;
}

// Integration test: basic prefix lookup through SearchEngine.
TEST(TrieSearchInteg, SuggestBasicPrefix) {
    SearchEngine engine = build_test_engine();

    // "alg" should match: algo, algorithm, algorithmic, algebra
    auto suggestions = engine.suggest("alg", 10);
    EXPECT_FALSE(suggestions.empty());

    // All suggestions must start with "alg".
    for (const auto& [term, df] : suggestions) {
        EXPECT_EQ(term.substr(0, 3), "alg")
            << "'" << term << "' does not start with 'alg'";
    }
}

// Integration test: DF ranking through SearchEngine.
TEST(TrieSearchInteg, SuggestDFOrdering) {
    SearchEngine engine = build_test_engine();

    // "algorithm" appears in Doc1 and Doc2 (DF=2).
    // "algo" appears in Doc1 only (DF=1).
    // So "algorithm" should rank before "algo" for prefix "algor".
    auto suggestions = engine.suggest("algor", 10);
    ASSERT_FALSE(suggestions.empty());
    EXPECT_EQ(suggestions[0].first, "algorithm");
}

// Integration test: exact word that is also a prefix.
TEST(TrieSearchInteg, SuggestExactWordAlsoPrefix) {
    SearchEngine engine = build_test_engine();

    // "algorithm" itself should appear when we search for prefix "algorithm".
    auto suggestions = engine.suggest("algorithm", 10);
    EXPECT_FALSE(suggestions.empty());

    bool found = std::any_of(suggestions.begin(), suggestions.end(),
        [](const std::pair<std::string, std::size_t>& s) {
            return s.first == "algorithm";
        });
    EXPECT_TRUE(found) << "'algorithm' should be in suggestions for prefix 'algorithm'";
}

// Integration test: limit is respected through SearchEngine.
TEST(TrieSearchInteg, SuggestLimitRespected) {
    SearchEngine engine = build_test_engine();

    auto suggestions = engine.suggest("alg", 2);
    EXPECT_LE(suggestions.size(), 2u);

    auto none = engine.suggest("alg", 0);
    EXPECT_TRUE(none.empty());
}

// Integration test: prefix with no matches returns empty.
TEST(TrieSearchInteg, SuggestPrefixNotFound) {
    SearchEngine engine = build_test_engine();

    auto suggestions = engine.suggest("zzz", 10);
    EXPECT_TRUE(suggestions.empty());
}
