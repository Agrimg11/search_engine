/// @file test_tokenizer.cpp
/// @brief Unit tests for the Tokenizer class.

#include "core/tokenizer.hpp"
#include <gtest/gtest.h>

using search::Tokenizer;

// ── Basic behaviour ─────────────────────────────────────────────────────────

TEST(TokenizerTest, LowercasesAllTokens) {
    Tokenizer tok;
    tok.enable_stop_word_removal(false);

    auto tokens = tok.tokenize("Hello WORLD fOo");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
    EXPECT_EQ(tokens[2], "foo");
}

TEST(TokenizerTest, RemovesPunctuation) {
    Tokenizer tok;
    tok.enable_stop_word_removal(false);

    auto tokens = tok.tokenize("Hello, World! C++ is great.");
    // Expected: "hello", "world", "c", "is", "great"
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}

TEST(TokenizerTest, PreservesNumbers) {
    Tokenizer tok;
    tok.enable_stop_word_removal(false);

    auto tokens = tok.tokenize("IPv4 has 32 bits");
    bool found_ipv4 = false, found_32 = false;
    for (const auto& t : tokens) {
        if (t == "ipv4") found_ipv4 = true;
        if (t == "32")   found_32   = true;
    }
    EXPECT_TRUE(found_ipv4);
    EXPECT_TRUE(found_32);
}

// ── Stop-word removal ───────────────────────────────────────────────────────

TEST(TokenizerTest, FiltersStopWords) {
    Tokenizer tok;  // stop words ON by default

    auto tokens = tok.tokenize("the quick brown fox is a fast animal");
    for (const auto& t : tokens) {
        EXPECT_NE(t, "the");
        EXPECT_NE(t, "is");
        EXPECT_NE(t, "a");
    }
    // "quick", "brown", "fox", "fast", "animal" should remain.
    EXPECT_EQ(tokens.size(), 5u);
}

TEST(TokenizerTest, StopWordRemovalCanBeDisabled) {
    Tokenizer tok;
    tok.enable_stop_word_removal(false);

    auto tokens = tok.tokenize("the cat is here");
    // All words should survive when stop-word removal is off.
    EXPECT_EQ(tokens.size(), 4u);
}

// ── Edge cases ──────────────────────────────────────────────────────────────

TEST(TokenizerTest, EmptyInput) {
    Tokenizer tok;
    EXPECT_TRUE(tok.tokenize("").empty());
}

TEST(TokenizerTest, OnlyPunctuation) {
    Tokenizer tok;
    EXPECT_TRUE(tok.tokenize("!!! ??? ...").empty());
}

TEST(TokenizerTest, OnlyWhitespace) {
    Tokenizer tok;
    EXPECT_TRUE(tok.tokenize("   \t \n  ").empty());
}
