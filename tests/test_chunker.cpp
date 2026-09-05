/// @file test_chunker.cpp
/// @brief Unit tests for split_into_chunks().
/// Phase 4 tests.

#include "core/chunker.hpp"
#include <gtest/gtest.h>
#include <string>

using namespace search;

// ── Edge cases ────────────────────────────────────────────────────────────────

TEST(Chunker, EmptyStringReturnsEmpty) {
    auto chunks = split_into_chunks("");
    EXPECT_TRUE(chunks.empty());
}

TEST(Chunker, ZeroChunkSizeReturnsEmpty) {
    auto chunks = split_into_chunks("hello world", 0);
    EXPECT_TRUE(chunks.empty());
}

// ── Single-chunk cases ────────────────────────────────────────────────────────

TEST(Chunker, ShortStringProducesOneChunk) {
    // "hello" is only 5 chars — well below default chunk_size of 512.
    auto chunks = split_into_chunks("hello");
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].text, "hello");
    EXPECT_EQ(chunks[0].start_char, 0u);
    EXPECT_EQ(chunks[0].chunk_index, 0u);
}

TEST(Chunker, ExactChunkSizeOneChunk) {
    // Text length == chunk_size should produce exactly one chunk.
    std::string text(10, 'x');
    auto chunks = split_into_chunks(text, 10, 2);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].text.size(), 10u);
    EXPECT_EQ(chunks[0].start_char, 0u);
}

// ── Multi-chunk cases ─────────────────────────────────────────────────────────

TEST(Chunker, TwoChunksProduced) {
    // Text = 20 chars, chunk_size = 12, overlap = 4, step = 8.
    // chunk 0: [0, 12)
    // chunk 1: [8, 20)
    std::string text(20, 'a');
    auto chunks = split_into_chunks(text, 12, 4);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].start_char, 0u);
    EXPECT_EQ(chunks[1].start_char, 8u);
}

TEST(Chunker, ChunkIndexIncremental) {
    std::string text(100, 'z');
    auto chunks = split_into_chunks(text, 20, 5);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        EXPECT_EQ(chunks[i].chunk_index, i);
    }
}

TEST(Chunker, ChunkContentsMatchOriginal) {
    // Verify each chunk's text is the exact substring of the original.
    std::string text = "ABCDEFGHIJKLMNOPQRST";  // 20 chars
    auto chunks = split_into_chunks(text, 10, 3);  // step = 7
    for (const auto& c : chunks) {
        std::string expected = text.substr(c.start_char,
                                            std::min(std::size_t{10},
                                                     text.size() - c.start_char));
        EXPECT_EQ(c.text, expected);
    }
}

TEST(Chunker, LastChunkCanBeShorter) {
    // Total 15, chunk_size=10, overlap=2, step=8.
    // chunk 0: [0,10) = 10 chars
    // chunk 1: [8,15) = 7 chars (shorter — reaches end of text)
    std::string text(15, 'q');
    auto chunks = split_into_chunks(text, 10, 2);
    ASSERT_GE(chunks.size(), 2u);
    EXPECT_EQ(chunks.back().text.size(), 7u);
}

// ── Overlap clamping ──────────────────────────────────────────────────────────

TEST(Chunker, OverlapClampedWhenTooLarge) {
    // overlap >= chunk_size should be clamped to chunk_size/2.
    // Must not hang or crash; must produce > 1 chunk for long enough text.
    std::string text(100, 'k');
    // overlap=50 == chunk_size=50 → clamped to 25, step=25.
    EXPECT_NO_THROW({
        auto chunks = split_into_chunks(text, 50, 50);
        EXPECT_FALSE(chunks.empty());
    });
}

// ── Start character offsets ───────────────────────────────────────────────────

TEST(Chunker, StartCharOffsetsCorrect) {
    // chunk_size=10, overlap=2, step=8.
    // chunk 0: start=0, chunk 1: start=8, chunk 2: start=16 ...
    std::string text(50, 'x');
    auto chunks = split_into_chunks(text, 10, 2);
    for (std::size_t i = 0; i < chunks.size() - 1; ++i) {
        EXPECT_EQ(chunks[i + 1].start_char, chunks[i].start_char + 8u);
    }
}
