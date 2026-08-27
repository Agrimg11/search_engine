/// @file test_inverted_index.cpp
/// @brief Unit tests for the InvertedIndex class.

#include "core/inverted_index.hpp"
#include <gtest/gtest.h>

using search::InvertedIndex;

// ── Basic operations ────────────────────────────────────────────────────────

TEST(InvertedIndexTest, SingleDocumentPostings) {
    InvertedIndex idx;
    idx.add_document(0, {"hello", "world", "hello"});

    const auto& postings = idx.get_postings("hello");
    ASSERT_EQ(postings.size(), 1u);
    EXPECT_EQ(postings[0].doc_id, 0u);
    EXPECT_EQ(postings[0].term_frequency, 2u);
}

TEST(InvertedIndexTest, MultipleDocuments) {
    InvertedIndex idx;
    idx.add_document(0, {"hello", "world"});
    idx.add_document(1, {"hello", "search"});

    EXPECT_EQ(idx.document_frequency("hello"),  2u);
    EXPECT_EQ(idx.document_frequency("world"),  1u);
    EXPECT_EQ(idx.document_frequency("search"), 1u);
}

TEST(InvertedIndexTest, DocumentCount) {
    InvertedIndex idx;
    EXPECT_EQ(idx.total_documents(), 0u);

    idx.add_document(0, {"a"});
    EXPECT_EQ(idx.total_documents(), 1u);

    idx.add_document(1, {"b"});
    EXPECT_EQ(idx.total_documents(), 2u);
}

// ── Missing terms ───────────────────────────────────────────────────────────

TEST(InvertedIndexTest, MissingTermReturnsEmptyPostings) {
    InvertedIndex idx;
    idx.add_document(0, {"hello"});

    EXPECT_TRUE(idx.get_postings("missing").empty());
    EXPECT_EQ(idx.document_frequency("missing"), 0u);
}

TEST(InvertedIndexTest, ContainsCheck) {
    InvertedIndex idx;
    idx.add_document(0, {"hello", "world"});

    EXPECT_TRUE(idx.contains("hello"));
    EXPECT_TRUE(idx.contains("world"));
    EXPECT_FALSE(idx.contains("missing"));
}

// ── Term frequency correctness ──────────────────────────────────────────────

TEST(InvertedIndexTest, TermFrequencyIsPerDocument) {
    InvertedIndex idx;
    idx.add_document(0, {"a", "a", "a", "b"});     // tf(a,0)=3, tf(b,0)=1
    idx.add_document(1, {"a", "b", "b"});           // tf(a,1)=1, tf(b,1)=2

    const auto& pa = idx.get_postings("a");
    ASSERT_EQ(pa.size(), 2u);
    // Doc 0 was added first.
    EXPECT_EQ(pa[0].doc_id, 0u);
    EXPECT_EQ(pa[0].term_frequency, 3u);
    EXPECT_EQ(pa[1].doc_id, 1u);
    EXPECT_EQ(pa[1].term_frequency, 1u);

    const auto& pb = idx.get_postings("b");
    ASSERT_EQ(pb.size(), 2u);
    EXPECT_EQ(pb[0].term_frequency, 1u);
    EXPECT_EQ(pb[1].term_frequency, 2u);
}
