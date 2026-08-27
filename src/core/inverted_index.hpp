#pragma once
/// @file inverted_index.hpp
/// @brief Inverted index mapping terms → posting lists.
///
/// This is the heart of keyword search.  For every unique word in the
/// corpus the index stores a sorted list of (doc_id, term_frequency) pairs.
///
/// Design decision: a simple hash-map of vectors.  Good enough for
/// corpora up to ~100 K documents.  For larger corpora you would move
/// to a B-tree backed on-disk structure.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace search {

/// A single entry in a posting list.
struct Posting {
    uint32_t doc_id;
    uint32_t term_frequency;
};

class InvertedIndex {
public:
    /// Index all tokens for a given document.
    ///
    /// Time:  O(T)  where T = tokens.size()
    /// Space: O(U)  where U = unique tokens in this document
    void add_document(uint32_t doc_id, const std::vector<std::string>& tokens);

    /// Return the posting list for a term (empty sentinel if absent).
    [[nodiscard]] const std::vector<Posting>& get_postings(const std::string& term) const;

    /// Number of documents that contain `term`.
    [[nodiscard]] std::size_t document_frequency(const std::string& term) const;

    /// Total number of distinct documents that have been indexed.
    [[nodiscard]] std::size_t total_documents() const noexcept { return doc_count_; }

    /// Check whether a term exists in the index.
    [[nodiscard]] bool contains(const std::string& term) const;

private:
    std::unordered_map<std::string, std::vector<Posting>> index_;
    std::size_t doc_count_{0};

    /// Returned by get_postings() when the term is absent.
    static const std::vector<Posting> kEmptyPostings;
};

} // namespace search
