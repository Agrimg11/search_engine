#pragma once
/// @file bm25.hpp
/// @brief BM25 (Okapi BM25) relevance scorer.
///
/// BM25 formula per query term q_i in document D:
///
///   score(D, Q) = Σ  IDF(q_i) · tf(q_i, D) · (k1 + 1)
///                                ─────────────────────────────────
///                                tf(q_i, D) + k1·(1 − b + b·|D|/avgdl)
///
/// where
///   IDF(q)  = ln( (N − df + 0.5) / (df + 0.5) + 1 )
///   tf      = raw term frequency in the document
///   |D|     = document length (token count)
///   avgdl   = average document length across the corpus
///   N       = total number of documents
///   df      = number of documents containing q
///
/// Default parameters k1 = 1.5 and b = 0.75 are the most widely used
/// values in information retrieval literature.

#include "document.hpp"
#include "inverted_index.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace search {

/// A single search result: document ID + relevance score.
struct SearchResult {
    uint32_t doc_id;
    double   score;
};

/// Explanation of a single term's contribution to the score.
struct TermExplanation {
    std::string term;
    uint32_t tf;
    uint32_t df;
    double idf;
    double contribution;
};

/// Detailed explanation of a document's BM25 score for a query.
struct DocumentExplanation {
    uint32_t doc_id;
    double final_score;
    uint32_t document_length;
    double average_length;
    std::vector<TermExplanation> terms;
};

class BM25Scorer {
public:
    /// @param index  Reference to the inverted index (read-only).
    /// @param store  Reference to the document store (read-only).
    /// @param k1     Term-frequency saturation parameter (default 1.5).
    /// @param b      Document-length normalisation parameter (default 0.75).
    explicit BM25Scorer(const InvertedIndex& index,
                        const DocumentStore&  store,
                        double k1 = 1.5,
                        double b  = 0.75);

    /// Score a single document against a tokenized query.
    [[nodiscard]] double score_document(const std::vector<std::string>& query_tokens,
                                        uint32_t doc_id) const;

    /// Search the entire corpus and return the top-k results, sorted by
    /// descending score.
    ///
    /// Time:
    ///   Score accumulation : O(Q · P)   Q = query terms, P = avg postings length
    ///   Top-K extraction   : O(R log K) R = candidate docs matched, K = top_k
    ///
    ///   Phase 2 improvement: uses a min-heap of size K instead of a full
    ///   O(R log R) sort.  When K << R this is significantly faster.
    ///
    /// Space:
    ///   Candidate score map : O(R)  — one entry per matched document
    ///   Top-K heap          : O(K)
    ///   Total               : O(R)
    [[nodiscard]] std::vector<SearchResult> search(const std::vector<std::string>& query_tokens,
                                                    std::size_t top_k = 10) const;

    /// Search the corpus and return detailed explanations for the top-k results.
    [[nodiscard]] std::vector<DocumentExplanation> explain(const std::vector<std::string>& query_tokens,
                                                           std::size_t top_k = 10) const;


private:
    const InvertedIndex& index_;
    const DocumentStore& store_;
    double k1_;
    double b_;

    /// Inverse document frequency for a single term.
    [[nodiscard]] double idf(const std::string& term) const;
};

} // namespace search
