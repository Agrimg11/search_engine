#pragma once
/// @file document.hpp
/// @brief Document model and in-memory document store.
///
/// Design decision: Documents are stored in a flat std::vector keyed by
/// sequential uint32_t IDs.  This gives O(1) lookup by ID and excellent
/// cache locality compared to an std::unordered_map.

#include <cstdint>
#include <string>
#include <vector>

namespace search {

// ── Document ────────────────────────────────────────────────────────────────
/// A single ingested document.
struct Document {
    uint32_t    id;
    std::string title;
    std::string content;
    uint32_t    token_count{0};   ///< Number of indexed tokens (after stop-word removal).
};

// ── DocumentStore ───────────────────────────────────────────────────────────
/// Owns all documents and maintains corpus-level statistics needed by BM25.
///
/// Time complexity
///   add_document      : amortised O(1)
///   get_document      : O(1)
///   average_doc_length: O(1)  — incrementally maintained
///
/// Space complexity: O(D) where D = number of documents.
class DocumentStore {
public:
    /// Insert a new document and return its auto-assigned ID.
    uint32_t add_document(std::string title, std::string content);

    /// Retrieve a document by ID.  Throws std::out_of_range on invalid ID.
    [[nodiscard]] const Document& get_document(uint32_t doc_id) const;

    /// Set the token count for a document and update the running average.
    void update_token_count(uint32_t doc_id, uint32_t count);

    // ── Accessors ───────────────────────────────────────────────────────
    [[nodiscard]] std::size_t size()                    const noexcept { return documents_.size(); }
    [[nodiscard]] double      average_document_length() const noexcept { return avg_doc_length_; }

private:
    std::vector<Document> documents_;
    uint64_t              total_tokens_{0};
    double                avg_doc_length_{0.0};
};

} // namespace search
