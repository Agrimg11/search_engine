#include "core/document.hpp"

#include <stdexcept>
#include <string>

namespace search {

uint32_t DocumentStore::add_document(std::string title, std::string content) {
    auto id = static_cast<uint32_t>(documents_.size());
    documents_.push_back(Document{id, std::move(title), std::move(content), 0});
    return id;
}

const Document& DocumentStore::get_document(uint32_t doc_id) const {
    if (doc_id >= documents_.size()) {
        throw std::out_of_range("Document ID " + std::to_string(doc_id) + " not found");
    }
    return documents_[doc_id];
}

void DocumentStore::update_token_count(uint32_t doc_id, uint32_t count) {
    if (doc_id >= documents_.size()) {
        throw std::out_of_range("Document ID " + std::to_string(doc_id) + " not found");
    }
    // Update running total: subtract old count, add new count.
    total_tokens_ -= documents_[doc_id].token_count;
    documents_[doc_id].token_count = count;
    total_tokens_ += count;

    // Recompute average incrementally.
    avg_doc_length_ = documents_.empty()
                          ? 0.0
                          : static_cast<double>(total_tokens_) / static_cast<double>(documents_.size());
}

} // namespace search
