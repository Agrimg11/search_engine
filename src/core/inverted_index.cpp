#include "core/inverted_index.hpp"

#include <unordered_map>

namespace search {

// Static sentinel returned when a term has no postings.
const std::vector<Posting> InvertedIndex::kEmptyPostings{};

void InvertedIndex::add_document(uint32_t doc_id,
                                  const std::vector<std::string>& tokens) {
    // Step 1: count term frequencies locally for this document.
    //         Using a local map avoids scanning the global postings list.
    std::unordered_map<std::string, uint32_t> tf_map;
    for (const auto& token : tokens) {
        ++tf_map[token];
    }

    // Step 2: append one Posting per unique term.
    for (auto& [term, freq] : tf_map) {
        index_[term].push_back(Posting{doc_id, freq});
    }

    ++doc_count_;
}

const std::vector<Posting>& InvertedIndex::get_postings(const std::string& term) const {
    if (auto it = index_.find(term); it != index_.end()) {
        return it->second;
    }
    return kEmptyPostings;
}

std::size_t InvertedIndex::document_frequency(const std::string& term) const {
    if (auto it = index_.find(term); it != index_.end()) {
        return it->second.size();
    }
    return 0;
}

bool InvertedIndex::contains(const std::string& term) const {
    return index_.contains(term);
}

} // namespace search
