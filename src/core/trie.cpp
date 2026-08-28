#include "core/trie.hpp"

#include <algorithm>   // std::lower_bound, std::sort, std::stable_sort
#include <cctype>      // std::isalnum

namespace search {

// ── Construction / Destruction ────────────────────────────────────────────────

Trie::Trie() : root_(std::make_unique<TrieNode>()) {}

// ── Static helpers ────────────────────────────────────────────────────────────

bool Trie::is_valid_char(char c) noexcept {
    // Match the Tokenizer: only alphanumeric ASCII characters survive
    // tokenization.  Anything else cannot be in the inverted index vocabulary.
    return std::isalnum(static_cast<unsigned char>(c)) != 0;
}

// Recursively collect every complete word reachable from `node`.
// `prefix` holds the characters accumulated on the path from the Trie root to
// `node`; it is extended by one character for each recursive call and restored
// (popped) on the way back.
void Trie::collect_all(const TrieNode* node,
                       std::string& prefix,
                       std::vector<std::pair<std::string, std::size_t>>& out) {
    if (node->is_end_of_word) {
        out.emplace_back(prefix, node->document_frequency);
    }

    for (const auto& [ch, child] : node->children) {
        prefix.push_back(ch);
        collect_all(child.get(), prefix, out);
        prefix.pop_back();
    }
}

// ── Core API ──────────────────────────────────────────────────────────────────

void Trie::insert(const std::string& token, std::size_t document_frequency) {
    // Validate: reject tokens containing characters outside the indexed alphabet.
    for (char c : token) {
        if (!is_valid_char(c)) return;
    }
    // Reject empty tokens.
    if (token.empty()) return;

    TrieNode* current = root_.get();

    for (char c : token) {
        // Binary search in the sorted children vector.
        auto it = std::lower_bound(
            current->children.begin(),
            current->children.end(),
            c,
            [](const std::pair<char, std::unique_ptr<TrieNode>>& entry, char ch) {
                return entry.first < ch;
            });

        if (it == current->children.end() || it->first != c) {
            // Character not found → insert a new child at the correct sorted position.
            it = current->children.emplace(it, c, std::make_unique<TrieNode>());
        }
        current = it->second.get();
    }

    // Mark the terminal node.
    current->is_end_of_word = true;
    // Replace (do not accumulate) the stored document frequency.
    current->document_frequency = document_frequency;
}

std::vector<std::pair<std::string, std::size_t>>
Trie::autocomplete(const std::string& prefix, std::size_t limit) const {
    if (limit == 0) return {};

    // Validate the prefix: any invalid character means no valid index term
    // can start with this prefix.
    for (char c : prefix) {
        if (!is_valid_char(c)) return {};
    }

    // Walk the Trie to the node corresponding to the last character of `prefix`.
    const TrieNode* current = root_.get();
    for (char c : prefix) {
        auto it = std::lower_bound(
            current->children.cbegin(),
            current->children.cend(),
            c,
            [](const std::pair<char, std::unique_ptr<TrieNode>>& entry, char ch) {
                return entry.first < ch;
            });

        if (it == current->children.cend() || it->first != c) {
            // Prefix not found in the Trie.
            return {};
        }
        current = it->second.get();
    }

    // Collect all complete words in the subtree rooted at `current`.
    std::vector<std::pair<std::string, std::size_t>> results;
    std::string accumulated = prefix;
    collect_all(current, accumulated, results);

    // Sort:  1. document_frequency descending
    //        2. term lexicographically ascending (tie-break)
    std::stable_sort(results.begin(), results.end(),
        [](const std::pair<std::string, std::size_t>& a,
           const std::pair<std::string, std::size_t>& b) {
            if (a.second != b.second) return a.second > b.second;  // higher DF first
            return a.first < b.first;                              // lexicographic tie-break
        });

    // Truncate to `limit`.
    if (results.size() > limit) {
        results.resize(limit);
    }

    return results;
}

void Trie::clear() {
    // Replace the root node — this destroys the entire tree via unique_ptr
    // chain destruction, which is O(N) but avoids a recursive clear function.
    root_ = std::make_unique<TrieNode>();
}

bool Trie::empty() const noexcept {
    return root_->children.empty();
}

} // namespace search
