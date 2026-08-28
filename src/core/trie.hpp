#pragma once
/// @file trie.hpp
/// @brief In-memory Trie (prefix tree) for autocomplete over the indexed vocabulary.
///
/// Design — each TrieNode stores its children in a sorted vector of
///           (char, unique_ptr<TrieNode>) pairs.  The vector is kept in
///           ascending character order so binary search can be used for
///           O(log A) child lookup (A = alphabet size, ≤ 36 for [a-z0-9]).
///
/// Insertion : O(L)        L = word length
/// Lookup    : O(L)
/// Autocomplete : O(L + M) M = number of matching nodes visited
///
/// Ranking:
///   Suggestions are returned sorted by:
///     1. document_frequency descending (most common first)
///     2. lexicographic order ascending (tie-break)
///
/// Duplicate insertion replaces the stored document_frequency.
///
/// Thread safety: NOT thread-safe.  Single-threaded use only.
///
/// Characters:
///   Only alphanumeric ASCII characters (std::isalnum) are valid inside
///   tokens.  This mirrors the existing Tokenizer behaviour.  Any token
///   containing an invalid character is silently rejected.

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace search {

// ── TrieNode ──────────────────────────────────────────────────────────────────
class TrieNode {
public:
    /// Children stored in ascending character order for binary-search O(log A).
    std::vector<std::pair<char, std::unique_ptr<TrieNode>>> children;

    bool        is_end_of_word      = false;
    std::size_t document_frequency  = 0;   ///< DF of the word ending here (0 if not a word-end).
};

// ── Trie ─────────────────────────────────────────────────────────────────────
class Trie {
public:
    Trie();
    ~Trie() = default;

    // Non-copyable (unique_ptr ownership chain).
    Trie(const Trie&)            = delete;
    Trie& operator=(const Trie&) = delete;

    // Movable.
    Trie(Trie&&)            = default;
    Trie& operator=(Trie&&) = default;

    // ── Core API ──────────────────────────────────────────────────────────────

    /// Insert a token with its corpus document-frequency.
    ///
    /// If the same token is inserted again the document_frequency is
    /// replaced (not accumulated).
    ///
    /// Tokens containing characters that are not alphanumeric ASCII are
    /// silently ignored — they can never appear in the inverted index.
    ///
    /// Time: O(L)  where L = token.size()
    void insert(const std::string& token, std::size_t document_frequency = 1);

    /// Return at most `limit` completions for the given prefix.
    ///
    /// Completions are ordered by:
    ///   1. document_frequency descending
    ///   2. lexicographic order ascending (tie-break)
    ///
    /// Returns an empty vector when:
    ///   • the Trie is empty
    ///   • no word starts with `prefix`
    ///   • prefix contains invalid characters
    ///   • limit == 0
    ///
    /// An empty prefix returns up to `limit` words from the entire vocabulary.
    ///
    /// Time: O(L + M·log M)  L = prefix length, M = matching words
    [[nodiscard]] std::vector<std::pair<std::string, std::size_t>>
    autocomplete(const std::string& prefix, std::size_t limit = 10) const;

    /// Remove all words from the Trie.
    void clear();

    // ── Accessors ─────────────────────────────────────────────────────────────
    [[nodiscard]] bool empty() const noexcept;

private:
    std::unique_ptr<TrieNode> root_;

    /// Returns true for characters that may appear in indexed tokens.
    /// Mirrors the Tokenizer: only std::isalnum characters survive tokenization.
    [[nodiscard]] static bool is_valid_char(char c) noexcept;

    /// Recursively collect all complete words reachable from `node`,
    /// building the string by appending to `prefix`.
    /// Collected results are appended to `out`.
    static void collect_all(const TrieNode* node,
                             std::string& prefix,
                             std::vector<std::pair<std::string, std::size_t>>& out);
};

} // namespace search
