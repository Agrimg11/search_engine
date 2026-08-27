#pragma once
/// @file tokenizer.hpp
/// @brief Text tokenizer with Unicode-safe lowercasing, punctuation removal,
///        and configurable stop-word filtering.
///
/// Design decision: tokenization is intentionally simple (whitespace +
/// punctuation split, ASCII lowercase, stop-word list).  More advanced
/// NLP (stemming, lemmatisation) is left for Phase 3 where the semantic
/// pipeline takes over.

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace search {

class Tokenizer {
public:
    /// Construct with a built-in English stop-word list.
    Tokenizer();

    /// Tokenize raw text into a vector of normalised terms.
    ///
    /// Pipeline:  split on non-alphanumeric → lowercase → drop stop words.
    ///
    /// Time:  O(N) where N = text.size()
    /// Space: O(T) where T = number of output tokens
    [[nodiscard]] std::vector<std::string> tokenize(std::string_view text) const;

    // ── Configuration ───────────────────────────────────────────────────
    void enable_stop_word_removal(bool enable) noexcept { remove_stop_words_ = enable; }
    void set_stop_words(std::unordered_set<std::string> words) { stop_words_ = std::move(words); }

private:
    std::unordered_set<std::string> stop_words_;
    bool remove_stop_words_{true};

    [[nodiscard]] bool is_stop_word(const std::string& token) const;
};

} // namespace search
