#include "core/tokenizer.hpp"

#include <cctype>

namespace search {

// ── Default English stop words ──────────────────────────────────────────────
// Kept intentionally lean: common function words that carry almost no
// discriminative value in an information-retrieval context.
Tokenizer::Tokenizer()
    : stop_words_{
          "a",      "an",     "the",    "is",     "are",    "was",
          "were",   "be",     "been",   "being",  "have",   "has",
          "had",    "do",     "does",   "did",    "will",   "would",
          "shall",  "should", "may",    "might",  "must",   "can",
          "could",  "of",     "in",     "to",     "for",    "on",
          "with",   "at",     "by",     "from",   "as",     "into",
          "through","during", "before", "after",  "above",  "below",
          "between","out",    "off",    "over",   "under",  "again",
          "further","then",   "once",   "and",    "but",    "or",
          "if",     "it",     "its",    "this",   "that",   "these",
          "those",  "i",      "me",     "my",     "we",     "our",
          "you",    "your",   "he",     "him",    "his",    "she",
          "her",    "they",   "them",   "their",  "what",   "which",
          "who",    "whom",   "not",    "no",     "nor",    "so",
          "very",   "too",    "here",   "there",  "when",   "where",
          "why",    "how",    "all",    "each",   "every",  "both",
          "few",    "more",   "most",   "other",  "some",   "such",
          "own",    "same",   "than",   "just",   "about"} {}

std::vector<std::string> Tokenizer::tokenize(std::string_view text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            // Accumulate alphanumeric characters, lowercasing on the fly.
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        } else if (!current.empty()) {
            // Non-alphanumeric boundary → flush the current token.
            if (!remove_stop_words_ || !is_stop_word(current)) {
                tokens.push_back(std::move(current));
            }
            current.clear();
        }
    }

    // Flush any trailing token.
    if (!current.empty()) {
        if (!remove_stop_words_ || !is_stop_word(current)) {
            tokens.push_back(std::move(current));
        }
    }

    return tokens;
}

bool Tokenizer::is_stop_word(const std::string& token) const {
    return stop_words_.contains(token);
}

} // namespace search
