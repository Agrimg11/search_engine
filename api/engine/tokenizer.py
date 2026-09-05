"""
tokenizer.py — Text tokenizer with lowercasing, punctuation removal, and
               configurable stop-word filtering.

Ports: src/core/tokenizer.hpp / tokenizer.cpp

Pipeline:  split on non-alphanumeric → lowercase → drop stop-words.
           Mirrors C++ behaviour exactly: only [a-z0-9] tokens survive.
"""

from __future__ import annotations

import re

# ── Default English stop-word list ────────────────────────────────────────────
# Mirrors the hardcoded list in tokenizer.cpp.
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "then", "too", "very", "just", "that",
    "this", "these", "those", "it", "its", "we", "our", "you", "your",
    "he", "she", "they", "them", "their", "i", "me", "my", "us", "who",
    "what", "which", "when", "where", "why", "how", "all", "any", "if",
    "as", "up", "out", "about", "into", "through", "after", "before",
    "between", "during", "without", "within", "along", "following", "across",
    "behind", "beyond", "plus", "except", "until", "toward", "among",
})

# Compiled regex: replace any non-alphanumeric character with a space.
_NON_ALNUM = re.compile(r'[^a-z0-9]+')


class Tokenizer:
    """
    Stateless text tokenizer.

    Time:  O(N) where N = len(text)
    Space: O(T) where T = number of output tokens
    """

    def __init__(self) -> None:
        self._stop_words: frozenset[str] = _DEFAULT_STOP_WORDS
        self._remove_stop_words: bool = True

    # ── Core API ──────────────────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize raw text into a list of normalised terms.

        Steps:
          1. Lowercase
          2. Replace all non-alphanumeric chars with spaces
          3. Split on whitespace
          4. Drop stop-words (if enabled) and empty tokens
        """
        lowered = text.lower()
        cleaned = _NON_ALNUM.sub(' ', lowered)
        tokens = cleaned.split()
        if self._remove_stop_words:
            tokens = [t for t in tokens if t not in self._stop_words]
        return tokens

    # ── Configuration ─────────────────────────────────────────────────────────

    def enable_stop_word_removal(self, enable: bool) -> None:
        self._remove_stop_words = enable

    def set_stop_words(self, words: set[str]) -> None:
        self._stop_words = frozenset(words)
