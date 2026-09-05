"""
trie.py — In-memory Trie (prefix tree) for autocomplete over the indexed
          vocabulary.

Ports: src/core/trie.hpp / trie.cpp

Design:
  Each TrieNode stores its children in a plain dict (char → TrieNode).
  Python dicts give O(1) average child lookup — better than the C++ sorted
  vector + binary search, while keeping the same external API.

  Insertion : O(L)          L = word length
  Lookup    : O(L)
  Autocomplete: O(L + M·log M)  M = matching words (sort by df)

Ranking:
  Suggestions returned sorted by:
    1. document_frequency descending (most common first)
    2. lexicographic ascending (tie-break)

Characters:
  Only alphanumeric ASCII [a-z0-9] are valid (mirrors Tokenizer).
  Tokens with other characters are silently rejected.
"""

from __future__ import annotations

import re

_VALID = re.compile(r'^[a-z0-9]+$')


class _TrieNode:
    __slots__ = ('children', 'is_end', 'doc_freq')

    def __init__(self) -> None:
        self.children: dict[str, '_TrieNode'] = {}
        self.is_end:   bool = False
        self.doc_freq: int  = 0


class Trie:
    """Prefix tree for vocabulary autocomplete."""

    def __init__(self) -> None:
        self._root = _TrieNode()

    # ── Core API ──────────────────────────────────────────────────────────────

    def insert(self, token: str, document_frequency: int = 1) -> None:
        """
        Insert a token with its corpus document-frequency.

        Tokens containing characters outside [a-z0-9] are silently ignored.
        If the same token is inserted again the document_frequency is replaced.

        Time: O(L)  where L = len(token)
        """
        if not token or not _VALID.match(token):
            return

        node = self._root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]

        node.is_end   = True
        node.doc_freq = document_frequency

    def autocomplete(
        self, prefix: str, limit: int = 10
    ) -> list[tuple[str, int]]:
        """
        Return at most `limit` completions for the given prefix.

        Completions ordered by:
          1. document_frequency descending
          2. lexicographic ascending (tie-break)

        Returns [] when Trie is empty, prefix has no matches, or limit == 0.
        An empty prefix returns up to `limit` words from the entire vocabulary.

        Time: O(L + M·log M)  L = prefix length, M = matching words
        """
        if limit == 0:
            return []

        # Reject prefixes with invalid characters (but allow empty prefix).
        if prefix and not _VALID.match(prefix):
            return []

        # Walk to the prefix node.
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # Collect all completions from this sub-trie.
        results: list[tuple[str, int]] = []
        self._collect(node, prefix, results)

        # Sort: highest df first, then lexicographic.
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:limit]

    def clear(self) -> None:
        """Remove all words from the Trie."""
        self._root = _TrieNode()

    @property
    def empty(self) -> bool:
        return not self._root.children

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect(
        self,
        node:    _TrieNode,
        current: str,
        out:     list[tuple[str, int]],
    ) -> None:
        """Recursively collect all complete words reachable from `node`."""
        if node.is_end:
            out.append((current, node.doc_freq))
        for ch, child in node.children.items():
            self._collect(child, current + ch, out)
