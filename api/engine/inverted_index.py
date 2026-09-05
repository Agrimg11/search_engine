"""
inverted_index.py — Inverted index mapping terms → posting lists.

Ports: src/core/inverted_index.hpp / inverted_index.cpp

Heart of keyword search: for every unique word in the corpus the index
stores a list of (doc_id, term_frequency) pairs.

Design: defaultdict of lists — good enough for corpora up to ~100 K docs,
exactly as documented in the C++ header.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


# ── Posting ───────────────────────────────────────────────────────────────────

@dataclass
class Posting:
    """A single entry in a posting list."""
    doc_id:         int
    term_frequency: int


# ── InvertedIndex ─────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    Inverted index: term → [Posting(doc_id, tf), ...]

    Time:
      add_document      : O(T) where T = len(tokens)
      get_postings      : O(1) average
      document_frequency: O(1) average
    """

    def __init__(self) -> None:
        # term → list of Posting
        self._index: dict[str, list[Posting]] = defaultdict(list)
        self._doc_count: int = 0
        # Sentinel returned when a term is absent.
        self._empty: list[Posting] = []

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """
        Index all tokens for a given document.

        Time:  O(T)  where T = len(tokens)
        Space: O(U)  where U = unique tokens in this document
        """
        # Count term frequencies for this document.
        tf_map: dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        for term, tf in tf_map.items():
            self._index[term].append(Posting(doc_id=doc_id, term_frequency=tf))

        self._doc_count += 1

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> list[Posting]:
        """Return the posting list for a term (empty list if absent)."""
        return self._index.get(term, self._empty)

    def document_frequency(self, term: str) -> int:
        """Number of documents that contain `term`."""
        return len(self._index.get(term, self._empty))

    def contains(self, term: str) -> bool:
        return term in self._index

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def total_documents(self) -> int:
        return self._doc_count

    @property
    def vocabulary(self) -> dict[str, list[Posting]]:
        """Read-only view of the entire vocabulary (term → posting list)."""
        return self._index
