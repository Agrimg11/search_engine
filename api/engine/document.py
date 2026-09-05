"""
document.py — Document model and in-memory document store.

Ports: src/core/document.hpp / document.cpp

Design mirrors the C++ implementation:
  - Documents stored in a flat list keyed by sequential int IDs → O(1) lookup.
  - Average document length maintained incrementally → O(1) update.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Document ──────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single ingested document."""
    id:          int
    title:       str
    content:     str
    token_count: int = 0


# ── DocumentStore ─────────────────────────────────────────────────────────────

class DocumentStore:
    """
    Owns all documents and maintains corpus-level statistics needed by BM25.

    Time complexity:
      add_document            : amortised O(1)
      get_document            : O(1)
      average_document_length : O(1) — incrementally maintained

    Space complexity: O(D) where D = number of documents.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._total_tokens: int = 0
        self._avg_doc_length: float = 0.0

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(self, title: str, content: str) -> int:
        """Insert a new document and return its auto-assigned ID."""
        doc_id = len(self._documents)
        self._documents.append(Document(id=doc_id, title=title, content=content))
        return doc_id

    def update_token_count(self, doc_id: int, count: int) -> None:
        """Set the token count for a document and update the running average."""
        self._documents[doc_id].token_count = count
        self._total_tokens += count
        n = len(self._documents)
        self._avg_doc_length = self._total_tokens / n if n > 0 else 0.0

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_document(self, doc_id: int) -> Document:
        """Retrieve a document by ID. Raises IndexError on invalid ID."""
        return self._documents[doc_id]

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def average_document_length(self) -> float:
        return self._avg_doc_length

    def __iter__(self):
        return iter(self._documents)
