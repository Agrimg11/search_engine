"""
bm25.py — Okapi BM25 relevance scorer.

Ports: src/core/bm25.hpp / bm25.cpp

BM25 formula per query term q_i in document D:

  score(D, Q) = Σ  IDF(q_i) · tf(q_i, D) · (k1 + 1)
                              ──────────────────────────────────────
                              tf(q_i, D) + k1·(1 − b + b·|D|/avgdl)

where
  IDF(q)  = ln( (N − df + 0.5) / (df + 0.5) + 1 )
  tf      = raw term frequency in the document
  |D|     = document length (token count)
  avgdl   = average document length across the corpus
  N       = total number of documents
  df      = number of documents containing q

Default parameters k1 = 1.5 and b = 0.75 match the C++ implementation
and are the most widely used values in IR literature.

Top-K extraction uses heapq.nlargest — O(R log K) — same strategy as the
C++ Phase 2.2 min-heap optimisation.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from engine.document import DocumentStore
from engine.inverted_index import InvertedIndex


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result: document ID + relevance score."""
    doc_id: int
    score:  float


@dataclass
class TermExplanation:
    """Explanation of a single term's contribution to the BM25 score."""
    term:         str
    tf:           int
    df:           int
    idf:          float
    contribution: float


@dataclass
class DocumentExplanation:
    """Detailed explanation of a document's BM25 score for a query."""
    doc_id:          int
    final_score:     float
    document_length: int
    average_length:  float
    terms:           list[TermExplanation] = field(default_factory=list)


# ── BM25Scorer ────────────────────────────────────────────────────────────────

class BM25Scorer:
    """
    Okapi BM25 scorer.

    Args:
        index:  Reference to the InvertedIndex (read-only).
        store:  Reference to the DocumentStore (read-only).
        k1:     Term-frequency saturation parameter (default 1.5).
        b:      Document-length normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        index: InvertedIndex,
        store: DocumentStore,
        k1:    float = 1.5,
        b:     float = 0.75,
    ) -> None:
        self._index = index
        self._store = store
        self._k1    = k1
        self._b     = b

    # ── Private helpers ───────────────────────────────────────────────────────

    def _idf(self, term: str) -> float:
        """
        Inverse document frequency for a single term.

        IDF(q) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        """
        N  = self._index.total_documents
        df = self._index.document_frequency(term)
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _tf_norm(self, tf: int, doc_length: int) -> float:
        """
        Normalised term frequency component of BM25.

        tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
        """
        avgdl = self._store.average_document_length or 1.0
        denom = tf + self._k1 * (1.0 - self._b + self._b * doc_length / avgdl)
        return tf * (self._k1 + 1.0) / denom

    # ── Public API ────────────────────────────────────────────────────────────

    def score_document(self, query_tokens: list[str], doc_id: int) -> float:
        """Score a single document against a tokenized query."""
        doc       = self._store.get_document(doc_id)
        doc_len   = doc.token_count or 1
        total     = 0.0

        for term in set(query_tokens):           # deduplicate query terms
            postings = self._index.get_postings(term)
            tf = next((p.term_frequency for p in postings if p.doc_id == doc_id), 0)
            if tf == 0:
                continue
            total += self._idf(term) * self._tf_norm(tf, doc_len)

        return total

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[SearchResult]:
        """
        Search the entire corpus and return the top-k results sorted by
        descending score.

        Score accumulation : O(Q · P)   Q = query terms, P = avg postings length
        Top-K extraction   : O(R log K) R = candidate docs, K = top_k
        """
        if not query_tokens or self._index.total_documents == 0:
            return []

        # Accumulate scores across all documents that match at least one term.
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            idf = self._idf(term)
            for posting in self._index.get_postings(term):
                doc     = self._store.get_document(posting.doc_id)
                doc_len = doc.token_count or 1
                contrib = idf * self._tf_norm(posting.term_frequency, doc_len)
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + contrib

        # Top-K extraction via min-heap — O(R log K).
        top = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
        return [SearchResult(doc_id=did, score=sc) for did, sc in top]

    def explain(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[DocumentExplanation]:
        """
        Search the corpus and return detailed term-level BM25 explanations
        for the top-k results.
        """
        if not query_tokens or self._index.total_documents == 0:
            return []

        avgdl = self._store.average_document_length

        # Collect candidate doc IDs (docs that match at least one query term).
        candidate_ids: set[int] = set()
        for term in set(query_tokens):
            for posting in self._index.get_postings(term):
                candidate_ids.add(posting.doc_id)

        explanations: list[DocumentExplanation] = []
        for doc_id in candidate_ids:
            doc     = self._store.get_document(doc_id)
            doc_len = doc.token_count or 1
            total   = 0.0
            terms_detail: list[TermExplanation] = []

            for term in set(query_tokens):
                postings = self._index.get_postings(term)
                tf = next((p.term_frequency for p in postings if p.doc_id == doc_id), 0)
                idf    = self._idf(term)
                contrib = idf * self._tf_norm(tf, doc_len) if tf > 0 else 0.0
                total  += contrib
                terms_detail.append(TermExplanation(
                    term=term, tf=tf,
                    df=self._index.document_frequency(term),
                    idf=idf, contribution=contrib,
                ))

            explanations.append(DocumentExplanation(
                doc_id=doc_id, final_score=total,
                document_length=doc_len, average_length=avgdl,
                terms=sorted(terms_detail, key=lambda t: -t.contribution),
            ))

        # Sort by score descending, take top_k.
        explanations.sort(key=lambda e: -e.final_score)
        return explanations[:top_k]
