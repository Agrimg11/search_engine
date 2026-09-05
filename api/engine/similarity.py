"""
similarity.py — Dense-vector math utilities for the semantic search pipeline.

Ports: src/core/similarity.hpp / similarity.cpp

Key design note (same as C++):
  VectorStore normalises every embedding to unit L2 length at insertion time
  and normalises query embeddings before searching.
  For two unit vectors:
    cosine_similarity(a, b) = dot(a, b) / (1 · 1) = dot(a, b)

  So dot_product() IS cosine_similarity for this engine — no sqrt calls needed
  during hot-path search comparisons.

All operations backed by numpy for C-speed array math.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Return a unit-L2-length copy of `v`.

    If `v` is a zero vector (‖v‖ == 0), returns `v` unchanged and the
    caller should discard the embedding.

    Time: O(D)  Space: O(D)
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product of two vectors.

    For two L2-normalised vectors this equals cosine similarity.

    Time: O(D)
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )
    return float(np.dot(a, b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two arbitrary vectors.

    Normalises copies internally. Returns 0.0 if either vector is zero.
    Use dot_product() when vectors are already unit-normalised (faster).

    Time: O(D)
    Returns: value in [-1.0, 1.0]
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )
    na = l2_normalize(a.astype(np.float32))
    nb = l2_normalize(b.astype(np.float32))
    return float(np.dot(na, nb))
