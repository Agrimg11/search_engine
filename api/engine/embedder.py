"""
embedder.py — Abstract embedding provider interface and Ollama implementation.

Ports:
  src/embedding/embedding_provider.hpp  → EmbeddingProvider (ABC)
  src/embedding/ollama_embedder.cpp     → OllamaEmbedder

OllamaEmbedder uses httpx (sync) to POST to the local Ollama server at
/api/embeddings — the same HTTP endpoint that the C++ libcurl implementation
calls.  The response is a raw float list; L2-normalisation is done by
VectorStore.add(), not here (same separation of concerns as C++).

is_available() performs a lightweight GET /api/tags health-check — the
same check used in api/routers/health.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx


# ── Abstract interface ────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """Abstract interface for text-to-vector embedding models."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Convert `text` into a dense float vector.

        The returned vector is NOT L2-normalised — VectorStore.add() handles
        that (same responsibility boundary as the C++ version).

        Returns an empty list on error (network failure, parse error, etc.).
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Quick health check: returns True if the embedder can serve requests."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for display and logging."""

    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension. Returns 0 if unknown or not yet connected."""


# ── Ollama implementation ─────────────────────────────────────────────────────

class OllamaEmbedder(EmbeddingProvider):
    """
    Embedding provider backed by a local Ollama server.

    Sends HTTP POST to /api/embeddings — the same endpoint called by the
    C++ OllamaEmbedder via libcurl.

    Args:
        model:    Ollama model name (default "nomic-embed-text").
        base_url: Ollama server URL (default "http://localhost:11434").
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model:    str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout:  float = 30.0,
    ) -> None:
        self._model    = model
        self._base_url = base_url.rstrip('/')
        self._timeout  = timeout
        self._dim:  int = 0

    # ── EmbeddingProvider interface ───────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Embed `text` using Ollama. Returns [] on any error.

        Calls POST /api/embeddings with {"model": ..., "prompt": ...}
        and extracts the "embedding" field from the JSON response.
        """
        if not text.strip():
            return []

        try:
            response = httpx.post(
                f"{self._base_url}/api/embeddings",
                json    = {"model": self._model, "prompt": text},
                timeout = self._timeout,
            )
            response.raise_for_status()
            vec = response.json().get("embedding", [])
            if vec and self._dim == 0:
                self._dim = len(vec)
            return vec
        except Exception:
            return []

    def is_available(self) -> bool:
        """
        Check Ollama connectivity by GET /api/tags.
        Same health-check as api/routers/health.py.
        """
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def name(self) -> str:
        return f"OllamaEmbedder({self._model})"

    def dimension(self) -> int:
        return self._dim
