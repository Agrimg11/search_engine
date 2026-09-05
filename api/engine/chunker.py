"""
chunker.py — Splits a document into overlapping fixed-size character chunks.

Ports: src/core/chunker.hpp / chunker.cpp

Why chunking matters:
  Embedding an entire document produces a diluted average vector.
  Splitting into overlapping windows gives many precise vectors — each
  representing a focused slice of the document.

Overlap prevents boundary effects: a sentence spanning two chunk boundaries
always appears intact in at least one chunk.

Example — 2000-char document, chunk_size=512, overlap=100, step=412:
  chunk 0: chars [   0,  511]
  chunk 1: chars [ 412,  923]
  chunk 2: chars [ 824, 1335]
  chunk 3: chars [1236, 1747]
  chunk 4: chars [1648, 1999]  (last chunk may be shorter)

Fully stateless — identical to the C++ free function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """Metadata about one chunk produced by split_into_chunks()."""
    text:        str
    start_char:  int
    chunk_index: int


def split_into_chunks(
    text:       str,
    chunk_size: int = 512,
    overlap:    int = 100,
) -> list[TextChunk]:
    """
    Split `text` into overlapping fixed-size character chunks.

    Args:
        text:       The input text to chunk.
        chunk_size: Target character length of each chunk (default 512).
        overlap:    Characters shared between adjacent chunks (default 100).
                    Clamped to chunk_size // 2 if overlap >= chunk_size.

    Returns:
        A list of TextChunk. Empty iff text is empty or chunk_size is 0.
        Always at least one chunk if text is non-empty.
    """
    if not text or chunk_size == 0:
        return []

    # Clamp overlap — mirrors the C++ behaviour.
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    step   = chunk_size - overlap
    chunks: list[TextChunk] = []
    start  = 0
    idx    = 0

    while start < len(text):
        end  = min(start + chunk_size, len(text))
        chunks.append(TextChunk(
            text        = text[start:end],
            start_char  = start,
            chunk_index = idx,
        ))
        if end == len(text):
            break
        start += step
        idx   += 1

    return chunks
