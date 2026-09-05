# Fast Hybrid C++ Search Engine

A high-performance, in-memory text search engine built in C++. This project implements an interactive CLI search tool using a hybrid architecture that combines a custom inverted index, Okapi BM25 relevance scoring, a Top-K Min-Heap, an LRU Cache, Trie-based autocomplete, and a **dense vector semantic search layer** powered by Ollama embeddings.

## Features

- **Okapi BM25 Scoring:** Mathematically robust relevance scoring for document retrieval.
- **Inverted Index:** Fast $O(1)$ term lookups mapping words to documents.
- **LRU Cache:** $O(1)$ query caching using a doubly-linked list and hash map for instant retrieval of repeated queries. Cache key includes `RankingMode` so BM25 and semantic results never collide.
- **Top-K Min-Heap:** $O(R \log K)$ optimization for ranking results without sorting the full dataset.
- **Trie Autocomplete:** Vocabulary prefix-tree returning ranked suggestions by document frequency.
- **Semantic Search (Phase 3):** Dense embedding vectors via a local Ollama server. Finds documents by *meaning*, not just keywords.
- **Hybrid Search (Phase 3):** Fuses BM25 and semantic scores: `α·NormBM25 + β·SemanticScore`.
- **Snippet Highlighting:** Matched query terms are bolded/coloured in result snippets (TTY-aware ANSI).
- **Session Analytics:** Tracks query frequency and average result count per session.

## Architecture & Phases

- **Phase 1:** Basic Tokenization, Inverted Index, and BM25 Scoring.
- **Phase 2.2:** LRU Cache (O(1) bypass) and Min-Heap Top-K extraction.
- **Phase 2.3:** Trie-Based Autocomplete (Tab-completion + `:suggest`).
- **Phase 3 (Current):** Semantic & Hybrid Search via Ollama embeddings.
  - `similarity.hpp` — `l2_normalize`, `dot_product`, `cosine_similarity`
  - `vector_store.hpp` — Brute-force KNN with min-heap Top-K (same pattern as Phase 2)
  - `OllamaEmbedder` — HTTP POST to local Ollama via `libcurl`
  - `RankingMode` enum — `BM25` | `SEMANTIC` | `HYBRID`
  - Snippet highlighting with ANSI colour codes

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

**Requires:** CMake ≥ 3.20, C++20 compiler, `libcurl` (ships with macOS; on Linux: `sudo apt install libcurl4-openssl-dev`).

## Semantic Search Setup (Optional)

To enable semantic and hybrid modes, install Ollama and pull the embedding model:

```bash
# macOS
brew install ollama
ollama serve                    # start the daemon (keep running)
ollama pull nomic-embed-text    # download the model (~274 MB, one-time)
```

The engine detects Ollama automatically at startup. If Ollama is not running, BM25 mode works as before.

## Usage

Place your `.txt` documents in the `data/` directory.

```bash
./build/search_engine --data ../data
# Skip embedding (BM25 only, faster startup):
./build/search_engine --data ../data --no-embed
```

## Commands

| Command | Description |
|---|---|
| `<query>` | Search with the current mode |
| `:mode bm25\|semantic\|hybrid` | Switch ranking mode |
| `:alpha <float>` | Set BM25 weight for hybrid mode (default 0.5) |
| `:beta <float>` | Set semantic weight for hybrid mode (default 0.5) |
| `:top <K> <query>` | Get the top K results |
| `:explain <query>` | See the BM25 math breakdown for each result |
| `:stats <word>` | See document frequencies for a specific term |
| `:cache` | Display hit/miss rates and current LRU cache contents |
| `:suggest <prefix>` | Up to 5 autocomplete suggestions for a prefix |
| `:embedder` | Show active embedder name and embedded doc count |
| `:analytics` | Session query frequency and average result count |
| `Tab` | Tab-completion while typing (uses Trie) |
| `quit` | Exit |

## Hybrid Scoring Formula

```
FinalScore = α × normalize(BM25) + β × ((cosine_sim + 1) / 2)
```

BM25 scores are **min-max normalised** to `[0,1]` within the candidate set before fusion. Cosine similarity `[-1,1]` is shifted to `[0,1]`. This ensures neither term dominates regardless of `α` / `β` values.
