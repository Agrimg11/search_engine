# Fast Hybrid C++ Search Engine

A high-performance, in-memory text search engine built in C++. This project implements an interactive CLI search tool using a hybrid architecture that combines a custom inverted index, Okapi BM25 relevance scoring, a Top-K Min-Heap, and an LRU Cache.

## Features

- **Okapi BM25 Scoring:** Mathematically robust relevance scoring for document retrieval.
- **Inverted Index:** Fast $O(1)$ term lookups mapping words to documents.
- **LRU Cache:** $O(1)$ query caching using a doubly-linked list and hash map for instant retrieval of repeated queries.
- **Top-K Min-Heap:** $O(R \log K)$ optimization for ranking results, efficiently extracting the best matches without sorting the entire dataset.
- **Interactive CLI:** A responsive command-line interface with history tracking.

## Architecture & Phases

- **Phase 1:** Basic Tokenization, Inverted Index, and BM25 Scoring.
- **Phase 2.2 (Current):** Introduced an LRU Cache to bypass heavy computations for identical queries, and a Min-Heap for highly optimized Top-K extraction.

## Build Instructions

This project uses CMake. To build the engine:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Place your `.txt` documents in the `data/` directory.

To run the interactive CLI:
```bash
./build/search_engine --data ../data
```

Once in the interactive prompt, you can type queries or use special commands:
- `:top <K> <query>`: Get the top K results.
- `:stats <word>`: See document frequencies for a specific term.
- `:explain <query>`: See the BM25 math breakdown for each result.
- `:cache`: Display hit/miss rates and current LRU cache statistics.
