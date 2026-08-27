#include "engine/search_engine.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace search {

// ── Construction ─────────────────────────────────────────────────────────────
SearchEngine::SearchEngine(std::size_t cache_capacity)
    : scorer_(index_, store_), cache_(cache_capacity) {}
    // scorer_ stores *references* to index_ and store_, both of which are
    // already constructed (C++ initialises members in declaration order).

// ── Single-document ingestion ─────────────────────────────────────────────────
uint32_t SearchEngine::ingest(const std::string& title, const std::string& content) {
    uint32_t doc_id = store_.add_document(title, content);
    auto tokens     = tokenizer_.tokenize(content);
    store_.update_token_count(doc_id, static_cast<uint32_t>(tokens.size()));
    index_.add_document(doc_id, tokens);

    // The index has changed — any cached search results may now be stale.
    // Clear the cache so the next query always reflects the current corpus.
    cache_.clear();

    return doc_id;
}

// ── Directory ingestion ───────────────────────────────────────────────────────
std::size_t SearchEngine::ingest_directory(const std::filesystem::path& dir_path) {
    if (!std::filesystem::exists(dir_path)) {
        std::cerr << "[error] Directory not found: " << dir_path << "\n";
        return 0;
    }

    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".txt") {
            continue;
        }
        std::ifstream file(entry.path());
        if (!file.is_open()) continue;

        std::ostringstream buf;
        buf << file.rdbuf();

        // ingest() calls cache_.clear() internally after each document.
        ingest(entry.path().stem().string(), buf.str());
        ++count;
    }
    return count;
}

// ── Search (with LRU cache) ───────────────────────────────────────────────────
//
// Flow:
//   1. Tokenise query.
//   2. Build a QueryKey(sorted_tokens, top_k).
//   3. Cache lookup.
//       HIT  → return cached copy immediately (BM25 skipped entirely).
//       MISS → run BM25 + Top-K heap, store result in cache, then return.
//
// search() is const-qualified because the cache_ member is mutable.
// Observable result is identical whether served from cache or computed fresh.

std::vector<SearchResult> SearchEngine::search(const std::string& query,
                                                std::size_t top_k) const {
    auto tokens = tokenizer_.tokenize(query);

    // Build the cache key (constructor sorts tokens for order-independence).
    QueryKey key(tokens, top_k);

    // ── Cache lookup ───────────────────────────────────────────────────
    if (auto cached = cache_.get(key)) {
        // Cache HIT: BM25 scoring is NOT executed.
        return *cached;
    }

    // ── Cache MISS: run the full BM25 + Top-K heap pipeline ──────────
    auto results = scorer_.search(tokens, top_k);

    // Store a copy in the cache.  Subsequent identical queries hit the cache.
    cache_.put(key, results);

    return results;
}

// ── Explain ──────────────────────────────────────────────────────────────────
std::vector<DocumentExplanation> SearchEngine::explain(const std::string& query,
                                                       std::size_t top_k) const {
    auto tokens = tokenizer_.tokenize(query);
    return scorer_.explain(tokens, top_k);
}

// ── Cache management ──────────────────────────────────────────────────────────

CacheStats SearchEngine::cache_stats() const noexcept {
    return CacheStats{
        cache_.capacity(),
        cache_.size(),
        cache_.hits(),
        cache_.misses()
    };
}

void SearchEngine::clear_cache() {
    cache_.clear();
}

void SearchEngine::set_cache_capacity(std::size_t new_cap) {
    cache_.set_capacity(new_cap);
}

// ── Individual cache stat accessors ──────────────────────────────────────────

std::size_t SearchEngine::cache_capacity() const noexcept { return cache_.capacity(); }

} // namespace search
