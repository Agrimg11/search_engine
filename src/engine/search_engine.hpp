#pragma once
/// @file search_engine.hpp
/// @brief High-level façade that ties together the document store,
///        tokenizer, inverted index, BM25 scorer, and LRU search cache.
///
/// Phase 2.2 addition: an in-memory LRU cache keyed on (normalised query
/// terms, top_k).  On a cache HIT the BM25 + heap pipeline is bypassed
/// entirely; results are returned in O(1) average time.
///
/// Phase 2.3 addition: a Trie (prefix tree) built from the inverted index
/// vocabulary.  It powers the autocomplete / suggest feature: given a prefix,
/// it returns up to `limit` matching terms ranked by document frequency.
///
/// Users interact only with SearchEngine; the internal components are
/// hidden behind a clean public API.

#include "core/bm25.hpp"
#include "core/document.hpp"
#include "core/inverted_index.hpp"
#include "core/lru_cache.hpp"
#include "core/tokenizer.hpp"
#include "core/trie.hpp"

#include <algorithm>   // std::sort
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>  // std::hash
#include <string>
#include <vector>

namespace search {

// ── Cache key ──────────────────────────────────────────────────────────────
// The key encodes both the normalised query terms and the requested top_k
// value, so ":top 1 algo" and ":top 10 algo" are never confused.
//
// Terms are stored in sorted order: BM25 is a bag-of-words model, so
// "algorithm data" and "data algorithm" yield identical results.  Sorting
// the terms in the key lets both queries share a single cache entry without
// changing search semantics (BM25 is called with the original token order).
//
// Normalisation (lowercasing, stop-word removal) is already performed by
// Tokenizer::tokenize() before this key is constructed — no extra work here.

struct QueryKey {
    std::vector<std::string> terms;   ///< Sorted, normalised query tokens.
    std::size_t              top_k;

    /// Construct key from a tokenized query.  Terms are sorted in-place so
    /// the key is independent of input token order.
    QueryKey(std::vector<std::string> t, std::size_t k)
        : terms(std::move(t)), top_k(k) {
        std::sort(terms.begin(), terms.end());
    }

    bool operator==(const QueryKey& other) const noexcept {
        return top_k == other.top_k && terms == other.terms;
    }
};

/// Boost-style hash combiner for QueryKey.
struct QueryKeyHash {
    std::size_t operator()(const QueryKey& key) const noexcept {
        // Seed with top_k so (terms=["x"], K=1) != (terms=["x"], K=10).
        std::size_t seed = std::hash<std::size_t>{}(key.top_k);
        for (const auto& term : key.terms) {
            // Boost hash_combine pattern: avalanche each term's hash into seed.
            seed ^= std::hash<std::string>{}(term)
                    + 0x9e3779b9ull + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// ── Cache statistics ─────────────────────────────────────────────────────────
struct CacheStats {
    std::size_t capacity;
    std::size_t size;
    std::size_t hits;
    std::size_t misses;
};

// ── SearchEngine ──────────────────────────────────────────────────────────────
class SearchEngine {
public:
    /// @param cache_capacity  Maximum number of cached query results.
    ///                        0 disables caching entirely.
    explicit SearchEngine(std::size_t cache_capacity = 100);

    // ── Ingestion ───────────────────────────────────────────────────────
    /// Ingest a single document.  Returns the assigned document ID.
    /// Clears the result cache (index has changed; cached results would be stale).
    uint32_t ingest(const std::string& title, const std::string& content);

    /// Ingest every .txt file in `dir_path`.  Returns the number of files loaded.
    /// Clears the result cache after the directory is fully ingested.
    std::size_t ingest_directory(const std::filesystem::path& dir_path);

    // ── Search ──────────────────────────────────────────────────────────
    /// Run a keyword search and return the top-k results.
    ///
    /// Cache HIT  → returns a copy of the cached result vector in O(K).
    /// Cache MISS → runs BM25 + Top-K heap, stores the result, then returns.
    [[nodiscard]] std::vector<SearchResult> search(const std::string& query,
                                                    std::size_t top_k = 10) const;

    /// Run a keyword search and return detailed explanations for the top-k results.
    [[nodiscard]] std::vector<DocumentExplanation> explain(const std::string& query,
                                                           std::size_t top_k = 10) const;

    // ── Cache management ────────────────────────────────────────────────
    /// Read-only snapshot of current cache statistics.
    [[nodiscard]] CacheStats cache_stats() const noexcept;

    /// Convenience accessors for common statistics (avoid struct decomposition).
    [[nodiscard]] std::size_t cache_hits()   const noexcept { return cache_.hits();   }
    [[nodiscard]] std::size_t cache_misses() const noexcept { return cache_.misses(); }
    [[nodiscard]] std::size_t cache_size()   const noexcept { return cache_.size();   }

    /// Return all currently cached query keys, ordered from most- to least-recently used.
    [[nodiscard]] std::vector<QueryKey> cached_queries() const {
        return cache_.get_keys();
    }

    /// Evict all cached entries.
    /// Hit/miss counters are NOT reset; call cache_.reset_stats() if needed.
    void clear_cache();

    /// Change the cache capacity.  If new_cap < current size, LRU entries
    /// are evicted.  Passing 0 disables caching and clears all entries.
    void set_cache_capacity(std::size_t new_cap);

    // ── Suggest / Autocomplete ───────────────────────────────────────────
    /// Return up to `limit` vocabulary terms starting with `prefix`,
    /// ranked by document frequency (descending) then lexicographically.
    ///
    /// Delegates to the internal Trie built from the inverted index vocabulary.
    /// Returns an empty vector if the prefix is not found, contains invalid
    /// characters, or limit == 0.
    [[nodiscard]] std::vector<std::pair<std::string, std::size_t>>
    suggest(const std::string& prefix, std::size_t limit = 10) const;

    // ── Individual cache stat accessors ─────────────────────────────────
    [[nodiscard]] std::size_t cache_capacity() const noexcept;

    // ── Accessors ───────────────────────────────────────────────────────
    [[nodiscard]] const DocumentStore&  document_store() const noexcept { return store_; }
    [[nodiscard]] const InvertedIndex&  index()          const noexcept { return index_; }
    [[nodiscard]] std::size_t           document_count() const noexcept { return store_.size(); }

private:
    // Declaration order matters: scorer_ holds references to index_ and
    // store_, so they must be constructed first (C++ initialises members
    // in declaration order).
    DocumentStore store_;
    Tokenizer     tokenizer_;
    InvertedIndex index_;
    BM25Scorer    scorer_;

    // ── LRU cache ────────────────────────────────────────────────────────
    // `mutable` so search() (logically const — same inputs → same outputs)
    // can update the cache without removing its const qualifier.  This
    // preserves the existing public API; callers may hold const references.
    //
    // Key   : QueryKey { sorted normalised tokens, top_k }
    // Value : vector<SearchResult> — a copy of the final ranked list
    mutable LRUCache<QueryKey, std::vector<SearchResult>, QueryKeyHash> cache_;

    // ── Trie (autocomplete) ──────────────────────────────────────────────
    // Built from the inverted index vocabulary after each full ingestion.
    // Not mutable: suggest() is a const read operation; the Trie is only
    // modified in non-const ingestion methods.
    Trie vocab_trie_;

    /// Rebuild the Trie from the current inverted index vocabulary.
    /// Called internally after every directory ingestion (once, not per-file)
    /// and after every single-document ingestion to keep the Trie in sync.
    void rebuild_trie();
};

} // namespace search
