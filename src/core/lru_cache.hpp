#pragma once
/// @file lru_cache.hpp
/// @brief Generic, capacity-bounded Least Recently Used (LRU) cache.
///
/// Design — O(1) for every operation:
///
///   Recency list  : std::list<pair<Key, Value>>
///                   MRU node at front(), LRU node at back().
///
///   Lookup table  : std::unordered_map<Key, list::iterator>
///                   Maps each key to its node in the list for O(1) access.
///
/// On a cache HIT  → node is spliced to the front (O(1) list splice).
/// On a cache MISS (not full) → new node pushed to front.
/// On a cache MISS (full)    → back node evicted, new node pushed to front.
///
/// Special case: capacity == 0 → get() always returns nullopt, put() is no-op.
///
/// Thread safety: NOT thread-safe.  Single-threaded use only.

#include <cstddef>
#include <functional>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>

namespace search {

/// @tparam Key    Key type.  Must be hashable via KeyHash and equality-comparable.
/// @tparam Value  Value type.  Must be copyable (copies are returned to callers).
/// @tparam KeyHash  Hash functor for Key.  Defaults to std::hash<Key>.
/// @tparam KeyEqual Equality functor for Key.  Defaults to std::equal_to<Key>.
template <
    typename Key,
    typename Value,
    typename KeyHash  = std::hash<Key>,
    typename KeyEqual = std::equal_to<Key>
>
class LRUCache {
public:
    // ── Types ────────────────────────────────────────────────────────────────
    using key_type    = Key;
    using mapped_type = Value;
    using entry_type  = std::pair<Key, Value>;
    using list_type   = std::list<entry_type>;
    using iterator_type = typename list_type::iterator;
    using map_type    = std::unordered_map<Key, iterator_type, KeyHash, KeyEqual>;

    // ── Construction ─────────────────────────────────────────────────────────
    /// @param capacity  Maximum number of entries.  0 means disabled (always miss).
    explicit LRUCache(std::size_t capacity = 100) : capacity_(capacity) {}

    // Non-copyable to avoid accidental expensive copies of the entire cache.
    LRUCache(const LRUCache&)            = delete;
    LRUCache& operator=(const LRUCache&) = delete;

    // Movable.
    LRUCache(LRUCache&&)            = default;
    LRUCache& operator=(LRUCache&&) = default;

    // ── Core API ─────────────────────────────────────────────────────────────

    /// Look up a key.
    ///
    /// @returns  A copy of the cached value if key is present (HIT),
    ///           std::nullopt otherwise (MISS).
    ///
    /// On a HIT the node is promoted to the MRU position (front of list).
    /// Time: O(1) average.
    [[nodiscard]] std::optional<Value> get(const Key& key) {
        if (capacity_ == 0) {
            ++misses_;
            return std::nullopt;
        }

        auto it = map_.find(key);
        if (it == map_.end()) {
            ++misses_;
            return std::nullopt;
        }

        // Promote to MRU (front).
        list_.splice(list_.begin(), list_, it->second);
        ++hits_;
        return it->second->second;   // return copy of value
    }

    /// Insert or update a key-value pair.
    ///
    /// If the key already exists its value is updated in-place and the node
    /// is promoted to MRU.  If the cache is at capacity the LRU entry (back
    /// of list) is evicted first.  If capacity == 0 the call is a no-op.
    ///
    /// Time: O(1) average.
    void put(const Key& key, const Value& value) {
        if (capacity_ == 0) return;

        auto it = map_.find(key);
        if (it != map_.end()) {
            // Update existing entry and promote to MRU.
            it->second->second = value;
            list_.splice(list_.begin(), list_, it->second);
            return;
        }

        // Evict LRU if at capacity.
        if (list_.size() >= capacity_) {
            map_.erase(list_.back().first);
            list_.pop_back();
        }

        // Insert new entry at MRU position.
        list_.emplace_front(key, value);
        map_[key] = list_.begin();
    }

    /// Check whether the cache contains key WITHOUT updating recency order or
    /// incrementing hit/miss counters.
    ///
    /// Time: O(1) average.
    [[nodiscard]] bool contains(const Key& key) const {
        if (capacity_ == 0) return false;
        return map_.count(key) > 0;
    }

    /// Remove all entries.  Does NOT reset hit/miss statistics.
    void clear() {
        list_.clear();
        map_.clear();
    }

    // ── Capacity / Size ───────────────────────────────────────────────────────
    [[nodiscard]] std::size_t size()     const noexcept { return list_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_;    }
    [[nodiscard]] bool        empty()    const noexcept { return list_.empty(); }

    /// Change the maximum capacity.
    ///
    /// If new_cap < current size, LRU entries are evicted until size == new_cap.
    void set_capacity(std::size_t new_cap) {
        capacity_ = new_cap;
        while (new_cap > 0 && list_.size() > new_cap) {
            map_.erase(list_.back().first);
            list_.pop_back();
        }
        if (new_cap == 0) {
            list_.clear();
            map_.clear();
        }
    }

    /// Return all keys currently in the cache, ordered from MRU (front) to LRU (back).
    [[nodiscard]] std::vector<Key> get_keys() const {
        std::vector<Key> keys;
        keys.reserve(list_.size());
        for (const auto& pair : list_) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    // ── Statistics ────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t hits()   const noexcept { return hits_;   }
    [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

    /// Reset hit/miss counters to zero.
    void reset_stats() noexcept { hits_ = 0; misses_ = 0; }

private:
    std::size_t capacity_;
    list_type   list_;   ///< Recency order; list_.front() is MRU, back() is LRU.
    map_type    map_;    ///< Key → iterator into list_.

    std::size_t hits_   = 0;
    std::size_t misses_ = 0;
};

} // namespace search
