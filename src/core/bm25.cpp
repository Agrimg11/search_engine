#include "core/bm25.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <vector>

namespace search {

BM25Scorer::BM25Scorer(const InvertedIndex& index,
                       const DocumentStore&  store,
                       double k1, double b)
    : index_(index), store_(store), k1_(k1), b_(b) {}

// ── IDF ─────────────────────────────────────────────────────────────────────
// Standard BM25 IDF with the "+1" floor to avoid negative values when
// a term appears in more than half the corpus.
double BM25Scorer::idf(const std::string& term) const {
    auto N  = static_cast<double>(index_.total_documents());
    auto df = static_cast<double>(index_.document_frequency(term));
    if (df == 0.0) return 0.0;
    return std::log((N - df + 0.5) / (df + 0.5) + 1.0);
}

// ── Score a single document ─────────────────────────────────────────────────
double BM25Scorer::score_document(const std::vector<std::string>& query_tokens,
                                   uint32_t doc_id) const {
    double avgdl = store_.average_document_length();
    if (avgdl == 0.0) return 0.0;

    double dl    = static_cast<double>(store_.get_document(doc_id).token_count);
    double score = 0.0;

    for (const auto& term : query_tokens) {
        double idf_val = idf(term);
        if (idf_val <= 0.0) continue;

        // Linear scan of the posting list for this term to find doc_id.
        // Acceptable for Phase 1; Phase 2 will use the search() path
        // which iterates postings directly and avoids repeated lookups.
        uint32_t tf = 0;
        for (const auto& p : index_.get_postings(term)) {
            if (p.doc_id == doc_id) { tf = p.term_frequency; break; }
        }
        if (tf == 0) continue;

        double tf_d   = static_cast<double>(tf);
        double numer  = tf_d * (k1_ + 1.0);
        double denom  = tf_d + k1_ * (1.0 - b_ + b_ * dl / avgdl);
        score        += idf_val * (numer / denom);
    }
    return score;
}

// ── Corpus-wide search (Phase 2: O(R log K) min-heap Top-K) ─────────────────
// Score accumulation is identical to Phase 1: iterate each query term's
// posting list and accumulate per-doc BM25 scores in a hash map — O(Q·P).
//
// Extraction changed: instead of sorting all R candidates (O(R log R)),
// we push candidates into a min-heap of capacity K.  At every step the
// heap root is the *weakest* document in our current Top-K candidates.
//
//   • If a new doc's score > root → evict root, push new doc  O(log K)
//   • Otherwise discard the new doc immediately               O(1)
//
// Total extraction cost: O(R log K).  When K << R this is a large win.
std::vector<SearchResult> BM25Scorer::search(
    const std::vector<std::string>& query_tokens,
    std::size_t top_k) const {

    if (query_tokens.empty() || index_.total_documents() == 0) return {};
    if (top_k == 0) return {};

    double avgdl = store_.average_document_length();
    if (avgdl == 0.0) return {};

    // ── Stage 1: Accumulate BM25 scores (unchanged from Phase 1) ───────────
    std::unordered_map<uint32_t, double> scores;

    for (const auto& term : query_tokens) {
        double idf_val = idf(term);
        if (idf_val <= 0.0) continue;

        for (const auto& posting : index_.get_postings(term)) {
            double dl    = static_cast<double>(
                store_.get_document(posting.doc_id).token_count);
            double tf_d  = static_cast<double>(posting.term_frequency);
            double numer = tf_d * (k1_ + 1.0);
            double denom = tf_d + k1_ * (1.0 - b_ + b_ * dl / avgdl);

            scores[posting.doc_id] += idf_val * (numer / denom);
        }
    }

    if (scores.empty()) return {};

    // ── Stage 2: Min-heap Top-K extraction — O(R log K) ────────────────────
    //
    // Min-heap comparator: top of heap = document with the LOWEST score.
    // We want to keep the K documents with the HIGHEST scores, so whenever
    // a new candidate beats the weakest element in the heap (the root),
    // we evict the root and insert the new candidate.
    //
    //   Heap invariant:  root.score == min(score of all K elements in heap)
    //
    using MinHeap = std::priority_queue<
        SearchResult,
        std::vector<SearchResult>,
        // Greater-than on score → smallest score at the top (min-heap)
        decltype([](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;   // a has higher priority when score is larger
                                        // → lowest score bubbles to top
        })>;

    MinHeap heap;

    for (auto& [doc_id, s] : scores) {
        if (heap.size() < top_k) {
            // Heap not full yet — just push.
            heap.push({doc_id, s});
        } else if (s > heap.top().score) {
            // New doc beats the current weakest Top-K member → swap it in.
            heap.pop();
            heap.push({doc_id, s});
        }
        // else: new doc is worse than every element already in the heap → discard.
    }

    // ── Stage 3: Drain heap and reverse to get descending order ─────────────
    // The heap yields elements in ascending score order (min first),
    // so we collect them and then reverse.
    std::vector<SearchResult> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    // Heap gave us ascending order; reverse → descending (best score first).
    std::reverse(results.begin(), results.end());
    return results;
}

// ── Corpus-wide search with detailed explanation ────────────────────────────
std::vector<DocumentExplanation> BM25Scorer::explain(
    const std::vector<std::string>& query_tokens,
    std::size_t top_k) const {

    if (query_tokens.empty() || index_.total_documents() == 0) return {};

    double avgdl = store_.average_document_length();
    if (avgdl == 0.0) return {};

    std::unordered_map<uint32_t, DocumentExplanation> docs;

    for (const auto& term : query_tokens) {
        double idf_val = idf(term);
        if (idf_val <= 0.0) continue;
        uint32_t df = index_.document_frequency(term);

        for (const auto& posting : index_.get_postings(term)) {
            uint32_t doc_id = posting.doc_id;
            if (docs.find(doc_id) == docs.end()) {
                docs[doc_id].doc_id = doc_id;
                docs[doc_id].final_score = 0.0;
                docs[doc_id].document_length = store_.get_document(doc_id).token_count;
                docs[doc_id].average_length = avgdl;
            }

            double dl    = static_cast<double>(docs[doc_id].document_length);
            double tf_d  = static_cast<double>(posting.term_frequency);
            double numer = tf_d * (k1_ + 1.0);
            double denom = tf_d + k1_ * (1.0 - b_ + b_ * dl / avgdl);
            double contrib = idf_val * (numer / denom);

            docs[doc_id].final_score += contrib;
            docs[doc_id].terms.push_back({term, posting.term_frequency, df, idf_val, contrib});
        }
    }

    std::vector<DocumentExplanation> results;
    results.reserve(docs.size());
    for (auto& [doc_id, explanation] : docs) {
        results.push_back(std::move(explanation));
    }

    std::sort(results.begin(), results.end(),
              [](const DocumentExplanation& a, const DocumentExplanation& b) {
                  return a.final_score > b.final_score;
              });

    if (results.size() > top_k) {
        results.resize(top_k);
    }
    return results;
}

} // namespace search
