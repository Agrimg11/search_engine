/// @file main.cpp
/// @brief Interactive CLI for the Fast Hybrid Search Engine (Phase 1).

#include "engine/search_engine.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "vendor/linenoise/linenoise.h"

// ── Helpers ─────────────────────────────────────────────────────────────────

static void display_results(const std::vector<search::SearchResult>& results,
                            const search::SearchEngine& engine) {
    if (results.empty()) {
        std::cout << "  No results found.\n";
        return;
    }

    std::cout << "\n  Found " << results.size() << " result(s):\n";
    std::cout << "  " << std::string(54, '-') << "\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& doc = engine.document_store().get_document(results[i].doc_id);

        std::cout << "  " << (i + 1) << ". "
                  << "[score: " << std::fixed << std::setprecision(4)
                  << results[i].score << "]  " << doc.title << "\n";

        // Show a short snippet (first 150 characters).
        std::string snippet = doc.content.substr(0, 150);
        for (auto& c : snippet) { if (c == '\n') c = ' '; }
        std::cout << "     " << snippet << "...\n\n";
    }
}

static void display_cache_stats(const search::SearchEngine& engine) {
    auto s = engine.cache_stats();
    std::cout << "\n  LRU Cache Statistics:\n"
              << "  " << std::string(36, '-') << "\n"
              << "  Capacity : " << s.capacity << " entries\n"
              << "  Size     : " << s.size     << " entries\n"
              << "  Hits     : " << s.hits     << "\n"
              << "  Misses   : " << s.misses   << "\n";
    if (s.hits + s.misses > 0) {
        double ratio = 100.0 * static_cast<double>(s.hits)
                             / static_cast<double>(s.hits + s.misses);
        std::cout << "  Hit rate : " << std::fixed << std::setprecision(1)
                  << ratio << "%\n";
    }
    
    auto queries = engine.cached_queries();
    if (!queries.empty()) {
        std::cout << "  \n  Currently Cached Queries (MRU to LRU):\n";
        for (std::size_t i = 0; i < queries.size(); ++i) {
            std::cout << "  " << (i + 1) << ". [K=" << queries[i].top_k << "] \"";
            for (std::size_t j = 0; j < queries[i].terms.size(); ++j) {
                std::cout << queries[i].terms[j] << (j + 1 == queries[i].terms.size() ? "" : " ");
            }
            std::cout << "\"\n";
        }
    }
    std::cout << "\n";
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Parse arguments:  [--data <dir>] [query]
    std::filesystem::path data_dir = "data";
    std::string cli_query;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else {
            cli_query = argv[i];
        }
    }

    // Banner
    std::cout << "\n"
              << "╔══════════════════════════════════════════════╗\n"
              << "║   Fast Hybrid Search Engine  v2.1           ║\n"
              << "║   Phase 2.2 — LRU Cache + Top-K Heap       ║\n"
              << "╚══════════════════════════════════════════════╝\n\n";

    // Ingest
    search::SearchEngine engine;
    std::cout << "  Loading documents from: " << data_dir << "\n";
    std::size_t count = engine.ingest_directory(data_dir);
    std::cout << "  Indexed " << count << " document(s).\n";

    if (count == 0) {
        std::cerr << "\n  [!] No .txt files found.  "
                     "Place documents in the data/ directory.\n";
        return 1;
    }

    // One-shot query from the command line
    if (!cli_query.empty()) {
        std::cout << "\n  Query: " << cli_query << "\n";
        display_results(engine.search(cli_query), engine);
        return 0;
    }

    // Interactive mode
    std::cout << "\n  Type a search query (or 'quit' to exit):\n"
              << "  [Tip: Type ':stats <word>' to see document frequencies for a specific term]\n"
              << "  [Tip: Type ':top <K> <query>' to retrieve a specific number of results]\n"
              << "  [Tip: Type ':cache' to display LRU cache statistics]\n";
    std::string query;
    linenoiseHistorySetMaxLen(100);

    while (true) {
        std::cout << "\n";
        char* line = linenoise("  > ");
        if (line == nullptr) break;
        
        query = line;
        if (!query.empty()) {
            linenoiseHistoryAdd(line);
        }
        free(line);

        if (query.empty()) continue;
        if (query == "quit" || query == "exit" || query == "q") break;

        if (query.substr(0, 9) == ":explain ") {
            std::string explain_query = query.substr(9);
            auto explanations = engine.explain(explain_query);
            if (explanations.empty()) {
                std::cout << "  No documents found for query.\n";
            } else {
                for (const auto& expl : explanations) {
                    const auto& doc = engine.document_store().get_document(expl.doc_id);
                    std::cout << "\nDocument: " << doc.title << "\n";
                    for (const auto& term_expl : expl.terms) {
                        std::cout << "\nTerm: " << term_expl.term << "\n"
                                  << "    TF  = " << term_expl.tf << "\n"
                                  << "    DF  = " << term_expl.df << "\n"
                                  << "    IDF = " << std::fixed << std::setprecision(4) << term_expl.idf << "\n"
                                  << "    Contribution = " << term_expl.contribution << "\n";
                    }
                    std::cout << "\nDocument length = " << expl.document_length << "\n"
                              << "Average length  = " << expl.average_length << "\n"
                              << "\nFinal BM25 = " << expl.final_score << "\n"
                              << std::string(40, '-') << "\n";
                }
            }
            continue;
        }

        if (query.substr(0, 7) == ":stats ") {
            std::string term = query.substr(7);
            const auto& postings = engine.index().get_postings(term);
            if (postings.empty()) {
                std::cout << "  No statistics found for term: '" << term << "'\n";
            } else {
                std::cout << "  Statistics for term '" << term << "':\n";
                for (const auto& p : postings) {
                    const auto& doc = engine.document_store().get_document(p.doc_id);
                    std::cout << "  - DocID: " << p.doc_id 
                              << " | Freq: " << p.term_frequency 
                              << " | Title: " << doc.title << "\n";
                }
            }
            continue;
        }



        if (query.substr(0, 5) == ":top ") {
            std::size_t space_idx = query.find(' ', 5);
            if (space_idx != std::string::npos) {
                std::string k_str = query.substr(5, space_idx - 5);
                std::string actual_query = query.substr(space_idx + 1);
                try {
                    int k_int = std::stoi(k_str);
                    if (k_int < 0) {
                        throw std::invalid_argument("Negative K");
                    }
                    std::size_t k = static_cast<std::size_t>(k_int);
                    display_results(engine.search(actual_query, k), engine);
                } catch (const std::exception&) {
                    std::cout << "  [!] Invalid K value. Usage: :top <K> <query> (where K >= 0)\n";
                }
            } else {
                std::cout << "  [!] Usage: :top <K> <query>\n";
            }
            continue;
        }

        if (query == ":cache") {
            display_cache_stats(engine);
            continue;
        }

        display_results(engine.search(query), engine);
    }

    std::cout << "\n  Goodbye!\n\n";
    return 0;
}

