cmake_minimum_required(VERSION 3.20)
project(FastHybridSearch VERSION 1.0.0 LANGUAGES CXX C)

# ── C++20 ────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Core library (shared between main binary and tests) ─────────────────────
add_library(search_lib
    src/core/document.cpp
    src/core/tokenizer.cpp
    src/core/inverted_index.cpp
    src/core/bm25.cpp
    src/core/trie.cpp
    src/engine/search_engine.cpp
    src/vendor/linenoise/linenoise.c
)
target_include_directories(search_lib PUBLIC src)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(search_engine src/main.cpp)
target_link_libraries(search_engine PRIVATE search_lib)

# ── Tests (GoogleTest via FetchContent) ─────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    # Helper: add a test target in one line
    function(add_search_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE search_lib GTest::gtest_main)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_search_test(test_tokenizer      tests/test_tokenizer.cpp)
    add_search_test(test_inverted_index  tests/test_inverted_index.cpp)
    add_search_test(test_bm25           tests/test_bm25.cpp)
    add_search_test(test_topk_heap      tests/test_topk_heap.cpp)  # Phase 2.1
    add_search_test(test_lru_cache      tests/test_lru_cache.cpp)  # Phase 2.2
    add_search_test(test_trie           tests/test_trie.cpp)        # Phase 2.3
endif()
