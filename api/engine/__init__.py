"""
engine/ — Pure-Python search engine package.

Replaces the C++ subprocess architecture. All components run in-process
within the FastAPI server. Import order (dependency DAG):

  document  ← no deps
  tokenizer ← no deps
  inverted_index ← document
  bm25      ← inverted_index, document
  trie      ← no deps
  lru_cache ← no deps
  chunker   ← no deps
  similarity← numpy
  embedder  ← httpx, similarity
  vector_store ← faiss, similarity
  search_engine ← all of the above
"""
