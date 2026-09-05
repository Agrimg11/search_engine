# Hybrid Search Engine with RAG (Python + FastAPI)

A fast, hybrid in-memory search engine and Retrieval-Augmented Generation (RAG) gateway powered by Python, FastAPI, and local LLMs via Ollama. 

The engine has been completely written in pure Python to seamlessly integrate with FAISS, LangChain, and a modern web frontend.

## Features

- **Hybrid Search**: Fuses Okapi BM25 keyword scoring with dense vector semantic search powered by FAISS (HNSW index).
- **RAG Pipeline**: Leverages LangChain and local Ollama models (e.g., Llama 3.2) to generate accurate, document-grounded answers based on retrieved context.
- **FastAPI Gateway**: Provides a robust, async REST API for searching, document ingestion, and RAG requests.
- **In-Memory Engine**: Pure-Python implementation eliminating the need for C++ subprocesses. 
- **Modern Web UI**: A sleek, dark-themed interactive HTML/CSS frontend served directly by FastAPI.
- **Document Ingestion**: Supports uploading `.txt` and `.pdf` files, automatic chunking, and embedding generation.

## Architecture

- **Backend**: FastAPI (`api/main.py`)
- **Search Engine**: Pure-Python BM25 + FAISS
- **Embeddings**: Local Ollama (`nomic-embed-text`)
- **LLM / Generation**: Local Ollama (e.g., `llama3.2`) via LangChain (`api/rag_pipeline.py`)
- **Frontend**: Vanilla HTML/JS/CSS (`api/static/index.html`)

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (Highly recommended for semantic search and RAG capabilities)

## Installation & Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r api/requirements.txt
   ```

2. **Set up Ollama (Local LLM & Embeddings):**
   Ensure Ollama is running in the background. Pull the required models (you can configure models in `api/config.py`):
   ```bash
   ollama pull llama3.2            # For RAG generation
   ollama pull nomic-embed-text    # For semantic search embeddings
   ```

## Usage

1. **Start the FastAPI server:**
   ```bash
   cd api
   uvicorn main:app --reload --port 8000
   ```
   
2. **Access the Web UI:**
   Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

3. **Ingesting Documents:**
   - Place `.txt` files in the `data/` directory before startup to auto-ingest them.
   - Alternatively, use the "Documents" tab in the Web UI to upload files interactively.

4. **Searching:**
   - Use the "Search" tab for standard keyword, semantic, or hybrid search.
   - Use the "Ask AI" tab to ask natural language questions and get RAG-powered answers grounded in your documents.

## Testing

Run the pytest suite to ensure the API and engine are functioning correctly:
```bash
cd api
pytest
```
