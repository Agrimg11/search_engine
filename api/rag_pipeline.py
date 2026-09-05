"""
rag_pipeline.py — LangChain RAG pipeline using chunk-level retrieval.

Flow
────
1. Retrieve:  engine.search_chunks() → top-k ChunkResults (with ±window neighbours)
2. Context:   join chunks with separators, labelled by source doc
3. Generate:  PROMPT | ChatOllama | StrOutputParser → answer
4. Return:    RAGResponse with answer + chunk sources + latency

The engine_client argument now accepts a SearchEngine instance directly
(duck-typing — any object with search_chunks() returning list[ChunkResult]
works). parse_chunks() is no longer needed since the Python engine returns
typed objects, not raw dicts.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import RAGResponse, ChunkResult
import config

_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Your PRIMARY source of truth is the information provided in the context below. 

When answering a question or explaining a topic:
1. Base your answer heavily on the provided context.
2. You may use a small amount of your own internal knowledge to define terms or connect ideas if the context is incomplete, but your answer MUST NOT be solely based on internal knowledge.
3. If the context is completely unrelated and contains absolutely no helpful information, respond with: "I don't have enough information to answer that."

Do not use outside knowledge that contradicts the context.

Context:
{context}

User Input: {question}

Answer:"""
)


class RAGPipeline:
    def __init__(self, engine_client: Any, llm_model: str = config.LLM_MODEL) -> None:
        self._engine    = engine_client
        self._llm_model = llm_model
        self._llm       = ChatOllama(
            model       = llm_model,
            base_url    = config.OLLAMA_BASE_URL,
            temperature = 0.1,
        )
        self._chain = _PROMPT | self._llm | StrOutputParser()

    def run(
        self,
        question: str,
        mode:     str   = config.DEFAULT_MODE,
        top_k:    int   = config.DEFAULT_TOP_K,
        window:   int   = 0,
        alpha:    float = config.DEFAULT_ALPHA,
        beta:     float = config.DEFAULT_BETA,
    ) -> RAGResponse:
        t_start = time.perf_counter()

        # ── Step 1: Retrieve individual chunks (not whole docs) ────────────
        # Python engine returns typed ChunkResult objects directly —
        # no raw-dict parsing needed.
        chunks = self._engine.search_chunks(
            question, top_k=top_k, window=window, alpha=alpha, beta=beta
        )

        # Convert engine.ChunkResult → models.ChunkResult (Pydantic).
        pydantic_chunks = [
            ChunkResult(
                doc_id      = c.doc_id,
                title       = c.title,
                chunk_text  = c.chunk_text,
                chunk_index = c.chunk_index,
                score       = c.score,
            )
            for c in chunks
        ]

        # ── Step 2: Build context — each chunk labelled with its source ────
        if pydantic_chunks:
            context_parts = [
                f"[Source: {c.title}, chunk {c.chunk_index}]\n{c.chunk_text}"
                for c in pydantic_chunks[:config.MAX_CONTEXT_CHUNKS]
            ]
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(No relevant document chunks found.)"

        # ── Step 3: Generate answer ────────────────────────────────────────
        answer = self._chain.invoke({"context": context, "question": question})

        latency_ms = (time.perf_counter() - t_start) * 1000
        return RAGResponse(
            answer     = answer.strip(),
            sources    = pydantic_chunks,
            model      = self._llm_model,
            latency_ms = round(latency_ms, 2),
        )

    @property
    def llm_model(self) -> str:
        return self._llm_model
