"""
routers/rag.py — RAG (Retrieval-Augmented Generation) endpoint.

POST /rag/query
  Body: RAGRequest
  Returns: RAGResponse (LLM-generated answer + source chunks)

Pipeline:
  1. Python engine retrieves top-k relevant chunks
  2. Chunks are assembled into a context window
  3. LangChain sends context + question to Ollama LLM
  4. LLM answer + sources are returned together

Updated for the pure-Python engine:
  - Removed is_alive subprocess check (engine is always alive in-process)
  - LLM inference is still blocking I/O; offloaded to a thread pool executor
    so the FastAPI event loop stays responsive during long Ollama calls
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, HTTPException

from models import RAGRequest, RAGResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
async def rag_query(body: RAGRequest, request: Request) -> RAGResponse:
    """
    Full RAG pipeline: retrieve relevant document chunks from the Python engine,
    then generate a grounded answer using the local Ollama LLM.

    The LLM inference call is blocking I/O; offloaded to a thread pool executor
    to keep the FastAPI event loop responsive.
    """
    pipeline = request.app.state.rag

    loop = asyncio.get_event_loop()
    try:
        response: RAGResponse = await loop.run_in_executor(
            None,
            lambda: pipeline.run(
                question = body.question,
                mode     = body.mode,
                top_k    = body.top_k,
                window   = body.window,
                alpha    = body.alpha,
                beta     = body.beta,
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {e}"
        )

    return response
