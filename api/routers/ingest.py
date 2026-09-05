"""
routers/ingest.py — Document upload and ingestion endpoint.

POST /ingest
  Body: multipart/form-data with a single `file` field (.txt or .pdf)
  Returns: IngestResponse (doc_id, title, updated engine stats)

Flow:
  1. Validate file extension (.txt or .pdf only) and size (≤ 5 MB)
  2. Extract text:
       .txt → decode bytes as UTF-8 (with replacement for invalid bytes)
       .pdf → pypdf PdfReader to extract text from all pages
  3. Guard against empty-text PDFs (scanned/image-only — no OCR support)
  4. Call engine.ingest(title, content) — reuses the existing ingestion pipeline
       (Tokenizer → InvertedIndex → Trie → VectorStore if embedder is attached)
  5. Return IngestResponse with doc_id and updated corpus stats
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Request, HTTPException, UploadFile, File

from models import IngestResponse

router = APIRouter(tags=["ingest"])

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_BYTES      = 5 * 1024 * 1024          # 5 MB
_ALLOWED_EXTS   = {".txt", ".pdf"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_txt(raw: bytes) -> str:
    """Decode raw bytes as UTF-8 (replace invalid sequences)."""
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes) -> str:
    """Extract all text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pypdf is not installed. Run: pip install pypdf",
        )

    reader = PdfReader(io.BytesIO(raw))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="A .txt or .pdf document to index."),
) -> IngestResponse:
    """
    Upload a .txt or .pdf document and add it to the live search index.

    The document becomes immediately searchable via POST /search and
    POST /rag/query once this endpoint returns successfully.
    """
    engine = request.app.state.engine

    # ── 1. Validate extension ─────────────────────────────────────────────────
    filename = file.filename or "untitled"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Only .txt and .pdf are accepted.",
        )

    # ── 2. Read and validate size ─────────────────────────────────────────────
    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) // 1024} KB). Maximum allowed is 5 MB.",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Extract text ───────────────────────────────────────────────────────
    if ext == ".txt":
        content = _extract_txt(raw)
    else:  # .pdf
        try:
            content = _extract_pdf(raw)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse PDF: {exc}",
            )

        # Guard: scanned/image-only PDFs produce no text.
        if not content.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned/image-only PDF, which requires OCR "
                    "(not supported)."
                ),
            )

    if not content.strip():
        raise HTTPException(status_code=422, detail="Document contains no readable text.")

    # ── 4. Ingest into the existing search engine ─────────────────────────────
    title  = filename.rsplit(".", 1)[0]
    doc_id = engine.ingest(title, content)

    # ── 5. Return updated stats ───────────────────────────────────────────────
    return IngestResponse(
        doc_id      = doc_id,
        title       = title,
        doc_count   = engine.document_count,
        chunk_count = engine.embedded_chunk_count,
    )
