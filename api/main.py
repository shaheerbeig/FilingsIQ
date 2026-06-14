"""FastAPI app — serves the investor Q&A frontend and the /api/ask endpoint.

The RAG pipeline is built once at startup (it opens the Chroma connection and
the OpenAI clients), then reused for every request. The static frontend is
served from api/static at the root path, so the whole thing runs on one origin
(no CORS needed).

Run from the project root:
    uvicorn api.main:app --reload
then open http://127.0.0.1:8000
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.logging import configure_logging
from src.pipeline import RAGPipeline

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="10-K Investor Q&A", version="1.0")

# Built lazily on first request so import (and tests) don't require a live index.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        configure_logging(level=settings.logging.level, to_file=settings.logging.to_file)
        logger.info("Building RAG pipeline...")
        _pipeline = RAGPipeline()
        logger.info(f"Pipeline ready ({_pipeline.retriever.store.count} vectors indexed)")
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class Source(BaseModel):
    number: int
    chunk_id: str
    section: str
    pages: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str


@app.get("/api/health")
def health() -> dict:
    """Lightweight check that the index is loaded and how many vectors it holds."""
    return {"status": "ok", "vectors": get_pipeline().retriever.store.count}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer one investor question against the indexed 10-K."""
    result = get_pipeline().answer(req.question)
    sources = [
        Source(
            number=s.number,
            chunk_id=s.chunk_id,
            section=" > ".join(s.section_path) if s.section_path else "(no section)",
            pages=f"p{s.first_page}-{s.last_page}",
        )
        for s in result.sources
    ]
    return AskResponse(
        question=result.query, answer=result.answer, sources=sources, model=result.model
    )


# Serve the frontend at the root. Mounted last so /api/* routes win.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
