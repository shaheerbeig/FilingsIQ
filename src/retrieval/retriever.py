"""Retriever — turns a natural-language question into ranked chunks.

This is the read side of the pipeline. Given a query string it:
  1. embeds the query with the SAME model/dimensions used to embed the chunks
     (a query and a chunk must live in the same vector space to be comparable),
  2. asks the vector store for the nearest chunks,
  3. optionally drops hits below a similarity threshold,
and returns a RetrievalResult.

The query is embedded raw — no section_path breadcrumb is prepended (unlike
chunks), because a question has no place in the document outline.

The store and OpenAI client are created once and reused across calls. Both can
be injected (used by tests, or to share one client across the app).
"""
from typing import Optional

from loguru import logger
from openai import OpenAI

from src.common.config import EmbeddingConfig, RetrievalConfig, get_settings
from src.retrieval.retrieval import RetrievalResult
from src.storage.vector_store import VectorStore


class Retriever:
    """Embeds queries and searches the vector store for the nearest chunks."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        store: Optional[VectorStore] = None,
        client: Optional[OpenAI] = None,
    ):
        settings = get_settings()
        self.config = config or settings.retrieval
        # Query embeddings must match the chunk embeddings exactly, so reuse the
        # embedding config (same model + dimensions) rather than a separate one.
        self.embedding_config = embedding_config or settings.embedding
        self.store = store or VectorStore()
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            max_retries=self.embedding_config.max_retries,
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """Return the chunks most similar to a query, best first.

        k falls back to config.top_k. Hits below config.min_score are dropped.
        """
        k = k if k is not None else self.config.top_k
        query_vec = self._embed_query(query)

        hits = self.store.search(query_vec, k=k)
        if self.config.min_score > 0:
            kept = [h for h in hits if h.score >= self.config.min_score]
            if len(kept) < len(hits):
                logger.debug(
                    f"min_score={self.config.min_score} dropped "
                    f"{len(hits) - len(kept)}/{len(hits)} hits"
                )
            hits = kept

        logger.info(f"Retrieved {len(hits)} chunks for query: {query!r}")
        return RetrievalResult(query=query, results=hits)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a vector."""
        response = self.client.embeddings.create(
            model=self.embedding_config.model,
            input=[query],
            dimensions=self.embedding_config.dimensions,
        )
        return response.data[0].embedding
