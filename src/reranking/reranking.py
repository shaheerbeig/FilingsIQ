"""Data model for a reranking result.

Reranking takes the chunks retrieval already found and re-scores them with a
model that reads the question and each chunk together. It does not fetch new
chunks — every RerankedChunk wraps one of the SearchResults retrieval returned.

A RerankedChunk keeps BOTH scores: the original cosine similarity (on the
wrapped SearchResult) and the new rerank_score, so you can see how the order
changed. A RerankResult bundles the query with the re-sorted, trimmed list.
"""
from typing import Optional

from pydantic import BaseModel

from src.storage.vector_store import SearchResult

__all__ = ["RerankedChunk", "RerankResult"]


class RerankedChunk(BaseModel):
    """One retrieved chunk plus the reranker's relevance score for it."""
    result: SearchResult        # the original retrieval hit (text, metadata, cosine score)
    rerank_score: float         # reranker's "how well does this answer the query", higher = better


class RerankResult(BaseModel):
    """A query plus its reranked chunks, best first, trimmed to top_n."""
    query: str
    results: list[RerankedChunk]

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def top(self) -> Optional[RerankedChunk]:
        return self.results[0] if self.results else None
