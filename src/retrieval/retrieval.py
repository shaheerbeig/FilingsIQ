"""Data model for a retrieval result.

A RetrievalResult bundles the user's query with the ranked chunks the vector
store returned for it. The hits are SearchResult objects (defined in the storage
layer) — each carries the chunk's text, metadata, and similarity score, which is
everything a downstream generator needs to answer and cite.

Re-exporting SearchResult here lets callers depend on the retrieval package
alone, without reaching into storage internals.
"""
from typing import Optional

from pydantic import BaseModel

from src.storage.vector_store import SearchResult

__all__ = ["SearchResult", "RetrievalResult"]


class RetrievalResult(BaseModel):
    """A query plus the chunks retrieved for it, best first."""
    query: str
    results: list[SearchResult]

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def top(self) -> Optional[SearchResult]:
        """The single best hit, or None if nothing was retrieved."""
        return self.results[0] if self.results else None
