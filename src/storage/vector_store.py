"""Vector store — persists chunk embeddings in Chroma and searches them.

Wraps a local, on-disk Chroma collection. index() loads an EmbeddedDocument's
vectors (plus the chunk text and metadata) into the collection; search() takes a
query vector and returns the nearest chunks with similarity scores.

Chroma metadata values must be scalars (str/int/float/bool), so list fields like
section_path and source_element_ids are stored joined into strings and split
back out on the way to a SearchResult.
"""
from typing import Optional

import chromadb
from loguru import logger
from pydantic import BaseModel

from src.chunking.chunks import Chunk
from src.common.config import StorageConfig, get_settings
from src.common.paths import VECTOR_STORE_DIR
from src.embeddings.embeddings import EmbeddedDocument

_SECTION_SEP = " > "


class SearchResult(BaseModel):
    """One hit from a vector search: the chunk's content + how close it was."""
    chunk_id: str
    text: str
    section_path: list[str]
    first_page: int
    last_page: int
    element_type: str
    score: float  # cosine similarity in [-1, 1]; higher = more similar


class VectorStore:
    """A persistent Chroma collection of chunk embeddings."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_settings().storage
        self._client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        # hnsw:space sets the distance metric used by the index.
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def index(self, embedded_doc: EmbeddedDocument, batch_size: int = 500) -> int:
        """Upsert every embedded chunk into the collection. Idempotent.

        Re-running with the same chunk_ids overwrites rather than duplicates.
        """
        ecs = embedded_doc.embedded_chunks
        logger.info(
            f"Indexing {len(ecs)} vectors into collection "
            f"'{self.config.collection_name}' (distance={self.config.distance})"
        )

        for start in range(0, len(ecs), batch_size):
            batch = ecs[start:start + batch_size]
            self._collection.upsert(
                ids=[ec.chunk.chunk_id for ec in batch],
                embeddings=[ec.embedding for ec in batch],
                documents=[ec.chunk.text for ec in batch],
                metadatas=[_to_metadata(ec.chunk) for ec in batch],
            )

        logger.info(f"Collection now holds {self.count} vectors")
        return self.count

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        """Return the k nearest chunks to a query vector, best first."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        results: list[SearchResult] = []
        # Chroma returns parallel lists, one row per query; we sent one query.
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    section_path=_split_section(meta.get("section_path", "")),
                    first_page=int(meta.get("first_page", 0)),
                    last_page=int(meta.get("last_page", 0)),
                    element_type=str(meta.get("element_type", "")),
                    # cosine distance = 1 - similarity, so invert it back.
                    score=1.0 - float(dist),
                )
            )
        return results


def _to_metadata(chunk: Chunk) -> dict:
    """Flatten a Chunk's metadata into Chroma-safe scalar values."""
    return {
        "section_path": _SECTION_SEP.join(chunk.section_path),
        "first_page": chunk.first_page,
        "last_page": chunk.last_page,
        "element_type": chunk.element_type,
        "token_count": chunk.token_count,
        "source_element_ids": ",".join(chunk.source_element_ids),
    }


def _split_section(joined: str) -> list[str]:
    return joined.split(_SECTION_SEP) if joined else []
