"""Data models for chunked document content.

Mirrors src/parsing/elements.py: a Chunk is the search-ready unit
produced by the chunker, a ChunkedDocument is the bundle of all chunks
from one source document plus provenance.
"""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One search-ready piece of content built from parsed elements."""
    chunk_id: str
    text: str
    source_element_ids: list[str]
    first_page: int
    last_page: int
    section_path: list[str]
    element_type: str  # "prose" | "table" | "image"
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    """All chunks from one source document plus how they were produced."""
    source_filename: str
    chunker_version: str
    chunks: list[Chunk]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(self.source_filename).stem}.chunks.json"
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def load(cls, json_path: Path) -> "ChunkedDocument":
        return cls.model_validate_json(json_path.read_text())
