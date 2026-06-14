"""Data models for embedded chunks.

Mirrors src/chunking/chunks.py: an EmbeddedChunk is a Chunk plus its meaning
vector, an EmbeddedDocument bundles all embedded chunks from one source plus
provenance (which model produced the vectors, at what dimension).

The full Chunk is kept inline so this file is self-contained — retrieval can
load it and have both the vectors (to search) and the text + metadata (to
return and cite) without re-reading the chunks file.
"""
from pathlib import Path

from pydantic import BaseModel

from src.chunking.chunks import Chunk


class EmbeddedChunk(BaseModel):
    """One chunk plus the vector embedding of its (possibly enriched) text."""
    chunk: Chunk
    embedding: list[float]
    # The exact text sent to the embedding model. May differ from chunk.text
    # when include_section_path prepends the section breadcrumb. Kept for
    # provenance and debugging.
    embed_input: str


class EmbeddedDocument(BaseModel):
    """All embedded chunks from one source document plus how they were produced."""
    source_filename: str
    embedding_model: str
    embedding_dim: int
    chunker_version: str
    embedded_chunks: list[EmbeddedChunk]

    @property
    def total_chunks(self) -> int:
        return len(self.embedded_chunks)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(self.source_filename).stem}.embeddings.json"
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def load(cls, json_path: Path) -> "EmbeddedDocument":
        return cls.model_validate_json(json_path.read_text())
