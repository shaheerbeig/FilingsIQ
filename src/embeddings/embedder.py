"""Embedder — turns chunks into vectors via the OpenAI embeddings API.

Each chunk's text is sent to the embedding model and the returned vector is
attached back onto the chunk. Optionally the chunk's section_path breadcrumb is
prepended to the text first, so the vector reflects where the content lives.

Chunks are sent in batches (one API request per batch) to cut round trips.
The OpenAI client handles retries/backoff for transient and rate-limit errors.
"""
from typing import Optional

from loguru import logger
from openai import OpenAI

from src.chunking.chunks import Chunk, ChunkedDocument
from src.common.config import EmbeddingConfig, get_settings
from src.embeddings.embeddings import EmbeddedChunk, EmbeddedDocument


def embed_document(
    chunked_doc: ChunkedDocument,
    config: Optional[EmbeddingConfig] = None,
    client: Optional[OpenAI] = None,
) -> EmbeddedDocument:
    """Embed every chunk in a ChunkedDocument.

    config / client fall back to settings-derived defaults. Passing them in is
    used by tests (e.g. to inject a fake client).
    """
    settings = get_settings()
    if config is None:
        config = settings.embedding
    if client is None:
        client = OpenAI(api_key=settings.openai_api_key, max_retries=config.max_retries)

    chunks = chunked_doc.chunks
    inputs = [_build_embed_input(c, config.include_section_path) for c in chunks]

    logger.info(
        f"Embedding {len(chunks)} chunks with {config.model} "
        f"(dim={config.dimensions}, batch_size={config.batch_size}, "
        f"include_section_path={config.include_section_path})"
    )

    embedded: list[EmbeddedChunk] = []
    for start in range(0, len(inputs), config.batch_size):
        end = start + config.batch_size
        batch_inputs = inputs[start:end]
        batch_chunks = chunks[start:end]

        logger.debug(f"Embedding batch {start}-{start + len(batch_inputs)}")
        response = client.embeddings.create(
            model=config.model,
            input=batch_inputs,
            dimensions=config.dimensions,
        )

        # The API returns one item per input; .index gives its position in the
        # batch. Sort by index so alignment is robust regardless of order.
        for item, chunk, embed_input in zip(
            sorted(response.data, key=lambda d: d.index), batch_chunks, batch_inputs
        ):
            embedded.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=item.embedding,
                    embed_input=embed_input,
                )
            )

    logger.info(f"Embedded {len(embedded)} chunks ({config.dimensions}-dim vectors)")

    return EmbeddedDocument(
        source_filename=chunked_doc.source_filename,
        embedding_model=config.model,
        embedding_dim=config.dimensions,
        chunker_version=chunked_doc.chunker_version,
        embedded_chunks=embedded,
    )


def _build_embed_input(chunk: Chunk, include_section_path: bool) -> str:
    """Text actually sent to the embedding model.

    When enabled and a section path exists, prepend the breadcrumb so the vector
    captures the chunk's location in the document, e.g.:

        Item 7 > Liquidity and Capital Resources
        Total cash and marketable securities were ...
    """
    if include_section_path and chunk.section_path:
        breadcrumb = " > ".join(chunk.section_path)
        return f"{breadcrumb}\n\n{chunk.text}"
    return chunk.text
