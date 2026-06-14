"""Chunker — turns parsed elements into search-ready chunks.

Walks elements in document order with a buffer + section path.

Emit rules:
  - Buffer hits target_size_tokens          -> emit as prose chunk
  - Title (real-section)                    -> flush buffer (discard if tiny),
                                               update section path
  - Title (cover-page noise)                -> treat as content, accumulate
  - Isolated type (Table, Image):
       * if buffer is tiny (< min_size)     -> fold buffer text INTO the
                                               isolated chunk (e.g. table caption
                                               merges with its table)
       * else                                -> flush buffer (discard if tiny),
                                               emit isolated chunk
  - finalize()                              -> flush remaining (discard if tiny)

Buffers below min_size are not emitted on their own; they either fold into
the next table/image, or are discarded at section/document boundaries.
"""
import re
from typing import Optional

import tiktoken
from loguru import logger

from src.chunking.chunks import Chunk, ChunkedDocument
from src.common.config import ChunkingConfig, get_settings
from src.parsing.elements import ParsedDocument, ParsedElement

CHUNKER_VERSION = "v1.3"
MAX_SECTION_DEPTH = 4

_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_ALPHA_RUN_RE = re.compile(r"[A-Za-z]{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_document(
    parsed_doc: ParsedDocument,
    config: Optional[ChunkingConfig] = None,
) -> ChunkedDocument:
    if config is None:
        config = get_settings().chunking

    encoding = tiktoken.get_encoding(config.tokenizer_encoding)
    state = _ChunkerState(config, encoding)

    logger.info(
        f"Chunking {parsed_doc.total_elements} elements "
        f"(target={config.target_size_tokens}t, max={config.max_size_tokens}t, "
        f"min={config.min_size_tokens}t)"
    )

    for element in parsed_doc.elements:
        state.process(element)
    state.finalize()

    logger.info(
        f"Produced {len(state.chunks)} chunks "
        f"(discarded {state.discarded_count} tiny buffers)"
    )

    return ChunkedDocument(
        source_filename=parsed_doc.source_filename,
        chunker_version=CHUNKER_VERSION,
        chunks=state.chunks,
    )


class _ChunkerState:
    def __init__(self, config: ChunkingConfig, encoding):
        self.config = config
        self.encoding = encoding
        self.chunks: list[Chunk] = []
        self.buffer_text: list[str] = []
        self.buffer_element_ids: list[str] = []
        self.buffer_pages: set[int] = set()
        self.section_path: list[str] = []
        self.discarded_count = 0

    def process(self, element: ParsedElement) -> None:
        if self._should_drop(element):
            return
        if element.element_type == "Title":
            self._handle_title(element)
            return
        if element.element_type in self.config.isolated_element_types:
            self._handle_isolated(element)
            return
        self._add_to_buffer(element)
        if self._buffer_tokens() >= self.config.target_size_tokens:
            self._flush_buffer()

    def finalize(self) -> None:
        self._flush_buffer()

    def _should_drop(self, element: ParsedElement) -> bool:
        if element.element_type in self.config.drop_element_types:
            return True
        if len(element.text.strip()) < self.config.min_element_chars:
            return True
        return False

    def _handle_title(self, element: ParsedElement) -> None:
        title_text = element.text.strip()

        if not _is_real_section_heading(title_text):
            self._add_to_buffer(element)
            if self._buffer_tokens() >= self.config.target_size_tokens:
                self._flush_buffer()
            return

        # Real heading: flush (discard if tiny), then update path.
        self._flush_buffer()

        depth = element.metadata.get("category_depth")
        if isinstance(depth, int) and 0 <= depth < MAX_SECTION_DEPTH:
            self.section_path = self.section_path[:depth] + [title_text]
        else:
            self.section_path.append(title_text)
            if len(self.section_path) > MAX_SECTION_DEPTH:
                self.section_path = self.section_path[-MAX_SECTION_DEPTH:]

    def _handle_isolated(self, element: ParsedElement) -> None:
        """Emit Table/Image. Fold tiny preamble (caption) into its chunk."""
        buffer_tokens = self._buffer_tokens()
        elem_type = "table" if element.element_type == "Table" else "image"

        if 0 < buffer_tokens < self.config.min_size_tokens:
            # Fold tiny preamble into the isolated chunk.
            preamble_text = "\n\n".join(self.buffer_text)
            preamble_ids = list(self.buffer_element_ids)
            preamble_pages = set(self.buffer_pages)
            self._reset_buffer()

            combined_text = f"{preamble_text}\n\n{element.text}"
            self._emit_chunk(
                text=combined_text,
                element_ids=preamble_ids + [element.element_id],
                pages=sorted(preamble_pages | {element.page_number}),
                element_type=elem_type,
                token_count=len(self.encoding.encode(combined_text)),
            )
        else:
            # Buffer is substantial (or empty): flush, then emit isolated alone.
            self._flush_buffer()
            self._emit_chunk(
                text=element.text,
                element_ids=[element.element_id],
                pages=[element.page_number],
                element_type=elem_type,
                token_count=len(self.encoding.encode(element.text)),
            )

    def _add_to_buffer(self, element: ParsedElement) -> None:
        self.buffer_text.append(element.text.strip())
        self.buffer_element_ids.append(element.element_id)
        self.buffer_pages.add(element.page_number)

    def _buffer_tokens(self) -> int:
        if not self.buffer_text:
            return 0
        return len(self.encoding.encode("\n\n".join(self.buffer_text)))

    def _flush_buffer(self) -> None:
        """Emit buffer if >= min_size, else discard."""
        if not self.buffer_element_ids:
            return
        text = "\n\n".join(self.buffer_text)
        token_count = len(self.encoding.encode(text))

        if token_count < self.config.min_size_tokens:
            self.discarded_count += 1
            logger.debug(f"Discarding {token_count}-token buffer: {text[:60]!r}")
            self._reset_buffer()
            return

        if token_count > self.config.max_size_tokens:
            self._emit_split(text)
        else:
            self._emit_chunk(
                text=text,
                element_ids=list(self.buffer_element_ids),
                pages=sorted(self.buffer_pages),
                element_type="prose",
                token_count=token_count,
            )
        self._reset_buffer()

    def _emit_split(self, text: str) -> None:
        sentences = _split_sentences(text)
        current: list[str] = []
        for sentence in sentences:
            current.append(sentence)
            joined = " ".join(current)
            current_tokens = len(self.encoding.encode(joined))
            if current_tokens >= self.config.target_size_tokens:
                self._emit_chunk(
                    text=joined,
                    element_ids=list(self.buffer_element_ids),
                    pages=sorted(self.buffer_pages),
                    element_type="prose",
                    token_count=current_tokens,
                )
                current = self._overlap_tail(current)
        if current:
            joined = " ".join(current)
            current_tokens = len(self.encoding.encode(joined))
            if current_tokens >= self.config.min_size_tokens:
                self._emit_chunk(
                    text=joined,
                    element_ids=list(self.buffer_element_ids),
                    pages=sorted(self.buffer_pages),
                    element_type="prose",
                    token_count=current_tokens,
                )

    def _overlap_tail(self, sentences: list[str]) -> list[str]:
        """Trailing sentences to carry into the next split chunk as overlap.

        Accumulates whole sentences from the end until their combined size
        reaches config.overlap_tokens, so the next chunk starts with ~that many
        tokens of shared context. Overlap snaps to sentence boundaries (we never
        cut mid-sentence), so the carried size is approximate, not exact.

        Returns [] when overlap is disabled (overlap_tokens <= 0). Never carries
        the whole list — at least one sentence must advance, or splitting would
        not make progress.
        """
        if self.config.overlap_tokens <= 0 or not sentences:
            return []
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            tail.insert(0, sentence)
            tokens += len(self.encoding.encode(sentence))
            if tokens >= self.config.overlap_tokens:
                break
        if len(tail) >= len(sentences):
            tail = tail[1:]
        return tail

    def _emit_chunk(
        self,
        text: str,
        element_ids: list[str],
        pages: list[int],
        element_type: str,
        token_count: int,
    ) -> None:
        chunk_id = f"chunk_{len(self.chunks):05d}"
        first_page = min(pages) if pages else 0
        last_page = max(pages) if pages else 0
        self.chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source_element_ids=element_ids,
                first_page=first_page,
                last_page=last_page,
                section_path=list(self.section_path),
                element_type=element_type,
                token_count=token_count,
            )
        )

    def _reset_buffer(self) -> None:
        self.buffer_text = []
        self.buffer_element_ids = []
        self.buffer_pages = set()


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _is_real_section_heading(text: str) -> bool:
    if len(text) < 4 or len(text) > 100:
        return False
    if text.startswith("("):
        return False
    if _PHONE_RE.search(text):
        return False
    if not _ALPHA_RUN_RE.search(text):
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / len(text) < 0.4:
        return False
    return True
