"""Generator — writes a grounded answer from the question + retrieved chunks.

This is the "G" in RAG. It stuffs the top chunks into one prompt as numbered
CONTEXT, adds the question, and instructs the model to answer using ONLY that
context (and to say so when the answer isn't there). That grounding — answering
from the retrieved document text rather than the model's own memory — is what
separates RAG from just asking an LLM, and it's what lets answers cite sources.

Takes a list of SearchResults (works directly with retrieval output, or with
reranking output once unwrapped). The OpenAI client is created once and reused,
and can be injected for tests.
"""
from typing import Optional

from loguru import logger
from openai import OpenAI

from src.common.config import GenerationConfig, get_settings
from src.generation.generation import GeneratedAnswer, SourceCitation
from src.storage.vector_store import SearchResult

_SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about a specific document. "
    "Answer the question using ONLY the numbered context passages provided. "
    "You MAY perform arithmetic or combine facts that are stated in the context "
    "(for example, compute a percentage, sum, difference, or ratio from numbers "
    "given in the passages), and should do so when the question calls for it. "
    "Cite the passages you rely on with their bracketed numbers, e.g. [1] or [2]. "
    "Only say you cannot find it in the provided document when the underlying "
    "facts needed to answer are genuinely absent from the context. "
    "Do not use outside knowledge and do not invent facts."
)

# Shown when retrieval/reranking returned nothing, so we never call the model
# with empty context (it would have nothing to ground on).
_NO_CONTEXT_ANSWER = "I cannot find anything relevant to that question in the provided document."


class Generator:
    """Writes a grounded, cited answer from a question and its context chunks."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        client: Optional[OpenAI] = None,
    ):
        settings = get_settings()
        self.config = config or settings.generation
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            max_retries=self.config.max_retries,
        )

    def generate(self, query: str, chunks: list[SearchResult]) -> GeneratedAnswer:
        """Answer the query grounded in the given chunks, citing their numbers."""
        sources = [
            SourceCitation(
                number=i,
                chunk_id=c.chunk_id,
                section_path=c.section_path,
                first_page=c.first_page,
                last_page=c.last_page,
            )
            for i, c in enumerate(chunks, 1)
        ]

        if not chunks:
            logger.warning("No context chunks given; returning a no-context answer")
            return GeneratedAnswer(
                query=query, answer=_NO_CONTEXT_ANSWER, sources=[], model=self.config.model
            )

        context = self._build_context(chunks)
        logger.info(f"Generating answer from {len(chunks)} context chunks for: {query!r}")

        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
            ],
        )
        answer = (response.choices[0].message.content or "").strip()

        return GeneratedAnswer(
            query=query, answer=answer, sources=sources, model=self.config.model
        )

    @staticmethod
    def _build_context(chunks: list[SearchResult]) -> str:
        """Render chunks as numbered passages with a section breadcrumb header."""
        blocks = []
        for i, c in enumerate(chunks, 1):
            breadcrumb = " > ".join(c.section_path) if c.section_path else "(no section)"
            blocks.append(f"[{i}] ({breadcrumb}, p{c.first_page}-{c.last_page})\n{c.text}")
        return "\n\n".join(blocks)
