"""RAGPipeline — the single entry point for answering a question end to end.

Wires the read path together: retrieve a wide net -> (optionally) rerank to the
best few -> generate a grounded, cited answer. Everything before this (parse,
chunk, embed, store) happens offline; this is what the API/frontend calls at
request time.

Components are built once and reused across calls, so the OpenAI clients and the
Chroma connection are not re-created per question. They can be injected for tests.
"""
from typing import Optional

from loguru import logger

from src.common.config import get_settings
from src.generation.generation import GeneratedAnswer
from src.generation.generator import Generator
from src.reranking.reranker import Reranker
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """Answers a question against the indexed corpus: retrieve -> rerank -> generate."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
    ):
        settings = get_settings()
        self.rerank_config = settings.rerank
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()

    def answer(self, question: str) -> GeneratedAnswer:
        """Run the full read path and return a grounded, cited answer."""
        logger.info(f"Pipeline answering: {question!r}")

        # Retrieve a wide net so the right chunk is in the pile even if ranked low.
        candidates = self.retriever.retrieve(
            question, k=self.rerank_config.candidate_k
        ).results

        if self.rerank_config.enabled and candidates:
            reranked = self.reranker.rerank(
                question, candidates, top_n=self.rerank_config.top_n
            )
            chunks = [rc.result for rc in reranked.results]
        else:
            # Reranking off: just take the top_n by cosine similarity.
            chunks = candidates[: self.rerank_config.top_n]

        return self.generator.generate(question, chunks)
