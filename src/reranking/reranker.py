"""Reranker — re-scores retrieval candidates by reading question + chunk together.

Retrieval (the bi-encoder search) embeds the question and each chunk separately,
so it only knows topical closeness. The reranker fixes the order by sending the
question and each chunk to a chat model *together* and asking how well the chunk
answers the question. It outputs a relevance SCORE per chunk — not an answer —
then sorts by it and keeps the top_n.

It only reorders/trims the chunks it is given; it never fetches new ones. So the
caller should retrieve a wide net (candidate_k) before reranking.

One model call per candidate. The OpenAI client is created once and reused, and
can be injected for tests.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from loguru import logger
from openai import OpenAI

from src.common.config import RerankConfig, get_settings
from src.reranking.reranking import RerankedChunk, RerankResult
from src.storage.vector_store import SearchResult

# Asks for a single 0-10 relevance score. Kept terse so the model replies a number.
_SCORE_PROMPT = (
    "You are scoring how well a passage answers a question.\n\n"
    "Question: {query}\n\n"
    "Passage:\n{passage}\n\n"
    "On a scale of 0 to 10, how directly and completely does the passage answer "
    "the question? 0 = irrelevant, 10 = fully answers it.\n"
    "Reply with ONLY the number."
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class Reranker:
    """Re-scores and re-sorts retrieval candidates with an LLM relevance judge."""

    def __init__(
        self,
        config: Optional[RerankConfig] = None,
        client: Optional[OpenAI] = None,
    ):
        settings = get_settings()
        self.config = config or settings.rerank
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            max_retries=self.config.max_retries,
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> RerankResult:
        """Score each candidate against the query, re-sort, keep the best top_n."""
        top_n = top_n if top_n is not None else self.config.top_n
        if not candidates:
            return RerankResult(query=query, results=[])

        # Score candidates concurrently — one model call each, run in parallel so
        # latency is ~one call instead of the sum of all of them. map() preserves
        # input order, so scores line up with candidates.
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            scores = list(pool.map(lambda hit: self._score(query, hit.text), candidates))
        scored = [
            RerankedChunk(result=hit, rerank_score=score)
            for hit, score in zip(candidates, scores)
        ]
        # Highest relevance first. Ties keep retrieval's order (stable sort).
        scored.sort(key=lambda rc: rc.rerank_score, reverse=True)
        kept = scored[:top_n]

        logger.info(
            f"Reranked {len(candidates)} candidates -> kept top {len(kept)} "
            f"for query: {query!r}"
        )
        return RerankResult(query=query, results=kept)

    def _score(self, query: str, passage: str) -> float:
        """Ask the model for a 0-10 relevance score; parse it out robustly."""
        prompt = _SCORE_PROMPT.format(query=query, passage=passage[:2000])
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        return _parse_score(reply)


def _parse_score(reply: str) -> float:
    """Pull the first number from the reply and clamp it to [0, 10]."""
    match = _NUMBER_RE.search(reply)
    if not match:
        logger.warning(f"Reranker returned no parseable score: {reply!r}; using 0.0")
        return 0.0
    return max(0.0, min(10.0, float(match.group())))
