"""Evaluator — runs the pipeline on a test set and measures it with numbers.

For each ground-truth case it runs the live read path (retrieve -> rerank ->
generate), then scores two things:

  Retrieval (objective, from the known relevant_chunk_ids):
    - hit   : did a relevant chunk land in the top-k?
    - rank / reciprocal_rank : how high was the first relevant chunk?

  Answer (LLM-judged, 0..1):
    - correctness  : does the answer match the reference answer?
    - faithfulness : is every claim grounded in the retrieved context (no
                     hallucination)?

The judge is a strong chat model prompted to return a single 0-10 score, which
we normalize to 0..1. Components (retriever/reranker/generator/client) are
injectable so the harness can wire them once and reuse them.
"""
import re
from typing import Optional

from loguru import logger
from openai import OpenAI

from src.common.config import EvaluationConfig, get_settings
from src.evaluation.evaluation import CaseResult, EvalCase, EvalReport, EvalTestSet
from src.generation.generator import Generator
from src.reranking.reranker import Reranker
from src.retrieval.retriever import Retriever

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_CORRECTNESS_PROMPT = (
    "You are grading an answer against a reference answer.\n\n"
    "Question: {question}\n\n"
    "Reference answer: {reference}\n\n"
    "Generated answer: {answer}\n\n"
    "Score 0 to 10 how well the generated answer matches the reference in factual "
    "content (10 = fully correct and complete, 0 = wrong or missing). "
    "Reply with ONLY the number."
)

_FAITHFULNESS_PROMPT = (
    "You are checking whether an answer is grounded in the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Score 0 to 10 how well every factual claim in the answer is supported by the "
    "context (10 = fully supported, 0 = contradicts or invents facts not in the "
    "context). An answer that correctly says the information is not present should "
    "score 10. Reply with ONLY the number."
)


class Evaluator:
    """Scores the RAG pipeline against a ground-truth test set."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        client: Optional[OpenAI] = None,
    ):
        settings = get_settings()
        self.config = config or settings.evaluation
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()
        self.rerank_config = settings.rerank
        self.client = client or OpenAI(
            api_key=settings.openai_api_key, max_retries=self.config.max_retries
        )

    def evaluate(self, testset: EvalTestSet) -> EvalReport:
        logger.info(f"Evaluating {len(testset.cases)} cases (k={self.config.k})")
        results = [self._evaluate_case(case) for case in testset.cases]
        report = EvalReport(results=results)
        logger.info(
            f"Done. hit_rate={report.hit_rate:.2f} mrr={report.mrr:.2f} "
            f"correctness={report.mean_correctness:.2f} "
            f"faithfulness={report.mean_faithfulness:.2f}"
        )
        return report

    def _evaluate_case(self, case: EvalCase) -> CaseResult:
        # Run the live read path: retrieve wide -> rerank -> generate.
        candidates = self.retriever.retrieve(
            case.question, k=self.rerank_config.candidate_k
        ).results
        reranked = self.reranker.rerank(
            case.question, candidates, top_n=self.config.k
        )
        chunks = [rc.result for rc in reranked.results]
        retrieved_ids = [c.chunk_id for c in chunks]

        # Retrieval metrics: where did the first relevant chunk land in the top-k?
        hit: Optional[bool] = None
        rank: Optional[int] = None
        rr = 0.0
        if case.answerable:
            hit = False
            for position, cid in enumerate(retrieved_ids, 1):
                if cid in case.relevant_chunk_ids:
                    hit, rank, rr = True, position, 1.0 / position
                    break

        answer = self.generator.generate(case.question, chunks)

        # Answer metrics, judged by the LLM.
        correctness = self._judge(
            _CORRECTNESS_PROMPT.format(
                question=case.question,
                reference=case.reference_answer,
                answer=answer.answer,
            )
        )
        context = self.generator._build_context(chunks) if chunks else "(no context)"
        faithfulness = self._judge(
            _FAITHFULNESS_PROMPT.format(context=context, answer=answer.answer)
        )

        logger.info(
            f"[{'OK ' if (hit or not case.answerable) else 'MISS'}] "
            f"correctness={correctness:.2f} faithful={faithfulness:.2f} "
            f"| {case.question[:55]!r}"
        )

        return CaseResult(
            question=case.question,
            answerable=case.answerable,
            answer=answer.answer,
            retrieved_chunk_ids=retrieved_ids,
            hit=hit,
            rank=rank,
            reciprocal_rank=rr,
            correctness=correctness,
            faithfulness=faithfulness,
        )

    def _judge(self, prompt: str) -> float:
        """Ask the judge for a 0-10 score; return it normalized to 0..1."""
        response = self.client.chat.completions.create(
            model=self.config.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        match = _NUMBER_RE.search(reply)
        if not match:
            logger.warning(f"Judge returned no parseable score: {reply!r}; using 0.0")
            return 0.0
        return max(0.0, min(10.0, float(match.group()))) / 10.0
