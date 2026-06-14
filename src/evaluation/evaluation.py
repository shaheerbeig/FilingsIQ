"""Data models for evaluation: the test set, per-case results, and the report.

An EvalCase is one ground-truth question: which chunk(s) truly answer it
(relevant_chunk_ids) and what a correct answer looks like (reference_answer).
answerable=false cases have no relevant chunks — they check the system refuses
instead of hallucinating.

A CaseResult holds the metrics for one case. An EvalReport bundles all of them
and exposes the aggregate scores (the numbers that replace eyeballing).
"""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One ground-truth question with its known answer and source chunk(s)."""
    question: str
    reference_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    # False = the document does NOT answer this; a good system should refuse.
    answerable: bool = True


class EvalTestSet(BaseModel):
    """A collection of ground-truth cases."""
    description: str = ""
    cases: list[EvalCase]

    @classmethod
    def load(cls, json_path: Path) -> "EvalTestSet":
        return cls.model_validate_json(json_path.read_text())


class CaseResult(BaseModel):
    """The measured outcome for one EvalCase."""
    question: str
    answerable: bool
    answer: str
    retrieved_chunk_ids: list[str]

    # Retrieval metrics (None for unanswerable cases — no relevant chunk exists).
    hit: Optional[bool] = None              # did a relevant chunk land in the top-k?
    rank: Optional[int] = None              # rank of the first relevant chunk (1-based)
    reciprocal_rank: float = 0.0            # 1/rank, or 0 if not found

    # Answer metrics, judged by an LLM, normalized to 0..1.
    correctness: float = 0.0                # does the answer match the reference?
    faithfulness: float = 0.0               # is the answer grounded in the context?


class EvalReport(BaseModel):
    """All case results plus the aggregate scores across the test set."""
    results: list[CaseResult]

    @property
    def _answerable(self) -> list[CaseResult]:
        return [r for r in self.results if r.answerable]

    @property
    def hit_rate(self) -> float:
        """Fraction of answerable cases whose relevant chunk made the top-k."""
        rs = self._answerable
        return sum(1 for r in rs if r.hit) / len(rs) if rs else 0.0

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over answerable cases (rewards ranking it high)."""
        rs = self._answerable
        return sum(r.reciprocal_rank for r in rs) / len(rs) if rs else 0.0

    @property
    def mean_correctness(self) -> float:
        return sum(r.correctness for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mean_faithfulness(self) -> float:
        return sum(r.faithfulness for r in self.results) / len(self.results) if self.results else 0.0

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path
