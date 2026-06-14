"""Data model for a generated answer.

Generation is the end of the pipeline: it turns the question plus the retrieved
context chunks into a written answer. A GeneratedAnswer keeps the answer text
alongside the sources it was built from, so every answer is traceable back to
the exact chunks (and their pages/sections) that grounded it.

The sources are numbered [1], [2]... in the order they were shown to the model,
so a citation like "[2]" in the answer text maps to sources[1].
"""
from pydantic import BaseModel


class SourceCitation(BaseModel):
    """One context chunk that was made available to the generator."""
    number: int                 # the [n] shown in the prompt, for in-text citations
    chunk_id: str
    section_path: list[str]
    first_page: int
    last_page: int


class GeneratedAnswer(BaseModel):
    """The final written answer plus the sources it was grounded in."""
    query: str
    answer: str
    sources: list[SourceCitation]
    model: str
