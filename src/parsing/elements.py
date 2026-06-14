"""Data models for parsed document elements.

Every downstream stage (chunking, embedding, retrieval) consumes ParsedDocument
objects that hold ParsedElement lists. Modify with care.
"""
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel coordinates of an element on its page (top-left origin)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class ParsedElement(BaseModel):
    """A single typed element extracted from a document.

    element_type values come from unstructured: Title, NarrativeText, Table,
    ListItem, FigureCaption, Footer, Header, Image, Address, etc.
    """
    element_id: str
    element_type: str
    text: str
    page_number: int
    bounding_box: Optional[BoundingBox] = None
    # Catch-all for parser-specific metadata we may want later.
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """A whole document's worth of parsed elements + provenance."""
    source_filename: str
    parser_strategy: str
    parser_version: str
    elements: list[ParsedElement]

    @property
    def total_elements(self) -> int:
        return len(self.elements)

    @property
    def total_pages(self) -> int:
        if not self.elements:
            return 0
        return max((e.page_number for e in self.elements), default=0)

    def save(self, output_dir: Path) -> Path:
        """Write to <output_dir>/<source_stem>.parsed.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(self.source_filename).stem}.parsed.json"
        output_path.write_text(self.model_dump_json(indent=2))
        return output_path

    @classmethod
    def load(cls, json_path: Path) -> "ParsedDocument":
        """Load a previously saved ParsedDocument from JSON."""
        return cls.model_validate_json(json_path.read_text())
