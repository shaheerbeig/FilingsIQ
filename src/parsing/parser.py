"""PDF parser — wraps unstructured into our typed data model.

Pre-processes input PDFs by re-saving with PyMuPDF, which strips the
permission-only encryption metadata that some PDFs carry (common in
HTML-derived filings from SEC EDGAR). Without this step, pdfminer.six
silently extracts zero text from such files.
"""
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger
from unstructured.partition.pdf import partition_pdf

from src.common.config import get_settings
from src.parsing.elements import BoundingBox, ParsedDocument, ParsedElement


def parse_pdf(
    pdf_path: Path,
    strategy: Optional[str] = None,
    languages: Optional[list[str]] = None,
    infer_table_structure: Optional[bool] = None,
) -> ParsedDocument:
    """Parse a PDF into a ParsedDocument.

    Any None argument falls back to the value in settings.parsing.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    settings = get_settings()
    strategy = strategy or settings.parsing.default_strategy
    languages = languages or settings.parsing.ocr_languages
    if infer_table_structure is None:
        infer_table_structure = settings.parsing.pdf_infer_table_structure

    parsing_path = _ensure_clean(pdf_path)

    logger.info(
        f"Parsing {parsing_path.name}: strategy={strategy}, "
        f"languages={languages}, infer_table_structure={infer_table_structure}"
    )

    raw_elements = partition_pdf(
        filename=str(parsing_path),
        strategy=strategy,
        languages=languages,
        infer_table_structure=infer_table_structure,
    )

    elements: list[ParsedElement] = []
    for i, el in enumerate(raw_elements):
        bbox = _extract_bbox(el)
        elements.append(
            ParsedElement(
                element_id=f"el_{i:05d}",
                element_type=type(el).__name__,
                text=el.text or "",
                page_number=(el.metadata.page_number if el.metadata else 0) or 0,
                bounding_box=bbox,
                metadata={
                    "filename": el.metadata.filename if el.metadata else None,
                    "filetype": el.metadata.filetype if el.metadata else None,
                    "category_depth": getattr(el.metadata, "category_depth", None)
                    if el.metadata else None,
                },
            )
        )

    logger.info(f"Parsed {len(elements)} elements from {pdf_path.name}")

    return ParsedDocument(
        source_filename=pdf_path.name,
        parser_strategy=strategy,
        parser_version=f"unstructured {pkg_version('unstructured')}",
        elements=elements,
    )


def _ensure_clean(pdf_path: Path) -> Path:
    """Re-save the PDF with PyMuPDF to produce a clean, parser-friendly copy.

    Strips permission-only encryption metadata, runs garbage collection on
    unused objects, and recompresses streams. The output is structurally
    identical to the original but without the metadata flags that confuse
    pdfminer.six. Idempotent — reuses the cleaned copy on subsequent runs.
    """
    clean_path = pdf_path.with_name(pdf_path.stem + ".clean.pdf")

    if clean_path.exists():
        logger.debug(f"Using existing clean copy: {clean_path.name}")
        return clean_path

    logger.info(f"Pre-processing with PyMuPDF: {pdf_path.name} -> {clean_path.name}")
    doc = fitz.open(pdf_path)
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError(f"PDF {pdf_path.name} is password-protected")
        doc.save(str(clean_path), garbage=4, clean=True, deflate=True)
    finally:
        doc.close()

    return clean_path


def _extract_bbox(el) -> Optional[BoundingBox]:
    """Pull bounding-box coords from an unstructured element if available."""
    if not el.metadata or not el.metadata.coordinates:
        return None
    points = el.metadata.coordinates.points
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BoundingBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
