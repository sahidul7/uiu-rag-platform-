"""
UIU Academic RAG Platform — Document Parsing Pipeline
Unified parser dispatcher for PDF, DOCX, PPTX with OCR fallback.

Architecture:
  BaseParser (ABC)
    ├── PDFParser      (PyMuPDF + pdfplumber + OCR fallback)
    ├── DOCXParser     (python-docx)
    └── PPTXParser     (python-pptx)

Each parser returns a list of ParsedPage — a normalized structure
that feeds into the chunking engine regardless of source format.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.exceptions import CorruptedFileError, ParseError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """
    Normalized output from any parser.
    Page or slide is the atomic unit entering the chunker.
    """
    text: str
    page_number: int
    slide_number: int | None = None
    section_heading: str | None = None
    is_ocr: bool = False
    image_count: int = 0
    tables: list[list[list[str]]] = field(default_factory=list)  # rows x cols
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        return bool(self.text and len(self.text.strip()) > 20)


@dataclass
class ParsedDocument:
    """Full parse result from a single source file."""
    pages: list[ParsedPage]
    total_pages: int
    has_ocr: bool
    parser_used: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.has_content)


# ── Base Parser ───────────────────────────────────────────────────────────────
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        ...

    @abstractmethod
    def supports(self, mime_type: str) -> bool:
        ...


# ── PDF Parser ────────────────────────────────────────────────────────────────
class PDFParser(BaseParser):
    """
    Two-pass PDF parsing:
    1. PyMuPDF (fast text extraction)
    2. pdfplumber (table extraction)
    3. OCR fallback for scanned pages (pytesseract)

    Scanned page detection: if text density < threshold, trigger OCR.
    """

    MIN_TEXT_DENSITY = 0.01  # chars per pixel area — below this → scanned

    def supports(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ParseError("PyMuPDF (fitz) not installed")

        pages: list[ParsedPage] = []
        has_ocr = False

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            raise CorruptedFileError(f"Cannot open PDF {file_path.name}: {e}")

        # Extract tables with pdfplumber (richer table parsing than fitz)
        plumber_tables = self._extract_tables_plumber(file_path)

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text("text")  # type: ignore[attr-defined]

            # Detect scanned page
            is_scanned = self._is_scanned_page(page, text)

            if is_scanned:
                logger.debug("Scanned page detected, using OCR", page=page_num)
                text = self._ocr_page(page)
                has_ocr = True
                is_ocr = True
            else:
                is_ocr = False

            # Heading extraction (bold text on first 20% of page)
            heading = self._extract_heading(page, text)

            parsed_page = ParsedPage(
                text=text.strip(),
                page_number=page_num,
                is_ocr=is_ocr,
                section_heading=heading,
                tables=plumber_tables.get(page_num, []),
                image_count=len(page.get_images()),
            )
            pages.append(parsed_page)

        doc.close()

        return ParsedDocument(
            pages=pages,
            total_pages=len(pages),
            has_ocr=has_ocr,
            parser_used="PDFParser",
        )

    def _is_scanned_page(self, page: Any, text: str) -> bool:
        """
        Heuristic: pages with very little extractable text but images
        are likely scanned. Threshold calibrated for academic PDFs.
        """
        rect = page.rect
        page_area = rect.width * rect.height
        if page_area == 0:
            return False

        text_density = len(text) / page_area
        has_images = len(page.get_images()) > 0

        return text_density < self.MIN_TEXT_DENSITY and has_images

    def _ocr_page(self, page: Any) -> str:
        """Rasterize page and run Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image

            # Rasterize at 300 DPI for good OCR accuracy
            mat = page.get_pixmap(dpi=300)  # type: ignore[attr-defined]
            img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)
            text = pytesseract.image_to_string(
                img,
                lang="eng",
                config="--psm 6 --oem 3",  # assume uniform block of text
            )
            return text.strip()
        except Exception as e:
            logger.warning("OCR failed, using empty text", error=str(e))
            return ""

    def _extract_heading(self, page: Any, text: str) -> str | None:
        """Extract likely heading from page (first non-empty line, size heuristic)."""
        try:
            blocks = page.get_text("dict")["blocks"]  # type: ignore[call-arg]
            largest_size = 0
            heading_text = None
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["size"] > largest_size and span["text"].strip():
                            largest_size = span["size"]
                            heading_text = span["text"].strip()
            return heading_text
        except Exception:
            # Fallback: first line of text
            lines = text.strip().split("\n")
            return lines[0].strip() if lines else None

    def _extract_tables_plumber(self, file_path: Path) -> dict[int, list[list[list[str]]]]:
        """Extract tables per page using pdfplumber."""
        result: dict[int, list[list[list[str]]]] = {}
        try:
            import pdfplumber

            with pdfplumber.open(str(file_path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    tables = page.extract_tables() or []
                    if tables:
                        # Normalize: replace None cells with ""
                        cleaned = [
                            [[cell or "" for cell in row] for row in table]
                            for table in tables
                        ]
                        result[page_num] = cleaned
        except Exception as e:
            logger.warning("pdfplumber table extraction failed", error=str(e))
        return result


# ── DOCX Parser ───────────────────────────────────────────────────────────────
class DOCXParser(BaseParser):
    def supports(self, mime_type: str) -> bool:
        return mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from docx import Document
            from docx.enum.text import WD_STYLE_TYPE
        except ImportError:
            raise ParseError("python-docx not installed")

        try:
            doc = Document(str(file_path))
        except Exception as e:
            raise CorruptedFileError(f"Cannot open DOCX {file_path.name}: {e}")

        # Group paragraphs into logical "pages" by heading boundaries
        logical_pages: list[tuple[str | None, list[str]]] = []  # (heading, paragraphs)
        current_heading: str | None = None
        current_paragraphs: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            style_name = para.style.name.lower() if para.style else ""
            if "heading" in style_name:
                if current_paragraphs:
                    logical_pages.append((current_heading, current_paragraphs))
                current_heading = text
                current_paragraphs = []
            else:
                current_paragraphs.append(text)

        if current_paragraphs:
            logical_pages.append((current_heading, current_paragraphs))

        # Extract tables
        table_texts = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(cells)
            if rows:
                table_texts.append(
                    "\n".join(" | ".join(cell for cell in row) for row in rows)
                )

        pages = []
        for idx, (heading, paragraphs) in enumerate(logical_pages):
            text = "\n".join(paragraphs)
            if table_texts and idx == len(logical_pages) - 1:
                text += "\n\n" + "\n\n".join(table_texts)

            pages.append(ParsedPage(
                text=text,
                page_number=idx + 1,
                section_heading=heading,
            ))

        return ParsedDocument(
            pages=pages,
            total_pages=len(pages),
            has_ocr=False,
            parser_used="DOCXParser",
        )


# ── PPTX Parser ───────────────────────────────────────────────────────────────
class PPTXParser(BaseParser):
    def supports(self, mime_type: str) -> bool:
        return mime_type in {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
        }

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from pptx import Presentation
            from pptx.util import Pt
        except ImportError:
            raise ParseError("python-pptx not installed")

        try:
            prs = Presentation(str(file_path))
        except Exception as e:
            raise CorruptedFileError(f"Cannot open PPTX {file_path.name}: {e}")

        pages = []
        for slide_idx, slide in enumerate(prs.slides):
            slide_num = slide_idx + 1
            title_text = None
            body_parts: list[str] = []

            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue

                # Identify title shape
                is_title = shape.shape_type == 13 or (
                    hasattr(shape, "placeholder_format")
                    and shape.placeholder_format is not None
                    and shape.placeholder_format.idx == 0
                )

                shape_text = "\n".join(
                    para.text.strip()
                    for para in shape.text_frame.paragraphs
                    if para.text.strip()
                )

                if is_title and shape_text:
                    title_text = shape_text
                elif shape_text:
                    body_parts.append(shape_text)

            # Slide notes (often contain important context)
            notes_text = ""
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                if notes_frame:
                    notes_text = notes_frame.text.strip()

            full_text_parts = []
            if title_text:
                full_text_parts.append(title_text)
            full_text_parts.extend(body_parts)
            if notes_text:
                full_text_parts.append(f"[Speaker Notes]: {notes_text}")

            full_text = "\n\n".join(full_text_parts)

            pages.append(ParsedPage(
                text=full_text,
                page_number=slide_num,
                slide_number=slide_num,
                section_heading=title_text,
                image_count=sum(
                    1 for s in slide.shapes
                    if s.shape_type in {13, 11}  # picture / linked picture
                ),
            ))

        return ParsedDocument(
            pages=pages,
            total_pages=len(pages),
            has_ocr=False,
            parser_used="PPTXParser",
        )


# ── Parser Dispatcher ─────────────────────────────────────────────────────────
class DocumentParserDispatcher:
    """
    Routes files to the correct parser based on MIME type.
    Extensible: register new parsers without touching existing code.
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = [
            PDFParser(),
            DOCXParser(),
            PPTXParser(),
        ]

    def parse(self, file_path: Path, mime_type: str) -> ParsedDocument:
        for parser in self._parsers:
            if parser.supports(mime_type):
                logger.info(
                    "Parsing document",
                    file=file_path.name,
                    mime=mime_type,
                    parser=type(parser).__name__,
                )
                try:
                    result = parser.parse(file_path)
                    logger.info(
                        "Parse complete",
                        file=file_path.name,
                        pages=result.total_pages,
                        has_ocr=result.has_ocr,
                    )
                    return result
                except (ParseError, CorruptedFileError):
                    raise
                except Exception as e:
                    raise ParseError(f"Unexpected parse error for {file_path.name}: {e}")

        raise UnsupportedFileTypeError(
            f"No parser for mime type: {mime_type}",
            details={"file": file_path.name, "mime_type": mime_type},
        )

    def register(self, parser: BaseParser) -> None:
        """Register additional parsers at runtime (plugin pattern)."""
        self._parsers.insert(0, parser)


# Singleton
document_parser = DocumentParserDispatcher()
