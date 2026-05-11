"""
UIU Academic RAG Platform — Semantic Chunking Engine

Design philosophy:
  Naive fixed-length chunking destroys semantic coherence.
  Academic content has natural boundaries: headings, slide titles,
  topic transitions. The chunker must respect these boundaries
  while still bounding chunk size for embedding quality.

Chunking hierarchy (priority order):
  1. Slide-aware (for PPTX) — each slide = semantic unit
  2. Heading-aware (for PDFs/DOCX) — split at H1/H2/H3 boundaries
  3. Semantic paragraph-aware — split at paragraph boundaries
  4. Overlap injection — add configurable overlap at chunk boundaries
  5. Size enforcement — merge undersized, split oversized chunks

Output: List[Chunk] with full metadata for vector storage.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.parsers.pdf_parser import ParsedDocument, ParsedPage

logger = get_logger(__name__)

# Use cl100k_base tokenizer (same as GPT-4 family) for consistent token counting
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def token_slice(text: str, start: int, end: int) -> str:
    tokens = _tokenizer.encode(text)
    return _tokenizer.decode(tokens[start:end])


# ── Chunk dataclass ───────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """
    The atomic retrieval unit.
    All fields survive through the entire pipeline to vector storage.
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    content_hash: str = ""
    token_count: int = 0
    chunk_index: int = 0

    # Position context
    page_number: int | None = None
    slide_number: int | None = None
    section_heading: str | None = None

    # Academic metadata (propagated from DriveFile → Document → Chunk)
    course: str = ""
    document_type: str = ""
    source_filename: str = ""
    topic: str | None = None
    semester: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> "Chunk":
        """Compute derived fields after content is set."""
        self.token_count = count_tokens(self.content)
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return self

    @property
    def is_valid(self) -> bool:
        return self.token_count >= settings.MIN_CHUNK_SIZE


# ── Heading detection ─────────────────────────────────────────────────────────
# Matches Markdown-style headings and common academic document patterns
HEADING_PATTERNS = [
    re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE),              # ## Heading
    re.compile(r"^([A-Z][A-Z\s]{3,50})$", re.MULTILINE),       # ALL CAPS HEADING
    re.compile(r"^\d+[\.\)]\s+([A-Z].{5,80})$", re.MULTILINE), # 1. Section Title
    re.compile(r"^(Chapter\s+\d+.*)$", re.MULTILINE, re.I),    # Chapter 5: ...
    re.compile(r"^(Section\s+\d+[\.\d]*.*)$", re.MULTILINE, re.I),
    re.compile(r"^(Definition|Theorem|Lemma|Proof|Example|Algorithm)\s*[\d\.]*:?", re.MULTILINE, re.I),
]


def detect_heading(text: str) -> str | None:
    """Returns the heading text if the string starts with a heading pattern."""
    text = text.strip()
    for pattern in HEADING_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(0).strip()
    return None


def split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """
    Split text into sections based on heading boundaries.
    Returns list of (heading, content) tuples.
    """
    # Build a combined pattern for all heading types
    combined = re.compile(
        r"(?m)^(#{1,3}\s+.+|[A-Z][A-Z\s]{5,50}|\d+[\.\)]\s+[A-Z].{5,80}|"
        r"Chapter\s+\d+.*|Section\s+\d+[\.\d]*.*|"
        r"(?:Definition|Theorem|Lemma|Proof|Example|Algorithm)\s*[\d\.]*:?.*)$"
    )

    sections: list[tuple[str | None, str]] = []
    last_end = 0
    current_heading: str | None = None

    for match in combined.finditer(text):
        start = match.start()
        if start > last_end:
            content = text[last_end:start].strip()
            if content:
                sections.append((current_heading, content))
        current_heading = match.group(0).strip()
        last_end = match.end()

    # Remaining text after last heading
    remaining = text[last_end:].strip()
    if remaining:
        sections.append((current_heading, remaining))

    return sections if sections else [(None, text)]


# ── Chunking Strategies ───────────────────────────────────────────────────────
class SlideChunker:
    """
    For PPTX: each slide is a natural semantic unit.
    If a slide is too long (lecture slides with extensive notes), split further.
    """

    def chunk(self, page: ParsedPage, base_meta: dict[str, Any]) -> list[Chunk]:
        text = page.text.strip()
        if not text:
            return []

        tokens = count_tokens(text)

        if tokens <= settings.CHUNK_SIZE:
            # Slide fits in one chunk
            return [Chunk(
                content=text,
                page_number=page.page_number,
                slide_number=page.slide_number,
                section_heading=page.section_heading,
                chunk_index=0,
                **base_meta,
            ).finalize()]

        # Oversized slide (has speaker notes) — split on paragraphs
        return _split_by_paragraphs(
            text,
            heading=page.section_heading,
            page_number=page.page_number,
            slide_number=page.slide_number,
            base_meta=base_meta,
        )


class HeadingAwareChunker:
    """
    For PDF/DOCX: split on heading boundaries first,
    then handle oversized sections.
    """

    def chunk(self, page: ParsedPage, base_meta: dict[str, Any]) -> list[Chunk]:
        text = page.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        sections = split_on_headings(text)

        for heading, section_text in sections:
            effective_heading = heading or page.section_heading
            tokens = count_tokens(section_text)

            if tokens <= settings.CHUNK_SIZE:
                chunks.append(Chunk(
                    content=section_text,
                    page_number=page.page_number,
                    section_heading=effective_heading,
                    chunk_index=len(chunks),
                    **base_meta,
                ).finalize())
            else:
                # Section too long — split by paragraphs with overlap
                sub_chunks = _split_by_paragraphs(
                    section_text,
                    heading=effective_heading,
                    page_number=page.page_number,
                    base_meta=base_meta,
                    start_index=len(chunks),
                )
                chunks.extend(sub_chunks)

        return chunks


def _split_by_paragraphs(
    text: str,
    *,
    heading: str | None,
    page_number: int | None,
    slide_number: int | None = None,
    base_meta: dict[str, Any],
    start_index: int = 0,
) -> list[Chunk]:
    """
    Split text into chunks at paragraph boundaries with token overlap.
    Overlap preserves cross-boundary context for better retrieval.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []

    chunk_size = settings.CHUNK_SIZE
    overlap_size = settings.CHUNK_OVERLAP

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If this paragraph alone exceeds chunk size, force-split it
        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_parts:
                content = "\n\n".join(current_parts)
                chunks.append(Chunk(
                    content=content,
                    page_number=page_number,
                    slide_number=slide_number,
                    section_heading=heading,
                    chunk_index=start_index + len(chunks),
                    **base_meta,
                ).finalize())
                overlap_buffer = current_parts[-2:]  # keep last 2 paras for overlap
                current_parts = list(overlap_buffer)
                current_tokens = sum(count_tokens(p) for p in current_parts)

            # Split long paragraph at token boundaries
            sub_chunks = _token_split(para, chunk_size, overlap_size)
            for sub in sub_chunks:
                chunks.append(Chunk(
                    content=sub,
                    page_number=page_number,
                    slide_number=slide_number,
                    section_heading=heading,
                    chunk_index=start_index + len(chunks),
                    **base_meta,
                ).finalize())
            continue

        # Would adding this paragraph exceed limit?
        if current_tokens + para_tokens > chunk_size and current_parts:
            content = "\n\n".join(current_parts)
            chunks.append(Chunk(
                content=content,
                page_number=page_number,
                slide_number=slide_number,
                section_heading=heading,
                chunk_index=start_index + len(chunks),
                **base_meta,
            ).finalize())

            # Start new chunk with overlap from previous
            overlap_buffer = []
            overlap_tokens = 0
            for prev_para in reversed(current_parts):
                pt = count_tokens(prev_para)
                if overlap_tokens + pt <= overlap_size:
                    overlap_buffer.insert(0, prev_para)
                    overlap_tokens += pt
                else:
                    break

            current_parts = overlap_buffer + [para]
            current_tokens = overlap_tokens + para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunks.append(Chunk(
            content="\n\n".join(current_parts),
            page_number=page_number,
            slide_number=slide_number,
            section_heading=heading,
            chunk_index=start_index + len(chunks),
            **base_meta,
        ).finalize())

    return chunks


def _token_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Force-split a long text string by token count."""
    tokens = _tokenizer.encode(text)
    parts = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        parts.append(_tokenizer.decode(tokens[start:end]))
        start = end - overlap if end < len(tokens) else end
    return parts


# ── Chunk Manager (orchestrator) ──────────────────────────────────────────────
class ChunkManager:
    """
    Orchestrates chunking across all document types.
    Dispatches to SlideChunker or HeadingAwareChunker based on document_type.
    Applies post-processing: dedup, min-size filtering, index normalization.
    """

    def __init__(self) -> None:
        self._slide_chunker = SlideChunker()
        self._heading_chunker = HeadingAwareChunker()

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        *,
        course: str,
        document_type: str,
        source_filename: str,
        topic: str | None = None,
        semester: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Chunk a fully-parsed document into retrieval units.
        Returns list of validated, de-duplicated chunks.
        """
        base_meta: dict[str, Any] = {
            "course": course,
            "document_type": document_type,
            "source_filename": source_filename,
            "topic": topic,
            "semester": semester,
            "extra_metadata": extra_metadata or {},
        }

        is_slide_doc = document_type == "lecture_slide" or parsed_doc.parser_used == "PPTXParser"
        chunker = self._slide_chunker if is_slide_doc else self._heading_chunker

        all_chunks: list[Chunk] = []
        for page in parsed_doc.pages:
            page_chunks = chunker.chunk(page, base_meta)
            all_chunks.extend(page_chunks)

        # Post-processing
        valid_chunks = [c for c in all_chunks if c.is_valid]
        deduped_chunks = self._deduplicate(valid_chunks)
        normalized = self._normalize_indices(deduped_chunks)

        logger.info(
            "Chunking complete",
            source=source_filename,
            raw_chunks=len(all_chunks),
            valid_chunks=len(valid_chunks),
            final_chunks=len(normalized),
        )
        return normalized

    def _deduplicate(self, chunks: list[Chunk]) -> list[Chunk]:
        seen: set[str] = set()
        result: list[Chunk] = []
        for chunk in chunks:
            if chunk.content_hash not in seen:
                seen.add(chunk.content_hash)
                result.append(chunk)
        return result

    def _normalize_indices(self, chunks: list[Chunk]) -> list[Chunk]:
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        return chunks


# Singleton
chunk_manager = ChunkManager()
