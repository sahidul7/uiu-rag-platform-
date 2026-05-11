"""
UIU Academic RAG Platform — Prompt Construction System

Design principles:
  1. GROUNDED — LLM can only use retrieved context, never its own knowledge
  2. CITED — every factual claim must be traceable to a chunk
  3. TOKEN-OPTIMIZED — compress context to fit within LLM token budget
  4. STRUCTURED — consistent format for reliable citation extraction

The system prompt enforces strict grounding:
  - If context doesn't contain the answer → say so explicitly
  - Never invent citations or paraphrase beyond retrieved text
  - Always use [Source N] bracketed citation format
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.hybrid_retriever import RetrievedChunk

logger = get_logger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the UIU CSE Academic Assistant — a course-grounded AI tutor \
for United International University's Computer Science & Engineering department.

## Your Rules

1. **GROUNDED ONLY**: You MUST answer exclusively using the provided [Retrieved Context].
   Never use your own knowledge, training data, or assumptions.

2. **CITATIONS REQUIRED**: Every factual claim must be followed by a citation in the format:
   [Source 1], [Source 2], etc.
   Where the number corresponds to the source listed in the Retrieved Context.

3. **HONESTY**: If the answer is not found in the retrieved context, respond EXACTLY:
   "Not found in the course materials."
   Do not attempt to answer from memory.

4. **NO HALLUCINATION**: Do not infer, extrapolate, or guess beyond what is explicitly stated
   in the retrieved context.

5. **ACADEMIC TONE**: Be precise, clear, and pedagogically helpful.
   Format code blocks with proper syntax highlighting.
   Use mathematical notation where appropriate.

6. **PARTIAL ANSWERS**: If the context partially answers the question, provide what is available
   and clearly note what is not covered.

## Citation Format
When referencing sources, use: [Source N] where N is the source number from context.
Example: "A B-tree maintains sorted data [Source 1] and allows searches in O(log n) [Source 2]."
"""

USER_PROMPT_TEMPLATE = """## Retrieved Context

{context_block}

---

## Student Question
{question}

## Instructions
Answer the question using ONLY the Retrieved Context above.
Include [Source N] citations for every factual claim.
If the answer is not in the context, respond: "Not found in the course materials."
"""


# ── Context Builder ───────────────────────────────────────────────────────────
@dataclass
class BuiltPrompt:
    system_prompt: str
    user_prompt: str
    context_chunks: list[RetrievedChunk]
    source_map: dict[int, str]   # {1: "Lecture_5.pdf — p.18", ...}
    total_tokens: int
    was_compressed: bool


class PromptBuilder:
    """
    Constructs a citation-aware prompt from retrieved chunks.

    Token budget management:
      - Reserve tokens for system prompt, question, and LLM response
      - Fill remaining budget with top-ranked chunks
      - If context exceeds budget, compress by truncating lower-ranked chunks
      - Source map enables client-side citation resolution
    """

    # Reserve tokens for system prompt + question + response buffer
    SYSTEM_TOKEN_RESERVE = 800
    QUESTION_TOKEN_RESERVE = 200
    RESPONSE_TOKEN_RESERVE = settings.LLM_MAX_TOKENS
    TEMPLATE_OVERHEAD = 150  # formatting tokens

    @property
    def _context_token_budget(self) -> int:
        # GPT-4o-mini context: 128k tokens
        max_context = 100_000  # conservative safety margin
        return max_context - (
            self.SYSTEM_TOKEN_RESERVE
            + self.QUESTION_TOKEN_RESERVE
            + self.RESPONSE_TOKEN_RESERVE
            + self.TEMPLATE_OVERHEAD
        )

    def build(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> BuiltPrompt:
        """
        Build a complete prompt from retrieved chunks.
        Applies context compression if needed.
        """
        if not retrieved_chunks:
            return self._build_no_context_prompt(query)

        # Select chunks that fit within token budget
        selected_chunks, was_compressed = self._fit_context(retrieved_chunks)

        # Build numbered context block
        context_lines = []
        source_map: dict[int, str] = {}

        for i, chunk in enumerate(selected_chunks, start=1):
            source_ref = self._format_source_ref(chunk)
            source_map[i] = source_ref

            context_lines.append(
                f"[Source {i}] {source_ref}\n"
                f"{chunk.content}\n"
            )

        context_block = "\n---\n".join(context_lines)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context_block=context_block,
            question=query,
        )

        total_tokens = (
            count_tokens(SYSTEM_PROMPT)
            + count_tokens(user_prompt)
        )

        logger.info(
            "Prompt built",
            chunks_used=len(selected_chunks),
            total_tokens=total_tokens,
            was_compressed=was_compressed,
        )

        return BuiltPrompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context_chunks=selected_chunks,
            source_map=source_map,
            total_tokens=total_tokens,
            was_compressed=was_compressed,
        )

    def _fit_context(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        """
        Greedily include chunks until token budget is exhausted.
        Chunks already sorted by rerank score (highest first).
        """
        budget = self._context_token_budget
        selected: list[RetrievedChunk] = []
        used_tokens = 0
        was_compressed = False

        for chunk in chunks:
            # Each chunk uses: [Source N] header + content + separator
            chunk_tokens = count_tokens(chunk.content) + 30

            if used_tokens + chunk_tokens <= budget:
                selected.append(chunk)
                used_tokens += chunk_tokens
            else:
                was_compressed = True
                # Try to include a truncated version of this chunk
                remaining = budget - used_tokens - 30
                if remaining > 100:
                    truncated_content = self._truncate_to_tokens(chunk.content, remaining)
                    truncated_chunk = RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        content=truncated_content + " [...]",
                        score=chunk.score,
                        course=chunk.course,
                        document_type=chunk.document_type,
                        source_filename=chunk.source_filename,
                        page_number=chunk.page_number,
                        slide_number=chunk.slide_number,
                        section_heading=chunk.section_heading,
                    )
                    selected.append(truncated_chunk)
                break

        return selected, was_compressed

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = _tokenizer.encode(text)
        return _tokenizer.decode(tokens[:max_tokens])

    def _format_source_ref(self, chunk: RetrievedChunk) -> str:
        """Human-readable source reference for citation UI."""
        parts = [chunk.source_filename]
        if chunk.page_number:
            parts.append(f"Page {chunk.page_number}")
        elif chunk.slide_number:
            parts.append(f"Slide {chunk.slide_number}")
        if chunk.section_heading:
            parts.append(f"§ {chunk.section_heading[:60]}")
        doc_type_label = chunk.document_type.replace("_", " ").title()
        parts.append(f"({doc_type_label})")
        return " | ".join(parts)

    def _build_no_context_prompt(self, query: str) -> BuiltPrompt:
        """When retrieval returns nothing — prompt directs LLM to say so."""
        no_context_user = (
            f"## Retrieved Context\n\n[No relevant content found]\n\n"
            f"---\n\n## Student Question\n{query}\n\n"
            f"## Instructions\nSince no relevant context was retrieved, "
            f"respond: \"Not found in the course materials.\""
        )
        return BuiltPrompt(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=no_context_user,
            context_chunks=[],
            source_map={},
            total_tokens=count_tokens(SYSTEM_PROMPT) + count_tokens(no_context_user),
            was_compressed=False,
        )

    def build_messages(
        self,
        built_prompt: BuiltPrompt,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the full messages list for OpenAI Chat API.
        Injects conversation history for multi-turn support.
        History is truncated to fit within token budget.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": built_prompt.system_prompt}
        ]

        if conversation_history:
            # Include recent history (up to 8 turns = 16 messages)
            recent = conversation_history[-16:]
            messages.extend(recent)

        messages.append({"role": "user", "content": built_prompt.user_prompt})
        return messages


# ── Citation Extractor ────────────────────────────────────────────────────────
class CitationExtractor:
    """
    Extracts [Source N] citations from LLM response.
    Maps citation numbers to actual source references.
    """

    CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

    def extract(
        self,
        response_text: str,
        source_map: dict[int, str],
        context_chunks: list[RetrievedChunk],
    ) -> list[dict[str, Any]]:
        """
        Returns list of citation objects for frontend rendering.
        Each citation: {number, source_ref, chunk_id, document_type, ...}
        """
        found_numbers = set()
        for match in self.CITATION_PATTERN.finditer(response_text):
            found_numbers.add(int(match.group(1)))

        citations = []
        chunk_map = {i + 1: c for i, c in enumerate(context_chunks)}

        for num in sorted(found_numbers):
            if num in source_map and num in chunk_map:
                chunk = chunk_map[num]
                citations.append({
                    "number": num,
                    "source_ref": source_map[num],
                    "chunk_id": chunk.chunk_id,
                    "source_filename": chunk.source_filename,
                    "document_type": chunk.document_type,
                    "page_number": chunk.page_number,
                    "slide_number": chunk.slide_number,
                    "section_heading": chunk.section_heading,
                    "course": chunk.course,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                })

        return citations


# Singletons
prompt_builder = PromptBuilder()
citation_extractor = CitationExtractor()
