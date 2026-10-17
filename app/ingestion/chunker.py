"""
Hierarchical document chunker with semantic boundary detection.

Design decision — hierarchical (parent/child) chunking:
  - Small chunks (~512 tokens) are embedded for dense retrieval — they pin-point
    specific passages (e.g. "EBITDA grew 12% in Q3").
  - Large parent chunks (~2048 tokens) are retrieved alongside the child to provide
    the LLM with surrounding context — preventing "lost in the middle" and letting
    the model understand the passage's role within the document.

Semantic boundary detection:
  Instead of naive token-splitting, we detect paragraph breaks, section headings,
  and bullet points. This keeps semantically related content together within a chunk,
  which dramatically improves both embedding quality and LLM comprehension.

Financial domain note:
  Financial documents have a predictable structure (MD&A, Financial Statements, Notes).
  We detect section headers to keep them as natural chunk boundaries rather than
  splitting mid-section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

import tiktoken

from app.config import settings
from app.models.documents import Chunk, SourceMetadata
from app.tokenizer_assets import get_tokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

# Financial document section headers that should never be split across chunks.
# Keeping sections intact is critical for financial analysis — splitting "Revenue"
# from "Revenue Recognition Policies" would confuse the LLM.
SECTION_HEADERS: Pattern[str] = re.compile(
    r"^(#{1,3}\s+)?(?P<header>"
    r"(MANAGEMENT.?S DISCUSSION|FINANCIAL (STATEMENT|HIGHLIGHTS)|"
    r"NOTES TO (THE )?FINANCIAL|REVENUE|COST OF REVENUE|GROSS PROFIT|"
    r"OPERATING EXPENSES|SEGMENT INFORMATION|LIQUIDITY|CAPITAL RESOURCES|"
    r"CRITICAL ACCOUNTING|MARKET RISK|CONTROLS AND PROCEDURES|"
    r"FORWARD.LOOKING|AUDIT|EXECUTIVE COMPENSATION|"
    r"EARNINGS CALL|Q[1-4] \d{4}|FISCAL YEAR \d{4}|"
    r"BALANCE SHEET|INCOME STATEMENT|CASH FLOW|"
    r"EARNINGS PER SHARE|DIVIDEND|SHAREHOLDER"
    r"))",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ChunkBoundary:
    """
    A detected boundary within the document text where a chunk should start or end.
    """

    position: int  # Character position in the document
    priority: int  # Higher = stronger boundary (section header = 3, paragraph = 2, sentence = 1)
    label: str  # Human-readable label for debugging


def detect_boundaries(text: str) -> list[ChunkBoundary]:
    """
    Scan the document text for natural chunk boundaries.

    Priority system (higher = stronger split point):
      3 — Section headers (never split across these)
      2 — Paragraph breaks (double newline)
      1 — Sentence boundaries (single newline or period at end of line)

    The prioritisation ensures that, when we need to split a large chunk,
    we prefer breaking at a paragraph boundary over a sentence boundary.
    """
    boundaries: list[ChunkBoundary] = []

    # Priority 3: Section headers
    for match in SECTION_HEADERS.finditer(text):
        boundaries.append(
            ChunkBoundary(
                position=match.start(),
                priority=3,
                label=f"section: {match.group('header')[:40]}",
            )
        )

    # Priority 2: Paragraph breaks (double newlines — common in prose sections)
    for match in re.finditer(r"\n\n+", text):
        # Only add if not already at a section header boundary
        pos = match.end()
        if not any(b.position == pos for b in boundaries):
            boundaries.append(ChunkBoundary(position=pos, priority=2, label="paragraph"))

    # Priority 1: Single newlines (sentence-level)
    for match in re.finditer(r"(?<=[.!?])\s+\n", text):
        pos = match.end()
        if not any(abs(b.position - pos) < 50 for b in boundaries):
            boundaries.append(ChunkBoundary(position=pos, priority=1, label="sentence"))

    # Sort by position ascending — we'll process in document order
    boundaries.sort(key=lambda b: b.position)
    return boundaries


def split_into_chunks(
    text: str,
    tokenizer: tiktoken.Encoding,
    max_tokens: int,
    overlap_tokens: int,
    boundaries: list[ChunkBoundary],
) -> list[tuple[int, int]]:
    """
    Split text into overlapping chunks aligned to natural boundaries.

    Args:
        text: The full document text.
        tokenizer: tiktoken tokenizer instance.
        max_tokens: Target chunk size in tokens.
        overlap_tokens: Token overlap between consecutive chunks.
        boundaries: Pre-detected natural boundaries.

    Returns:
        List of (start_char, end_char) tuples defining each chunk's span.

    Algorithm:
        1. Start at position 0
        2. Find the furthest boundary that fits within max_tokens
        3. Snip the chunk there
        4. Walk back by overlap_tokens and repeat

    This is a greedy algorithm — it prefers larger chunks (up to max_tokens) to
    maximise context per chunk while respecting natural boundaries.
    """
    spans: list[tuple[int, int]] = []
    start = 0

    while start < len(text):
        # Compute the ideal end position: max_tokens from start, token-counted
        encoded = tokenizer.encode(text[start:])
        ideal_end_char = _find_char_position(encoded[:max_tokens], tokenizer)
        if ideal_end_char <= 0:
            raise ValueError("token budget cannot fit the next Unicode character")

        # Find the highest-priority boundary within the window
        chunk_end = start + ideal_end_char
        best_boundary: ChunkBoundary | None = None

        for b in boundaries:
            if start < b.position <= start + ideal_end_char and (
                best_boundary is None or b.priority >= best_boundary.priority
            ):
                best_boundary = b

        if best_boundary is not None:
            chunk_end = best_boundary.position

        # Ensure minimum chunk size (at least 50 chars)
        if chunk_end - start < 50:
            chunk_end = start + ideal_end_char

        spans.append((start, chunk_end))

        # Walk back by overlap_tokens for the next chunk start
        if chunk_end >= len(text):
            break
        tokens = tokenizer.encode(text[start:chunk_end])
        overlap_chars = (
            len(tokenizer.decode_bytes(tokens[-overlap_tokens:]).decode("utf-8", errors="ignore"))
            if overlap_tokens
            else 0
        )
        start = max(start + max(1, (chunk_end - start) // 2), chunk_end - overlap_chars)

    return spans


def _find_char_position(tokens: list[int], tokenizer: tiktoken.Encoding) -> int:
    """
    Given a list of tokens, find how many characters they span in the original text.

    Decode the token prefix, dropping only an incomplete trailing UTF-8 character.
    The caller rejects a budget too small to fit even one complete character.
    """
    return len(tokenizer.decode_bytes(tokens).decode("utf-8", errors="ignore"))


def chunk_document(text: str, metadata: SourceMetadata) -> list[Chunk]:
    """
    Entry point: given document text + metadata, produce a flat list of Chunk objects.

    Steps:
      1. Detect natural boundaries (section headers, paragraphs, sentences)
      2. Generate parent chunks (large — for LLM context)
      3. Split each parent chunk into child chunks (small — for retrieval)
      4. Assign parent-child relationships via parent_chunk_id

    Returns a flat list — child chunks reference their parent via parent_chunk_id.
    The Azure AI Search index receives all chunks in a single flat index with
    vector columns for children only (parent chunks are retrieved by ID join).
    """
    tokenizer = get_tokenizer()
    boundaries = detect_boundaries(text)

    # Step 1: Parent chunks — large granularity for context
    parent_spans = split_into_chunks(
        text=text,
        tokenizer=tokenizer,
        max_tokens=settings.chunk_size_large,
        overlap_tokens=settings.chunk_overlap_tokens,
        boundaries=boundaries,
    )

    chunks: list[Chunk] = []

    for parent_idx, (parent_start, parent_end) in enumerate(parent_spans):
        parent_text = text[parent_start:parent_end]

        parent_chunk = Chunk(
            chunk_type="parent",
            content=parent_text,
            token_count=len(tokenizer.encode(parent_text)),
            chunk_index=parent_idx,
            document_id=metadata.document_id,
            filename=metadata.filename,
            ticker=metadata.ticker,
            fiscal_year=metadata.fiscal_year,
            document_type=metadata.document_type,
        )
        chunks.append(parent_chunk)

        # Step 2: Child chunks — split parent into smaller retrieval units
        child_spans = split_into_chunks(
            text=parent_text,
            tokenizer=tokenizer,
            max_tokens=settings.chunk_size_small,
            overlap_tokens=settings.chunk_overlap_tokens,
            boundaries=detect_boundaries(parent_text),
        )

        for child_idx, (child_start, child_end) in enumerate(child_spans):
            child_text = parent_text[child_start:child_end]

            child_chunk = Chunk(
                parent_chunk_id=parent_chunk.chunk_id,
                chunk_type="child",
                content=child_text,
                token_count=len(tokenizer.encode(child_text)),
                chunk_index=parent_idx * 1000 + child_idx,  # Globally unique index
                document_id=metadata.document_id,
                filename=metadata.filename,
                ticker=metadata.ticker,
                fiscal_year=metadata.fiscal_year,
                document_type=metadata.document_type,
            )
            chunks.append(child_chunk)

    return chunks
