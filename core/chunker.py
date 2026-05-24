"""Token-aware chunker — section-aware, 512/50 overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

import voyageai

from core.embeddings import EMBEDDING_MODEL

CHUNK_TOKEN_LIMIT = 512
CHUNK_OVERLAP_TOKENS = 50
DEFAULT_SECTION = "General"

# Phase 1 baseline — replace with pysbd/scispaCy in Phase 5.
_ABBREVIATIONS: frozenset[str] = frozenset([
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs",
    "e.g", "i.e", "etc", "a.m", "p.m", "inc", "ltd", "co", "no",
    "fig", "vol", "q.d", "b.i.d", "t.i.d", "q.i.d", "p.r.n",
    "i.v", "i.m", "p.o",
])

_ABBREV_PLACEHOLDER = "\x00PERIOD\x00"


@dataclass(frozen=True)
class Section:
    heading: str | None
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int
    section: str
    chunk_index: int


_tokenizer_client: voyageai.Client | None = None


def _get_tokenizer_client() -> voyageai.Client:
    global _tokenizer_client
    if _tokenizer_client is None:
        _tokenizer_client = voyageai.Client()
    return _tokenizer_client


def _count_tokens(text: str) -> int:
    return _get_tokenizer_client().count_tokens([text], model=EMBEDDING_MODEL)


def _split_sentences(text: str) -> list[str]:
    # Collapse whitespace runs to a single space.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    protected = normalized

    # Protect decimal numbers (e.g. 0.5 mg) before abbreviations.
    protected = re.sub(r"(\d+)\.(\d+)", r"\1" + _ABBREV_PLACEHOLDER + r"\2", protected)

    # Protect known abbreviations. Lambda rewrite preserves the matched text's casing.
    for abbr in _ABBREVIATIONS:
        pattern = re.compile(r"(?i)\b" + re.escape(abbr) + r"\.")
        protected = pattern.sub(lambda m: m.group(0).replace(".", _ABBREV_PLACEHOLDER), protected)

    sentences = re.split(r"(?<=[.!?])\s+", protected)

    # Restore placeholders and clean up.
    restored = [s.replace(_ABBREV_PLACEHOLDER, ".").strip() for s in sentences]
    return [s for s in restored if s]


def _force_split_long_sentence(
    sentence: str, section_name: str, start_index: int
) -> tuple[list[Chunk], int]:
    # Voyage SDK doesn't expose token→text decoding, so we operate at the word level.
    words = sentence.split()
    if not words:
        return [], start_index

    chunks: list[Chunk] = []
    next_index = start_index
    buffer: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = _count_tokens(word)

        if word_tokens > CHUNK_TOKEN_LIMIT:
            # Pathological edge case: a single word exceeds the token limit — emit as-is.
            if buffer:
                chunk_text = " ".join(buffer)
                chunks.append(Chunk(
                    text=chunk_text,
                    token_count=current_tokens,
                    section=section_name,
                    chunk_index=next_index,
                ))
                next_index += 1
                buffer = []
                current_tokens = 0
            chunks.append(Chunk(
                text=word,
                token_count=word_tokens,
                section=section_name,
                chunk_index=next_index,
            ))
            next_index += 1
            continue

        if current_tokens + word_tokens > CHUNK_TOKEN_LIMIT:
            # Flush current buffer.
            chunk_text = " ".join(buffer)
            chunks.append(Chunk(
                text=chunk_text,
                token_count=current_tokens,
                section=section_name,
                chunk_index=next_index,
            ))
            next_index += 1

            # Build overlap tail from the end of the just-emitted buffer.
            tail: list[str] = []
            tail_tokens = 0
            for w in reversed(buffer):
                wt = _count_tokens(w)
                tail.append(w)
                tail_tokens += wt
                if tail_tokens >= CHUNK_OVERLAP_TOKENS:
                    break
            tail.reverse()

            buffer = tail + [word]
            current_tokens = tail_tokens + word_tokens
        else:
            buffer.append(word)
            current_tokens += word_tokens

    if buffer:
        chunks.append(Chunk(
            text=" ".join(buffer),
            token_count=current_tokens,
            section=section_name,
            chunk_index=next_index,
        ))
        next_index += 1

    return chunks, next_index


def _chunk_section(section: Section, start_index: int) -> tuple[list[Chunk], int]:
    section_name = (
        section.heading.strip()
        if section.heading and section.heading.strip()
        else DEFAULT_SECTION
    )

    sentences = _split_sentences(section.text)
    if not sentences:
        return [], start_index

    # Pre-compute per-sentence token counts (local tokenization — cheap).
    sentence_tokens = [_count_tokens(s) for s in sentences]

    chunks: list[Chunk] = []
    next_index = start_index
    buffer: list[str] = []
    buffer_token_counts: list[int] = []
    current_tokens = 0

    for sentence, stokens in zip(sentences, sentence_tokens):
        if stokens > CHUNK_TOKEN_LIMIT:
            # Flush whatever is in the buffer before handling the oversized sentence.
            if buffer:
                chunks.append(Chunk(
                    text=" ".join(buffer),
                    token_count=current_tokens,
                    section=section_name,
                    chunk_index=next_index,
                ))
                next_index += 1
                buffer = []
                buffer_token_counts = []
                current_tokens = 0

            long_chunks, next_index = _force_split_long_sentence(sentence, section_name, next_index)
            chunks.extend(long_chunks)

        elif current_tokens + stokens > CHUNK_TOKEN_LIMIT:
            # Emit the current buffer as a chunk.
            chunks.append(Chunk(
                text=" ".join(buffer),
                token_count=current_tokens,
                section=section_name,
                chunk_index=next_index,
            ))
            next_index += 1

            # Build overlap tail: walk backward collecting sentences until ≥ CHUNK_OVERLAP_TOKENS.
            tail_sentences: list[str] = []
            tail_token_counts: list[int] = []
            tail_tokens = 0
            for s, t in zip(reversed(buffer), reversed(buffer_token_counts)):
                tail_sentences.append(s)
                tail_token_counts.append(t)
                tail_tokens += t
                if tail_tokens >= CHUNK_OVERLAP_TOKENS:
                    break
            tail_sentences.reverse()
            tail_token_counts.reverse()

            buffer = tail_sentences + [sentence]
            buffer_token_counts = tail_token_counts + [stokens]
            current_tokens = tail_tokens + stokens

        else:
            buffer.append(sentence)
            buffer_token_counts.append(stokens)
            current_tokens += stokens

    # Flush remaining sentences.
    if buffer:
        chunks.append(Chunk(
            text=" ".join(buffer),
            token_count=current_tokens,
            section=section_name,
            chunk_index=next_index,
        ))
        next_index += 1

    return chunks, next_index


def chunk_document(sections: list[Section]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    next_index = 0
    for section in sections:
        section_chunks, next_index = _chunk_section(section, next_index)
        all_chunks.extend(section_chunks)
    return all_chunks
