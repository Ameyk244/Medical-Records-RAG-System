"""Token-aware chunker — clinically section-aware, 512/50 overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

import voyageai

from core.clinical_entities import (
    ClinicalEntities,
    extract_clinical_entities,
    format_clinical_entities,
)
from core.embeddings import EMBEDDING_MODEL

CHUNK_TOKEN_LIMIT = 512
CHUNK_OVERLAP_TOKENS = 50
CLINICAL_ENTITY_TOKEN_RESERVE = 80
DEFAULT_SECTION = "General"
DEFAULT_SEMANTIC_CATEGORY = "general"

SECTION_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "demographics",
        (
            "demographic",
            "identification",
            "personal information",
            "member information",
            "subscriber",
            "birth",
            "dob",
            "sex",
            "gender",
        ),
    ),
    (
        "history",
        (
            "history",
            "past medical",
            "pmh",
            "previous illness",
            "social history",
            "family history",
            "risk factor",
            "smoking",
            "tobacco",
            "alcohol",
        ),
    ),
    (
        "diagnosis",
        (
            "diagnosis",
            "diagnoses",
            "condition",
            "conditions",
            "problem list",
            "assessment",
            "impression",
            "chief complaint",
            "reason for admission",
            "admission diagnosis",
            "discharge diagnosis",
        ),
    ),
    (
        "medications",
        (
            "medication",
            "medications",
            "medicine",
            "rx",
            "prescription",
            "drug",
            "treatment",
            "current meds",
        ),
    ),
    (
        "allergies",
        (
            "allergy",
            "allergies",
            "allergyintolerance",
            "adverse reaction",
        ),
    ),
    (
        "labs",
        (
            "lab",
            "labs",
            "laboratory",
            "observation",
            "observations",
            "diagnostic report",
            "diagnostic reports",
            "result",
            "results",
            "cbc",
            "cmp",
            "troponin",
            "hemoglobin",
            "glucose",
            "creatinine",
        ),
    ),
    (
        "procedures",
        (
            "procedure",
            "procedures",
            "surgery",
            "operation",
            "operative",
            "angiography",
            "pci",
            "catheterization",
            "intervention",
        ),
    ),
    (
        "hospital_course",
        (
            "hospital course",
            "course",
            "admission course",
            "encounter",
            "encounters",
            "visit",
            "progress",
            "clinical course",
            "ed course",
            "emergency department course",
        ),
    ),
    (
        "discharge",
        (
            "discharge",
            "disposition",
            "discharge condition",
            "discharge instructions",
            "discharge summary",
        ),
    ),
    (
        "followup",
        (
            "follow up",
            "follow-up",
            "followup",
            "plan",
            "care plan",
            "recommendation",
            "recommendations",
            "appointment",
            "return precautions",
        ),
    ),
)

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
    section_type: str | None = None
    semantic_category: str | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    token_count: int
    section: str
    chunk_index: int
    source_heading: str | None = None
    section_type: str = DEFAULT_SEMANTIC_CATEGORY
    semantic_category: str = DEFAULT_SEMANTIC_CATEGORY
    clinical_entities: ClinicalEntities | None = None


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


def _normalise_heading(heading: str | None) -> str:
    if heading and heading.strip():
        return re.sub(r"\s+", " ", heading).strip()
    return DEFAULT_SECTION


def _classify_section(
    heading: str | None,
    text: str,
    explicit_section_type: str | None = None,
    explicit_semantic_category: str | None = None,
) -> tuple[str, str]:
    """Map source headings to stable clinical buckets for retrieval.

    This lightweight classifier gives later retrieval stages a dependable
    signal for demographics/history/allergies/etc. without introducing a new
    NLP dependency into ingestion.
    """
    if explicit_semantic_category:
        category = explicit_semantic_category.strip().lower()
        section_type = explicit_section_type.strip().lower() if explicit_section_type else category
        return section_type, category

    if explicit_section_type:
        section_type = explicit_section_type.strip().lower()
        return section_type, section_type

    normalized_heading = re.sub(r"[_\-]+", " ", (heading or "").lower())
    normalized_heading = re.sub(r"\s+", " ", normalized_heading).strip()
    if normalized_heading in {"patient", "patient information", "patient details"}:
        return "demographics", "demographics"

    haystack = f"{heading or ''}\n{text[:500]}".lower()
    haystack = re.sub(r"[_\-]+", " ", haystack)
    haystack = re.sub(r"\s+", " ", haystack)

    for category, patterns in SECTION_CATEGORY_PATTERNS:
        if any(pattern in haystack for pattern in patterns):
            return category, category

    return DEFAULT_SEMANTIC_CATEGORY, DEFAULT_SEMANTIC_CATEGORY


def _metadata_prefix(
    section_name: str,
    semantic_category: str,
    clinical_entities: ClinicalEntities | None = None,
) -> str:
    # Prefixing embeds the clinical section signal with the chunk body, which
    # helps demographics/history chunks compete with medication-dense text.
    prefix = (
        f"Section: {section_name}\n"
        f"Section category: {semantic_category}\n"
    )
    entity_text = format_clinical_entities(clinical_entities) if clinical_entities else ""
    if entity_text:
        return f"{prefix}{entity_text}\n\n"
    return f"{prefix}\n"


def _make_chunk(
    body_text: str,
    body_token_count: int,
    section_name: str,
    chunk_index: int,
    source_heading: str | None,
    section_type: str,
    semantic_category: str,
    _prefix_token_budget: int,
) -> Chunk:
    clinical_entities = extract_clinical_entities(
        body_text,
        section_category=semantic_category,
    )
    metadata_prefix = _metadata_prefix(section_name, semantic_category, clinical_entities)
    actual_prefix_token_count = _count_tokens(metadata_prefix)
    return Chunk(
        text=f"{metadata_prefix}{body_text}",
        token_count=body_token_count + actual_prefix_token_count,
        section=section_name,
        chunk_index=chunk_index,
        source_heading=source_heading,
        section_type=section_type,
        semantic_category=semantic_category,
        clinical_entities=clinical_entities,
    )


def _force_split_long_sentence(
    sentence: str,
    section_name: str,
    start_index: int,
    source_heading: str | None,
    section_type: str,
    semantic_category: str,
    prefix_token_count: int,
    body_token_limit: int,
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

        if word_tokens > body_token_limit:
            # Pathological edge case: a single word exceeds the token limit — emit as-is.
            if buffer:
                chunk_text = " ".join(buffer)
                chunks.append(_make_chunk(
                    chunk_text,
                    current_tokens,
                    section_name,
                    next_index,
                    source_heading,
                    section_type,
                    semantic_category,
                    prefix_token_count,
                ))
                next_index += 1
                buffer = []
                current_tokens = 0
            chunks.append(_make_chunk(
                word,
                word_tokens,
                section_name,
                next_index,
                source_heading,
                section_type,
                semantic_category,
                prefix_token_count,
            ))
            next_index += 1
            continue

        if current_tokens + word_tokens > body_token_limit:
            # Flush current buffer.
            chunk_text = " ".join(buffer)
            chunks.append(_make_chunk(
                chunk_text,
                current_tokens,
                section_name,
                next_index,
                source_heading,
                section_type,
                semantic_category,
                prefix_token_count,
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
        chunks.append(_make_chunk(
            " ".join(buffer),
            current_tokens,
            section_name,
            next_index,
            source_heading,
            section_type,
            semantic_category,
            prefix_token_count,
        ))
        next_index += 1

    return chunks, next_index


def _chunk_section(section: Section, start_index: int) -> tuple[list[Chunk], int]:
    section_name = _normalise_heading(section.heading)
    source_heading = (
        section.heading.strip()
        if section.heading and section.heading.strip()
        else None
    )
    section_type, semantic_category = _classify_section(
        section.heading,
        section.text,
        section.section_type,
        section.semantic_category,
    )
    prefix_token_count = (
        _count_tokens(_metadata_prefix(section_name, semantic_category))
        + CLINICAL_ENTITY_TOKEN_RESERVE
    )
    body_token_limit = max(1, CHUNK_TOKEN_LIMIT - prefix_token_count)

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
        if stokens > body_token_limit:
            # Flush whatever is in the buffer before handling the oversized sentence.
            if buffer:
                chunks.append(_make_chunk(
                    " ".join(buffer),
                    current_tokens,
                    section_name,
                    next_index,
                    source_heading,
                    section_type,
                    semantic_category,
                    prefix_token_count,
                ))
                next_index += 1
                buffer = []
                buffer_token_counts = []
                current_tokens = 0

            long_chunks, next_index = _force_split_long_sentence(
                sentence,
                section_name,
                next_index,
                source_heading,
                section_type,
                semantic_category,
                prefix_token_count,
                body_token_limit,
            )
            chunks.extend(long_chunks)

        elif current_tokens + stokens > body_token_limit:
            # Emit the current buffer as a chunk.
            chunks.append(_make_chunk(
                " ".join(buffer),
                current_tokens,
                section_name,
                next_index,
                source_heading,
                section_type,
                semantic_category,
                prefix_token_count,
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
        chunks.append(_make_chunk(
            " ".join(buffer),
            current_tokens,
            section_name,
            next_index,
            source_heading,
            section_type,
            semantic_category,
            prefix_token_count,
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
