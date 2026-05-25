"""Lightweight hybrid retrieval scoring for medical record chunks.

Hybrid scoring keeps pgvector semantic retrieval as the candidate generator,
then improves ranking with exact lexical evidence. This helps clinical QA for
names, dates, abbreviations, allergies, smoking history, and other short facts
that embeddings can underweight.
"""

from __future__ import annotations

import re

SEMANTIC_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.20
INTENT_WEIGHT = 0.10

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "did",
        "do",
        "does",
        "during",
        "for",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "patient",
        "the",
        "there",
        "to",
        "was",
        "were",
        "what",
        "when",
        "who",
        "with",
    }
)

_CLINICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "allergy",
        "allergies",
        "aspirin",
        "cabg",
        "cad",
        "chest",
        "copd",
        "diabetes",
        "diagnosis",
        "discharge",
        "followup",
        "hypertension",
        "labs",
        "metformin",
        "nstemi",
        "penicillin",
        "pci",
        "procedure",
        "smoker",
        "smoking",
        "stemi",
        "surgery",
        "troponin",
    }
)


def normalize_text(text: str) -> str:
    """Lowercase text, remove punctuation, and normalize whitespace."""
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _query_terms(query: str) -> list[str]:
    terms = normalize_text(query).split()
    return [term for term in terms if term not in _STOPWORDS]


def _important_terms(terms: list[str]) -> set[str]:
    return {
        term
        for term in terms
        if len(term) >= 3 or term.isdigit() or term in _CLINICAL_KEYWORDS
    }


def _term_variants(term: str) -> set[str]:
    variants = {term}
    if term == "allergy":
        variants.add("allergies")
    elif term == "allergies":
        variants.add("allergy")
    elif term == "diagnosis":
        variants.add("diagnoses")
    elif term == "diagnoses":
        variants.add("diagnosis")
    elif term == "medication":
        variants.add("medications")
    elif term == "medications":
        variants.add("medication")
    elif term == "smoker":
        variants.add("smoking")
    elif term == "smoking":
        variants.add("smoker")
    return variants


def _expand_terms(terms: set[str]) -> set[str]:
    expanded: set[str] = set()
    for term in terms:
        expanded.update(_term_variants(term))
    return expanded


def _has_exact_phrase(query: str, chunk_text: str) -> bool:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return False

    query_terms = normalized_query.split()
    if len(query_terms) < 2:
        return False

    normalized_chunk = normalize_text(chunk_text)
    return normalized_query in normalized_chunk


def lexical_score(query: str, chunk_text: str) -> float:
    """Return a bounded 0..1 lexical relevance score.

    The score rewards exact phrase presence, overlap across meaningful query
    terms, clinical keyword matches, and exact numeric/date-like facts. These
    signals are intentionally simple but make names like "John Doe", diagnoses
    like "NSTEMI", and facts like "former smoker" survive semantic ambiguity.
    """
    terms = _query_terms(query)
    important = _important_terms(terms)
    if not important:
        return 0.0

    normalized_chunk = normalize_text(chunk_text)
    chunk_terms = _expand_terms(set(normalized_chunk.split()))
    matched_terms = important & chunk_terms

    score = 0.0

    overlap_ratio = len(matched_terms) / len(important)
    score += overlap_ratio * 0.45

    if _has_exact_phrase(query, chunk_text):
        score += 0.35

    clinical_matches = matched_terms & _CLINICAL_KEYWORDS
    if clinical_matches:
        score += min(0.15, 0.05 * len(clinical_matches))

    exact_fact_terms = {
        term
        for term in matched_terms
        if term.isdigit() or re.fullmatch(r"[a-z]*\d+[a-z0-9]*", term)
    }
    if exact_fact_terms:
        score += min(0.15, 0.05 * len(exact_fact_terms))

    if len(matched_terms) >= 2:
        score += 0.10

    return min(1.0, score)


def hybrid_score(
    semantic_score: float,
    lexical_score_value: float,
    intent_boost: float,
    *,
    semantic_weight: float = SEMANTIC_WEIGHT,
    lexical_weight: float = LEXICAL_WEIGHT,
    intent_weight: float = INTENT_WEIGHT,
) -> float:
    """Combine semantic, lexical, and intent-aware section signals."""
    bounded_lexical = max(0.0, min(1.0, lexical_score_value))
    bounded_intent = max(0.0, min(1.0, intent_boost))
    return (
        (semantic_score * semantic_weight)
        + (bounded_lexical * lexical_weight)
        + (bounded_intent * intent_weight)
    )
