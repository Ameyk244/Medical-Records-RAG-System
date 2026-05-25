"""Lightweight clinician query intent routing.

This module intentionally uses conservative rules instead of an ML model. The
goal is to add clinical retrieval awareness while keeping the existing vector
pipeline, database schema, and API contracts stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryIntent:
    intent_name: str
    target_sections: tuple[str, ...]
    boost_weight: float


GENERAL_INTENT = QueryIntent(
    intent_name="general",
    target_sections=(),
    boost_weight=0.0,
)

_INTENT_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], float], ...] = (
    (
        "demographics",
        (
            "demographics",
        ),
        (
            "who is the patient",
            "who is patient",
            "patient identity",
            "patient name",
            "name of the patient",
            "age",
            "how old",
            "sex",
            "gender",
            "dob",
            "date of birth",
        ),
        0.12,
    ),
    (
        "history",
        (
            "history",
        ),
        (
            "past illness",
            "past illnesses",
            "previous illness",
            "previous illnesses",
            "medical history",
            "past medical history",
            "pmh",
            "social history",
            "family history",
            "smoking history",
            "smoker",
            "smoking",
            "tobacco use",
            "risk factors",
            "comorbidities",
        ),
        0.12,
    ),
    (
        "allergies",
        (
            "allergies",
        ),
        (
            "allergy",
            "allergies",
            "allergic",
            "adverse reaction",
            "drug reaction",
        ),
        0.12,
    ),
    (
        "medications",
        (
            "medications",
        ),
        (
            "medication",
            "medications",
            "medicine",
            "medicines",
            "drug list",
            "prescription",
            "prescriptions",
            "home meds",
            "current meds",
        ),
        0.08,
    ),
    (
        "labs",
        (
            "labs",
        ),
        (
            "lab",
            "labs",
            "laboratory",
            "lab results",
            "blood work",
            "troponin",
            "hemoglobin",
            "creatinine",
            "glucose",
        ),
        0.10,
    ),
    (
        "procedures",
        (
            "procedures",
        ),
        (
            "procedure",
            "procedures",
            "surgery",
            "operation",
            "operative",
            "what surgery",
            "what procedure",
            "was performed",
            "were performed",
            "performed procedure",
            "angiography",
            "pci",
            "catheterization",
        ),
        0.12,
    ),
    (
        "timeline",
        (
            "hospital_course",
            "procedures",
            "labs",
            "diagnosis",
            "discharge",
        ),
        (
            "timeline",
            "chronology",
            "chronological",
            "sequence of events",
            "hospitalization timeline",
            "admission timeline",
        ),
        0.10,
    ),
    (
        "hospital_course",
        (
            "hospital_course",
        ),
        (
            "hospital course",
            "during admission",
            "during hospitalization",
            "what happened during admission",
            "clinical course",
            "admission course",
            "hospital stay",
        ),
        0.12,
    ),
    (
        "discharge",
        (
            "discharge",
        ),
        (
            "discharge",
            "discharged",
            "disposition",
            "discharge condition",
            "discharge instructions",
        ),
        0.10,
    ),
    (
        "followup",
        (
            "followup",
        ),
        (
            "follow up",
            "follow-up",
            "followup",
            "appointment",
            "recommendations",
            "return precautions",
            "care plan",
            "next steps",
        ),
        0.10,
    ),
    (
        "diagnosis",
        (
            "diagnosis",
        ),
        (
            "diagnosis",
            "diagnoses",
            "diagnosed",
            "what was diagnosed",
            "assessment",
            "impression",
            "problem list",
            "why was the patient admitted",
            "reason for admission",
        ),
        0.10,
    ),
)


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase_or_word(normalized_query: str, phrase: str) -> bool:
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return False
    if " " in normalized_phrase:
        return normalized_phrase in normalized_query
    return re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_query) is not None


def classify_query_intent(query: str) -> QueryIntent:
    """Classify a clinician query into preferred clinical sections.

    The rules are ordered so highly specific safety-critical intents like
    allergies beat broader medication/history wording when a query mentions
    both concepts.
    """
    normalized_query = _normalize(query)
    if not normalized_query:
        return GENERAL_INTENT

    for intent_name, target_sections, triggers, boost_weight in _INTENT_RULES:
        if any(_contains_phrase_or_word(normalized_query, trigger) for trigger in triggers):
            return QueryIntent(
                intent_name=intent_name,
                target_sections=target_sections,
                boost_weight=boost_weight,
            )

    return GENERAL_INTENT
