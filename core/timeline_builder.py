"""Deterministic clinical timeline construction from retrieved chunks.

The timeline is derived only from retrieved evidence. It gives synthesis a
chronological scaffold for progression, interventions, response, and discharge
trajectory without adding a temporal NLP model or new storage layer.
"""

from __future__ import annotations

import re
from datetime import date

from core.vector_store import ChunkResult

TimelineEvent = dict[str, str]

_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)

_EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("discharge", ("discharge", "discharged", "disposition", "sent home")),
    ("admission", ("admit", "admitted", "admission", "presented", "arrived")),
    ("procedure", ("pci", "stent", "angiography", "catheterization", "cabg", "procedure", "performed")),
    ("diagnosis", ("diagnosis", "diagnosed", "nstemi", "stemi", "impression", "assessment")),
    ("lab_change", ("troponin", "hba1c", "ldl", "glucose", "creatinine", "downtrend", "downtrending", "elevated")),
    ("medication_change", ("started", "discontinued", "changed", "increased", "decreased", "prescribed", "discharged on")),
    ("transfer", ("transfer", "transferred", "icu", "stepdown", "floor")),
    ("followup", ("follow up", "follow-up", "followup", "appointment", "return precautions")),
    ("symptoms", ("chest pain", "dyspnea", "shortness of breath", "sob", "nausea", "fever", "symptom")),
)

_SECTION_EVENT_TYPES: dict[str, str] = {
    "hospital_course": "admission",
    "diagnosis": "diagnosis",
    "procedures": "procedure",
    "medications": "medication_change",
    "labs": "lab_change",
    "discharge": "discharge",
    "followup": "followup",
}


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _section_category(chunk: ChunkResult) -> str | None:
    match = _SECTION_CATEGORY_RE.search(chunk.text)
    if match:
        return match.group(1).strip().lower()
    if not chunk.section:
        return None
    normalized = _normalize(chunk.section)
    if "discharge" in normalized:
        return "discharge"
    if "follow" in normalized or "plan" in normalized:
        return "followup"
    if "procedure" in normalized or "surgery" in normalized:
        return "procedures"
    if "lab" in normalized or "observation" in normalized:
        return "labs"
    if "diagnos" in normalized or "assessment" in normalized:
        return "diagnosis"
    if "hospital course" in normalized or "progress" in normalized:
        return "hospital_course"
    if "medication" in normalized or "rx" in normalized:
        return "medications"
    return None


def _sentences(text: str) -> list[str]:
    body = re.sub(r"(?im)^(Section|Section category|Clinical entities):.*$", " ", text)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _dates_for_sentence(sentence: str, fallback_date: date | None) -> list[str]:
    dates = [match.group(0).replace("/", "-") for match in _DATE_RE.finditer(sentence)]
    if dates:
        return dates
    if fallback_date is not None:
        return [fallback_date.isoformat()]
    return ["unknown"]


def _event_type(sentence: str, section_category: str | None) -> str | None:
    normalized = _normalize(sentence)
    for event_type, patterns in _EVENT_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return event_type
    if section_category:
        return _SECTION_EVENT_TYPES.get(section_category)
    return None


def _event_description(sentence: str) -> str:
    description = re.sub(r"\s+", " ", sentence).strip()
    return description[:280].rstrip()


def _sort_key(event: TimelineEvent) -> tuple[str, str, str]:
    date_value = event["date"]
    unknown_flag = "1" if date_value == "unknown" else "0"
    return (unknown_flag, date_value, event["event_type"])


def _dedupe(events: list[TimelineEvent]) -> list[TimelineEvent]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[TimelineEvent] = []
    for event in events:
        key = (
            event["date"],
            event["event_type"],
            _normalize(event["description"]),
            event["source_chunk"],
        )
        if key in seen:
            continue
        deduped.append(event)
        seen.add(key)
    return deduped


def build_timeline(chunks: list[ChunkResult]) -> list[TimelineEvent]:
    """Build ordered timeline events from retrieved chunks only."""
    events: list[TimelineEvent] = []

    for chunk in chunks:
        section_category = _section_category(chunk)
        for sentence in _sentences(chunk.text):
            event_type = _event_type(sentence, section_category)
            if event_type is None:
                continue

            for event_date in _dates_for_sentence(sentence, chunk.document_date):
                events.append(
                    {
                        "date": event_date,
                        "event_type": event_type,
                        "description": _event_description(sentence),
                        "source_chunk": str(chunk.chunk_id),
                    }
                )

    return sorted(_dedupe(events), key=_sort_key)
