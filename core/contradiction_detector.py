"""Deterministic first-pass contradiction detection for clinical evidence.

The goal is not to adjudicate truth. The detector surfaces conflicts that
Gemini should mention cautiously and resolve only when the numbered evidence
supports doing so.
"""

from __future__ import annotations

import re
from typing import Any

from core.clinical_entities import extract_clinical_entities
from core.vector_store import ChunkResult

Contradiction = dict[str, Any]

_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")

_SMOKING_FORMER = (
    "former smoker",
    "quit smoking",
    "quit tobacco",
    "ex smoker",
)
_SMOKING_CURRENT = (
    "current smoker",
    "active smoker",
    "currently smokes",
    "smokes ",
    "tobacco use",
    "current tobacco",
)

_STOPPED_PATTERNS = (
    r"\b(?:stop|stopped|discontinue|discontinued|held)\s+(?:the\s+)?(?P<med>[a-z][a-z0-9-]+)",
    r"\b(?P<med>[a-z][a-z0-9-]+)\s+(?:was\s+)?(?:stopped|discontinued|held)\b",
)
_ACTIVE_PATTERNS = (
    r"\b(?:start|started|continue|continued|prescribed|discharged on|taking|takes)\s+(?:the\s+)?(?P<med>[a-z][a-z0-9-]+)",
    r"\b(?:home meds?|medications?)\s*[:\-]\s*(?P<med>[a-z][a-z0-9-]+)",
)

_RULED_OUT_PATTERNS = (
    r"\b(?:rule out|ruled out|no evidence of|negative for)\s+(?P<dx>nstemi|stemi|mi|pneumonia|sepsis|stroke)\b",
)
_ACTIVE_DX_PATTERNS = (
    r"\b(?:diagnosed with|diagnosis(?: of)?|assessment)\s+(?P<dx>nstemi|stemi|mi|pneumonia|sepsis|stroke)\b",
    r"\b(?P<dx>nstemi|stemi|mi|pneumonia|sepsis|stroke)\b",
)

_ALLERGY_NAME_PATTERNS = (
    r"\b(?P<allergy>aspirin|penicillin|sulfa|latex|iodine|contrast|morphine)\s+allerg(?:y|ies)\b",
    r"\ballerg(?:y|ies)\s+to\s+(?P<allergy>aspirin|penicillin|sulfa|latex|iodine|contrast|morphine)\b",
    r"\ballergic\s+to\s+(?P<allergy>aspirin|penicillin|sulfa|latex|iodine|contrast|morphine)\b",
)

_ACTIVE_MEDICATION_CONTEXT = (
    "taking",
    "takes",
    "home med",
    "medications",
    "started",
    "continued",
    "continue",
    "prescribed",
    "discharged on",
    "administered",
)


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _section_category(text: str) -> str | None:
    match = _SECTION_CATEGORY_RE.search(text)
    if match:
        return match.group(1).strip().lower()
    return None


def _snippet(text: str, term: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    lowered = compact.lower()
    pos = lowered.find(term.lower())
    if pos == -1:
        return compact[:220]
    start = max(0, pos - 80)
    end = min(len(compact), pos + 140)
    return compact[start:end].strip()


def _add_evidence(
    bucket: dict[str, list[dict[str, str]]],
    key: str,
    chunk: ChunkResult,
    snippet_term: str,
) -> None:
    bucket.setdefault(key, []).append(
        {
            "source_chunk": str(chunk.chunk_id),
            "snippet": _snippet(chunk.text, snippet_term),
        }
    )


def _dedupe_evidence(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item["source_chunk"], item["snippet"])
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)
    return deduped


def _entity_evidence(chunks: list[ChunkResult]) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    allergies: dict[str, list[dict[str, str]]] = {}
    medications: dict[str, list[dict[str, str]]] = {}

    for chunk in chunks:
        section_category = _section_category(chunk.text)
        entities = extract_clinical_entities(
            chunk.text,
            section_category=section_category,
        )
        for allergy in entities.get("allergies", []):
            if isinstance(allergy, str):
                _add_evidence(allergies, _normalize(allergy), chunk, allergy)
        for medication in entities.get("medications", []):
            if isinstance(medication, str) and _is_active_medication_evidence(
                chunk.text,
                medication,
                section_category,
            ):
                _add_evidence(medications, _normalize(medication), chunk, medication)

        normalized = _normalize(chunk.text)
        for pattern in _ALLERGY_NAME_PATTERNS:
            for match in re.finditer(pattern, normalized):
                allergy = match.group("allergy")
                _add_evidence(allergies, _normalize(allergy), chunk, allergy)

    return allergies, medications


def _is_active_medication_evidence(
    text: str,
    medication: str,
    section_category: str | None,
) -> bool:
    if section_category in {"medications", "discharge", "hospital_course"}:
        return True

    normalized = _normalize(text)
    med = _normalize(medication)
    pos = normalized.find(med)
    if pos == -1:
        return False
    window = normalized[max(0, pos - 60):pos + len(med) + 60]
    if "allergy" in window or "allergic" in window:
        return False
    return any(context in window for context in _ACTIVE_MEDICATION_CONTEXT)


def _detect_allergy_medication_conflicts(chunks: list[ChunkResult]) -> list[Contradiction]:
    allergies, medications = _entity_evidence(chunks)
    conflicts: list[Contradiction] = []

    for item in sorted(set(allergies) & set(medications)):
        conflicts.append(
            {
                "conflict_type": "allergy_vs_medication",
                "severity": "high",
                "description": f"{item.title()} appears as both an allergy and an active/administered medication.",
                "evidence": _dedupe_evidence(allergies[item][:2] + medications[item][:2]),
            }
        )

    return conflicts


def _detect_smoking_conflicts(chunks: list[ChunkResult]) -> list[Contradiction]:
    former: list[dict[str, str]] = []
    current: list[dict[str, str]] = []

    for chunk in chunks:
        normalized = _normalize(chunk.text)
        for phrase in _SMOKING_FORMER:
            if phrase in normalized:
                former.append({"source_chunk": str(chunk.chunk_id), "snippet": _snippet(chunk.text, phrase)})
                break
        for phrase in _SMOKING_CURRENT:
            if phrase in normalized:
                current.append({"source_chunk": str(chunk.chunk_id), "snippet": _snippet(chunk.text, phrase)})
                break

    if former and current:
        return [
            {
                "conflict_type": "smoking_status",
                "severity": "moderate",
                "description": "Records contain both former-smoking and current-smoking language.",
                "evidence": _dedupe_evidence(former[:2] + current[:2]),
            }
        ]
    return []


def _medication_status_evidence(chunks: list[ChunkResult]) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    stopped: dict[str, list[dict[str, str]]] = {}
    active: dict[str, list[dict[str, str]]] = {}

    for chunk in chunks:
        normalized = _normalize(chunk.text)
        for pattern in _STOPPED_PATTERNS:
            for match in re.finditer(pattern, normalized):
                med = match.group("med")
                _add_evidence(stopped, med, chunk, med)
        for pattern in _ACTIVE_PATTERNS:
            for match in re.finditer(pattern, normalized):
                med = match.group("med")
                _add_evidence(active, med, chunk, med)

    return stopped, active


def _detect_medication_status_conflicts(chunks: list[ChunkResult]) -> list[Contradiction]:
    stopped, active = _medication_status_evidence(chunks)
    conflicts: list[Contradiction] = []

    for med in sorted(set(stopped) & set(active)):
        conflicts.append(
            {
                "conflict_type": "medication_status",
                "severity": "moderate",
                "description": f"{med.title()} appears in both stopped/discontinued and active/continued medication contexts.",
                "evidence": _dedupe_evidence(stopped[med][:2] + active[med][:2]),
            }
        )

    return conflicts


def _diagnosis_status_evidence(chunks: list[ChunkResult]) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    ruled_out: dict[str, list[dict[str, str]]] = {}
    active: dict[str, list[dict[str, str]]] = {}

    for chunk in chunks:
        normalized = _normalize(chunk.text)
        for pattern in _RULED_OUT_PATTERNS:
            for match in re.finditer(pattern, normalized):
                dx = match.group("dx")
                _add_evidence(ruled_out, dx, chunk, dx)
        for pattern in _ACTIVE_DX_PATTERNS:
            for match in re.finditer(pattern, normalized):
                dx = match.group("dx")
                # Avoid counting the same ruled-out phrase as active evidence.
                window_start = max(0, match.start() - 20)
                window = normalized[window_start:match.end()]
                if "ruled out" in window or "no evidence of" in window or "negative for" in window:
                    continue
                _add_evidence(active, dx, chunk, dx)

    return ruled_out, active


def _detect_diagnosis_status_conflicts(chunks: list[ChunkResult]) -> list[Contradiction]:
    ruled_out, active = _diagnosis_status_evidence(chunks)
    conflicts: list[Contradiction] = []

    for dx in sorted(set(ruled_out) & set(active)):
        conflicts.append(
            {
                "conflict_type": "diagnosis_status",
                "severity": "moderate",
                "description": f"{dx.upper()} appears in both ruled-out/negative and active diagnosis contexts.",
                "evidence": _dedupe_evidence(ruled_out[dx][:2] + active[dx][:2]),
            }
        )

    return conflicts


def detect_contradictions(chunks: list[ChunkResult]) -> list[Contradiction]:
    """Surface deterministic contradiction candidates from retrieved chunks."""
    conflicts: list[Contradiction] = []
    conflicts.extend(_detect_allergy_medication_conflicts(chunks))
    conflicts.extend(_detect_smoking_conflicts(chunks))
    conflicts.extend(_detect_medication_status_conflicts(chunks))
    conflicts.extend(_detect_diagnosis_status_conflicts(chunks))
    return conflicts
