"""Longitudinal reconciliation over retrieved clinical evidence.

This module keeps Phase 8 local and deterministic: it groups retrieved chunks
into encounter-like buckets, tracks clinical state changes, and links every
reconciled item back to chunk ids. It does not persist memory or change schema.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from core.clinical_entities import extract_clinical_entities
from core.contradiction_detector import detect_contradictions
from core.timeline_builder import build_timeline
from core.vector_store import ChunkResult

LongitudinalReconciliation = dict[str, Any]

_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)

_MED_CHANGE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("started", ("started", "start", "initiated", "began", "prescribed", "discharged on")),
    ("stopped", ("stopped", "stop", "discontinued", "held")),
    ("continued", ("continued", "continue", "resumed", "home meds", "taking")),
    ("changed", ("increased", "decreased", "changed", "adjusted")),
)

_DX_RULED_OUT = ("ruled out", "rule out", "no evidence of", "negative for")
_DX_RESOLVED = ("resolved", "improved", "stable")
_LAB_TREND_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("improving", ("downtrending", "decreasing", "improved", "improving", "lower")),
    ("worsening", ("uptrending", "increasing", "worsening", "rising", "elevated")),
    ("stable", ("stable", "unchanged")),
)

_CANONICAL: dict[str, dict[str, str]] = {
    "diagnoses": {
        "diabetes": "Type 2 Diabetes Mellitus",
        "dm2": "Type 2 Diabetes Mellitus",
        "t2dm": "Type 2 Diabetes Mellitus",
        "hypertension": "Hypertension",
        "htn": "Hypertension",
        "nstemi": "NSTEMI",
        "stemi": "STEMI",
        "cad": "Coronary Artery Disease",
        "hyperlipidemia": "Hyperlipidemia",
        "hld": "Hyperlipidemia",
    },
    "medications": {
        "aspirin": "Aspirin",
        "atorvastatin": "Atorvastatin",
        "metformin": "Metformin",
        "clopidogrel": "Clopidogrel",
        "ticagrelor": "Ticagrelor",
        "metoprolol": "Metoprolol",
        "lisinopril": "Lisinopril",
        "heparin": "Heparin",
    },
    "labs": {
        "hba1c": "HbA1c",
        "ldl": "LDL",
        "troponin": "Troponin",
        "glucose": "Glucose",
        "creatinine": "Creatinine",
        "hemoglobin": "Hemoglobin",
    },
    "risk_factors": {
        "smoker": "Smoking History",
        "smoking": "Smoking History",
        "tobacco": "Smoking History",
        "diabetes": "Type 2 Diabetes Mellitus",
        "hypertension": "Hypertension",
        "hyperlipidemia": "Hyperlipidemia",
        "family history": "Family History",
    },
}


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _canonical(category: str, value: str) -> str:
    key = _normalize(value)
    if not key:
        return ""
    mapped = _CANONICAL.get(category, {}).get(key)
    if mapped:
        return mapped
    if value.isupper() or any(ch.isdigit() for ch in value):
        return value
    return value.title()


def _section_category(chunk: ChunkResult) -> str | None:
    match = _SECTION_CATEGORY_RE.search(chunk.text)
    if match:
        return match.group(1).strip().lower()
    section = _normalize(chunk.section or "")
    if "discharge" in section:
        return "discharge"
    if "history" in section:
        return "history"
    if "medication" in section:
        return "medications"
    if "diagnos" in section or "assessment" in section:
        return "diagnosis"
    if "lab" in section:
        return "labs"
    return None


def _encounter_key(chunk: ChunkResult) -> str:
    if chunk.document_date:
        return chunk.document_date.isoformat()
    match = _DATE_RE.search(chunk.text)
    if match:
        return match.group(0).replace("/", "-")
    return "unknown"


def _snippet(text: str, term: str) -> str:
    without_metadata = re.sub(
        r"(?im)^(Section|Section category|Clinical entities):.*$",
        " ",
        text,
    )
    compact = re.sub(r"\s+", " ", without_metadata).strip()
    pos = compact.lower().find(term.lower())
    if pos == -1:
        return compact[:220]
    return compact[max(0, pos - 80):min(len(compact), pos + 140)].strip()


def _append_unique(target: list[dict[str, Any]], item: dict[str, Any]) -> None:
    item_key = (
        item.get("name") or item.get("fact") or item.get("lab") or item.get("medication"),
        item.get("status") or item.get("trend") or item.get("state") or item.get("action"),
        tuple(item.get("source_chunks", [])),
    )
    for existing in target:
        existing_key = (
            existing.get("name") or existing.get("fact") or existing.get("lab") or existing.get("medication"),
            existing.get("status") or existing.get("trend") or existing.get("state") or existing.get("action"),
            tuple(existing.get("source_chunks", [])),
        )
        if existing_key == item_key:
            return
    target.append(item)


def _medication_actions(text: str, medication: str, section_category: str | None) -> list[str]:
    normalized = _normalize(text)
    med = _normalize(medication)
    pos = normalized.find(med)
    window = normalized[max(0, pos - 80):pos + len(med) + 80] if pos != -1 else normalized
    actions: list[str] = []
    if "discharged on" in window:
        actions.append("current_at_discharge")
    for action, patterns in _MED_CHANGE_PATTERNS:
        if any(pattern in window for pattern in patterns):
            actions.append(action)
    if actions:
        return list(dict.fromkeys(actions))
    if section_category == "discharge":
        return ["current_at_discharge"]
    if section_category == "medications":
        return ["listed"]
    return ["mentioned"]


def _diagnosis_status(text: str, diagnosis: str, section_category: str | None) -> str:
    normalized = _normalize(text)
    dx = _normalize(diagnosis)
    pos = normalized.find(dx)
    window = normalized[max(0, pos - 80):pos + len(dx) + 80] if pos != -1 else normalized
    if any(phrase in window for phrase in _DX_RULED_OUT):
        return "ruled_out"
    if any(phrase in window for phrase in _DX_RESOLVED):
        return "resolved_or_stable"
    if section_category in {"diagnosis", "hospital_course", "discharge"}:
        return "active_or_current"
    if section_category == "history":
        return "historical"
    return "mentioned"


def _lab_trend(text: str, lab: str) -> str:
    normalized = _normalize(text)
    lab_key = _normalize(lab)
    pos = normalized.find(lab_key)
    window = normalized[max(0, pos - 100):pos + len(lab_key) + 120] if pos != -1 else normalized
    for trend, patterns in _LAB_TREND_PATTERNS:
        if any(pattern in window for pattern in patterns):
            return trend
    if re.search(rf"\b{re.escape(lab_key)}\b\s*[:=]?\s*[<>]?\s*\d", window):
        return "measured"
    return "mentioned"


def _current_or_historical_state(section_category: str | None, text: str) -> str:
    normalized = _normalize(text)
    if section_category == "discharge" or "discharged on" in normalized:
        return "current"
    if section_category == "history" or "past medical history" in normalized:
        return "historical"
    if "former smoker" in normalized or "history of" in normalized:
        return "historical"
    if section_category in {"diagnosis", "hospital_course"}:
        return "current"
    if "current" in normalized or "currently" in normalized:
        return "current"
    return "mentioned"


def reconcile_encounters(chunks: list[ChunkResult]) -> LongitudinalReconciliation:
    """Reconcile encounter-level trajectory from retrieved chunks only."""
    encounter_map: dict[str, dict[str, Any]] = {}
    current_facts: list[dict[str, Any]] = []
    historical_facts: list[dict[str, Any]] = []
    medication_evolution: list[dict[str, Any]] = []
    diagnosis_status: list[dict[str, Any]] = []
    lab_trends: list[dict[str, Any]] = []

    for chunk in chunks:
        key = _encounter_key(chunk)
        section_category = _section_category(chunk)
        entities = extract_clinical_entities(chunk.text, section_category=section_category)
        encounter = encounter_map.setdefault(
            key,
            {
                "encounter_date": key,
                "source_chunks": [],
                "sections": [],
                "diagnoses": [],
                "medications": [],
                "procedures": [],
                "labs": [],
                "risk_factors": [],
                "summary": "",
            },
        )
        chunk_id = str(chunk.chunk_id)
        if chunk_id not in encounter["source_chunks"]:
            encounter["source_chunks"].append(chunk_id)
        if chunk.section and chunk.section not in encounter["sections"]:
            encounter["sections"].append(chunk.section)

        for category in ("diagnoses", "medications", "procedures", "labs", "risk_factors"):
            values = entities.get(category)
            if not isinstance(values, list):
                continue
            for value in values:
                canonical = _canonical(category, value)
                if canonical and canonical not in encounter[category]:
                    encounter[category].append(canonical)

        state = _current_or_historical_state(section_category, chunk.text)
        for category in ("diagnoses", "medications", "risk_factors"):
            values = entities.get(category)
            if not isinstance(values, list):
                continue
            for value in values:
                fact = _canonical(category, value)
                item = {
                    "fact": fact,
                    "category": category,
                    "state": state,
                    "encounter_date": key,
                    "source_chunks": [chunk_id],
                }
                if state == "historical":
                    _append_unique(historical_facts, item)
                elif state == "current":
                    _append_unique(current_facts, item)

        medications = entities.get("medications")
        if isinstance(medications, list):
            for medication in medications:
                canonical = _canonical("medications", medication)
                for action in _medication_actions(chunk.text, medication, section_category):
                    _append_unique(
                        medication_evolution,
                        {
                            "medication": canonical,
                            "action": action,
                            "encounter_date": key,
                            "source_chunks": [chunk_id],
                            "evidence_snippet": _snippet(chunk.text, medication),
                        },
                    )

        diagnoses = entities.get("diagnoses")
        if isinstance(diagnoses, list):
            for diagnosis in diagnoses:
                canonical = _canonical("diagnoses", diagnosis)
                _append_unique(
                    diagnosis_status,
                    {
                        "name": canonical,
                        "status": _diagnosis_status(chunk.text, diagnosis, section_category),
                        "encounter_date": key,
                        "source_chunks": [chunk_id],
                        "evidence_snippet": _snippet(chunk.text, diagnosis),
                    },
                )

        labs = entities.get("labs")
        if isinstance(labs, list):
            for lab in labs:
                canonical = _canonical("labs", lab)
                _append_unique(
                    lab_trends,
                    {
                        "lab": canonical,
                        "trend": _lab_trend(chunk.text, lab),
                        "encounter_date": key,
                        "source_chunks": [chunk_id],
                        "evidence_snippet": _snippet(chunk.text, lab),
                    },
                )

    encounters = sorted(
        encounter_map.values(),
        key=lambda item: ("1" if item["encounter_date"] == "unknown" else "0", item["encounter_date"]),
    )
    for encounter in encounters:
        summary_parts = []
        if encounter["diagnoses"]:
            summary_parts.append(f"diagnoses: {', '.join(encounter['diagnoses'])}")
        if encounter["procedures"]:
            summary_parts.append(f"procedures: {', '.join(encounter['procedures'])}")
        if encounter["medications"]:
            summary_parts.append(f"medications: {', '.join(encounter['medications'])}")
        if encounter["labs"]:
            summary_parts.append(f"labs: {', '.join(encounter['labs'])}")
        encounter["summary"] = "; ".join(summary_parts)

    return {
        "encounters": encounters,
        "current_facts": current_facts,
        "historical_facts": historical_facts,
        "medication_evolution": sorted(medication_evolution, key=lambda item: item["encounter_date"]),
        "diagnosis_status": sorted(diagnosis_status, key=lambda item: item["encounter_date"]),
        "lab_trends": sorted(lab_trends, key=lambda item: item["encounter_date"]),
        "timeline_events": build_timeline(chunks),
        "unresolved_conflicts": detect_contradictions(chunks),
    }
