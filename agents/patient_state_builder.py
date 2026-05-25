"""Build structured patient state from retrieved evidence chunks.

The state is derived only from retrieved chunks. It gives Gemini a consolidated
clinical view before synthesis while preserving strict grounding: the numbered
chunks remain the only citeable evidence.
"""

from __future__ import annotations

import re
from copy import deepcopy

from core.clinical_entities import ClinicalEntities, extract_clinical_entities
from core.vector_store import ChunkResult

PatientState = dict[str, dict[str, str] | list[str] | str]

_SECTION_CATEGORY_RE = re.compile(r"(?im)^Section category:\s*([a-z_]+)\s*$")

_LABEL_TO_KEY = {
    "Diagnoses": "diagnoses",
    "Medications": "medications",
    "Allergies": "allergies",
    "Procedures": "procedures",
    "Labs": "labs",
    "Risk factors": "risk_factors",
}

_CANONICAL_ENTITIES: dict[str, dict[str, str]] = {
    "diagnoses": {
        "diabetes": "Type 2 Diabetes Mellitus",
        "diabetes mellitus": "Type 2 Diabetes Mellitus",
        "dm2": "Type 2 Diabetes Mellitus",
        "t2dm": "Type 2 Diabetes Mellitus",
        "hypertension": "Hypertension",
        "htn": "Hypertension",
        "nstemi": "NSTEMI",
        "stemi": "STEMI",
        "cad": "Coronary Artery Disease",
        "hyperlipidemia": "Hyperlipidemia",
        "heart failure": "Heart Failure",
        "copd": "COPD",
        "ckd": "Chronic Kidney Disease",
        "atrial fibrillation": "Atrial Fibrillation",
    },
    "medications": {
        "aspirin": "Aspirin",
        "atorvastatin": "Atorvastatin",
        "rosuvastatin": "Rosuvastatin",
        "metformin": "Metformin",
        "insulin": "Insulin",
        "clopidogrel": "Clopidogrel",
        "ticagrelor": "Ticagrelor",
        "metoprolol": "Metoprolol",
        "lisinopril": "Lisinopril",
        "amlodipine": "Amlodipine",
        "heparin": "Heparin",
        "nitroglycerin": "Nitroglycerin",
    },
    "allergies": {
        "penicillin": "Penicillin",
        "sulfa": "Sulfa",
        "aspirin": "Aspirin",
        "latex": "Latex",
        "iodine": "Iodine/Contrast",
        "contrast": "Iodine/Contrast",
        "morphine": "Morphine",
    },
    "procedures": {
        "pci": "PCI",
        "angiography": "Angiography",
        "stent placement": "Stent Placement",
        "cabg": "CABG",
        "echocardiogram": "Echocardiogram",
        "angioplasty": "Angioplasty",
    },
    "labs": {
        "hba1c": "HbA1c",
        "a1c": "HbA1c",
        "ldl": "LDL",
        "troponin": "Troponin",
        "glucose": "Glucose",
        "creatinine": "Creatinine",
        "hemoglobin": "Hemoglobin",
        "wbc": "WBC",
        "platelets": "Platelets",
    },
    "risk_factors": {
        "smoker": "Smoking History",
        "smoking": "Smoking History",
        "tobacco": "Smoking History",
        "obesity": "Obesity",
        "hypertension": "Hypertension",
        "diabetes": "Type 2 Diabetes Mellitus",
        "hyperlipidemia": "Hyperlipidemia",
        "family history": "Family History",
    },
}


def _empty_state() -> PatientState:
    return {
        "demographics": {},
        "diagnoses": [],
        "medications": [],
        "allergies": [],
        "procedures": [],
        "labs": [],
        "risk_factors": [],
        "clinical_summary": "",
    }


def _norm_key(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _canonicalize(category: str, value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    canonical = _CANONICAL_ENTITIES.get(category, {}).get(_norm_key(value))
    if canonical:
        return canonical
    if value.isupper() or any(ch.isdigit() for ch in value):
        return value
    return value.title()


def _append_unique(state: PatientState, category: str, values: list[str]) -> None:
    target = state[category]
    if not isinstance(target, list):
        return

    seen = {_norm_key(item) for item in target}
    for value in values:
        canonical = _canonicalize(category, value)
        key = _norm_key(canonical)
        if canonical and key not in seen:
            target.append(canonical)
            seen.add(key)


def _merge_demographics(state: PatientState, demographics: dict[str, str]) -> None:
    target = state["demographics"]
    if not isinstance(target, dict):
        return
    for key in ("patient_name", "age", "sex", "DOB"):
        value = demographics.get(key)
        if value and not target.get(key):
            target[key] = value


def _infer_section_category(text: str) -> str | None:
    match = _SECTION_CATEGORY_RE.search(text)
    if match:
        return match.group(1).strip().lower()
    return None


def _parse_entity_block(text: str) -> ClinicalEntities:
    entities: ClinicalEntities = {
        "demographics": {},
        "diagnoses": [],
        "medications": [],
        "allergies": [],
        "procedures": [],
        "labs": [],
        "risk_factors": [],
    }

    if "Clinical entities:" not in text:
        return entities

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"Clinical entities:", "Section category:"}:
            continue
        if line.startswith("Demographics:"):
            demographics: dict[str, str] = {}
            payload = line.split(":", 1)[1]
            for item in payload.split(","):
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    demographics[key] = value
            entities["demographics"] = demographics
            continue

        for label, key in _LABEL_TO_KEY.items():
            prefix = f"{label}:"
            if line.startswith(prefix):
                values = [
                    value.strip()
                    for value in line[len(prefix):].split(",")
                    if value.strip()
                ]
                entities[key] = values
                break

    return entities


def _merge_entities(state: PatientState, entities: ClinicalEntities) -> None:
    demographics = entities.get("demographics")
    if isinstance(demographics, dict):
        _merge_demographics(state, demographics)

    for category in ("diagnoses", "medications", "allergies", "procedures", "labs", "risk_factors"):
        values = entities.get(category)
        if isinstance(values, list):
            _append_unique(state, category, values)


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _build_clinical_summary(state: PatientState) -> str:
    demographics = state["demographics"]
    diagnoses = state["diagnoses"]
    medications = state["medications"]
    allergies = state["allergies"]
    procedures = state["procedures"]
    labs = state["labs"]
    risk_factors = state["risk_factors"]

    parts: list[str] = []
    if isinstance(demographics, dict):
        age = demographics.get("age")
        sex = demographics.get("sex")
        name = demographics.get("patient_name")
        descriptors = []
        if age:
            descriptors.append(f"{age}-year-old")
        if sex:
            descriptors.append(sex)
        if descriptors:
            parts.append(f"{' '.join(descriptors).capitalize()} patient")
        elif name:
            parts.append(f"Patient {name}")

    if isinstance(diagnoses, list) and diagnoses:
        diagnoses_text = _join(diagnoses)
        if parts:
            parts[0] = f"{parts[0]} with {diagnoses_text}"
        else:
            parts.append(f"Patient with {diagnoses_text}")

    if isinstance(risk_factors, list) and risk_factors:
        parts.append(f"Risk factors include {_join(risk_factors)}")

    if isinstance(procedures, list) and procedures:
        parts.append(f"Underwent {_join(procedures)}")

    if isinstance(medications, list) and medications:
        parts.append(f"Medications include {_join(medications)}")

    if isinstance(allergies, list) and allergies:
        parts.append(f"Allergies include {_join(allergies)}")

    if isinstance(labs, list) and labs:
        parts.append(f"Relevant labs include {_join(labs)}")

    if not parts:
        return ""

    summary = ". ".join(parts)
    return summary if summary.endswith(".") else f"{summary}."


def build_patient_state(chunks: list[ChunkResult]) -> PatientState:
    """Build a consolidated patient state from retrieved chunks only."""
    state = _empty_state()

    for chunk in chunks:
        # Prefer metadata injected during Phase 4, then fall back to extraction
        # for older chunks that have not been re-ingested.
        _merge_entities(state, _parse_entity_block(chunk.text))
        _merge_entities(
            state,
            extract_clinical_entities(
                chunk.text,
                section_category=_infer_section_category(chunk.text),
            ),
        )

    final_state = deepcopy(state)
    final_state["clinical_summary"] = _build_clinical_summary(final_state)
    return final_state
