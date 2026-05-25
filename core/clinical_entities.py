"""Lightweight clinical entity extraction for medical record chunks.

This is intentionally deterministic and dependency-free. It extracts common
clinical concepts that help downstream retrieval and synthesis understand
patient state without introducing a heavy NLP stack or changing persistence.
"""

from __future__ import annotations

import re
from typing import Any

ClinicalEntities = dict[str, dict[str, str] | list[str]]

_DIAGNOSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diabetes", ("diabetes", "diabetes mellitus", "dm2", "type 2 diabetes", "t2dm")),
    ("hypertension", ("hypertension", "htn", "high blood pressure")),
    ("NSTEMI", ("nstemi", "non st elevation myocardial infarction")),
    ("STEMI", ("stemi", "st elevation myocardial infarction")),
    ("CAD", ("cad", "coronary artery disease")),
    ("hyperlipidemia", ("hyperlipidemia", "hld", "dyslipidemia")),
    ("heart failure", ("heart failure", "chf", "hfref", "hfpef")),
    ("COPD", ("copd", "chronic obstructive pulmonary disease")),
    ("CKD", ("ckd", "chronic kidney disease")),
    ("atrial fibrillation", ("atrial fibrillation", "afib", "a fib")),
)

_MEDICATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aspirin", ("aspirin", "asa")),
    ("atorvastatin", ("atorvastatin", "lipitor")),
    ("rosuvastatin", ("rosuvastatin", "crestor")),
    ("metformin", ("metformin",)),
    ("insulin", ("insulin",)),
    ("clopidogrel", ("clopidogrel", "plavix")),
    ("ticagrelor", ("ticagrelor", "brilinta")),
    ("metoprolol", ("metoprolol",)),
    ("lisinopril", ("lisinopril",)),
    ("amlodipine", ("amlodipine",)),
    ("heparin", ("heparin",)),
    ("nitroglycerin", ("nitroglycerin", "nitro")),
)

_ALLERGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("penicillin", ("penicillin", "pcn")),
    ("sulfa", ("sulfa", "sulfonamide", "sulfamethoxazole")),
    ("aspirin", ("aspirin", "asa")),
    ("latex", ("latex",)),
    ("iodine", ("iodine", "contrast")),
    ("morphine", ("morphine",)),
)

_PROCEDURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PCI", ("pci", "percutaneous coronary intervention")),
    ("angiography", ("angiography", "coronary angiography", "cardiac catheterization", "catheterization")),
    ("stent placement", ("stent placement", "stent placed", "drug eluting stent", "des")),
    ("CABG", ("cabg", "coronary artery bypass")),
    ("echocardiogram", ("echocardiogram", "echo")),
    ("angioplasty", ("angioplasty",)),
)

_LABS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HbA1c", ("hba1c", "a1c")),
    ("LDL", ("ldl",)),
    ("troponin", ("troponin", "trop")),
    ("glucose", ("glucose", "blood sugar")),
    ("creatinine", ("creatinine", "cr")),
    ("hemoglobin", ("hemoglobin", "hgb", "hb")),
    ("WBC", ("wbc", "white blood cell")),
    ("platelets", ("platelet", "platelets")),
)

_RISK_FACTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("smoker", ("smoker", "smoking", "tobacco", "former smoker", "current smoker")),
    ("obesity", ("obesity", "obese", "bmi")),
    ("hypertension", ("hypertension", "htn", "high blood pressure")),
    ("diabetes", ("diabetes", "diabetes mellitus", "dm2", "t2dm")),
    ("hyperlipidemia", ("hyperlipidemia", "hld", "dyslipidemia")),
    ("family history", ("family history", "fhx")),
)

_DATE_PATTERN = r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"


def _normalize(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s.:%]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    if " " in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def _extract_vocab(text: str, vocabulary: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    normalized_text = _normalize(text)
    values: list[str] = []
    for canonical, variants in vocabulary:
        if any(_contains_term(normalized_text, variant) for variant in variants):
            values.append(canonical)
    return values


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _extract_demographics(text: str, section_category: str | None) -> dict[str, str]:
    demographics: dict[str, str] = {}

    name_match = re.search(
        r"(?i)\b(?:patient\s+name|name)\s*[:\-]\s*([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})",
        text,
    )
    if not name_match and section_category == "demographics":
        name_match = re.search(
            r"(?i)\bpatient\s*[:\-]\s*([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})",
            text,
        )
    if name_match:
        demographics["patient_name"] = name_match.group(1).strip(" .,")

    age_match = re.search(r"(?i)\bage\s*[:\-]?\s*(\d{1,3})\b", text)
    if not age_match:
        age_match = re.search(r"(?i)\b(\d{1,3})\s*(?:year|yr)s?\s*old\b", text)
    if age_match:
        demographics["age"] = age_match.group(1)

    sex_match = re.search(r"(?i)\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b", text)
    if sex_match:
        raw_sex = sex_match.group(1).lower()
        demographics["sex"] = {
            "m": "male",
            "f": "female",
        }.get(raw_sex, raw_sex)

    dob_match = re.search(rf"(?i)\b(?:dob|date of birth|birth date)\s*[:\-]?\s*({_DATE_PATTERN})\b", text)
    if dob_match:
        demographics["DOB"] = dob_match.group(1)

    return demographics


def _extract_medications(text: str, section_category: str | None) -> list[str]:
    if section_category == "allergies":
        return []
    return _dedupe(_extract_vocab(text, _MEDICATIONS))


def _extract_allergies(text: str, section_category: str | None) -> list[str]:
    if section_category == "allergies":
        return _dedupe(_extract_vocab(text, _ALLERGIES))

    contexts: list[str] = []
    for match in re.finditer(r"(?i)\ballerg(?:y|ies)\s*[:\-]?\s*([^.;\n]+)", text):
        contexts.append(match.group(1))
    for match in re.finditer(r"(?i)\ballergic\s+to\s+([^.;\n]+)", text):
        contexts.append(match.group(1))

    if not contexts:
        return []

    return _dedupe(_extract_vocab(" ".join(contexts), _ALLERGIES))


def _extract_labs(text: str) -> list[str]:
    labs = _extract_vocab(text, _LABS)
    normalized_labs = {lab.lower() for lab in labs}

    for canonical, variants in _LABS:
        if canonical.lower() in normalized_labs:
            continue
        for variant in variants:
            pattern = rf"(?i)\b{re.escape(variant)}\b\s*[:=]?\s*(?:is|was|of)?\s*([<>]?\s*\d+(?:\.\d+)?)"
            if re.search(pattern, text):
                labs.append(canonical)
                normalized_labs.add(canonical.lower())
                break

    return _dedupe(labs)


def extract_clinical_entities(
    text: str,
    *,
    section_category: str | None = None,
) -> ClinicalEntities:
    """Extract a standardized clinical entity dictionary from chunk text."""
    demographics = _extract_demographics(text, section_category)

    return {
        "demographics": demographics,
        "diagnoses": _dedupe(_extract_vocab(text, _DIAGNOSES)),
        "medications": _extract_medications(text, section_category),
        "allergies": _extract_allergies(text, section_category),
        "procedures": _dedupe(_extract_vocab(text, _PROCEDURES)),
        "labs": _extract_labs(text),
        "risk_factors": _dedupe(_extract_vocab(text, _RISK_FACTORS)),
    }


def has_clinical_entities(entities: ClinicalEntities) -> bool:
    return any(value for value in entities.values())


def format_clinical_entities(entities: ClinicalEntities) -> str:
    """Render compact entity metadata for embedding enrichment."""
    if not has_clinical_entities(entities):
        return ""

    lines: list[str] = ["Clinical entities:"]
    demographics = entities.get("demographics")
    if isinstance(demographics, dict) and demographics:
        ordered_keys = ("patient_name", "age", "sex", "DOB")
        values = [
            f"{key}: {demographics[key]}"
            for key in ordered_keys
            if demographics.get(key)
        ]
        if values:
            lines.append(f"Demographics: {', '.join(values)}")

    labels = (
        ("diagnoses", "Diagnoses"),
        ("medications", "Medications"),
        ("allergies", "Allergies"),
        ("procedures", "Procedures"),
        ("labs", "Labs"),
        ("risk_factors", "Risk factors"),
    )
    for key, label in labels:
        values = entities.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{label}: {', '.join(values)}")

    return "\n".join(lines)
