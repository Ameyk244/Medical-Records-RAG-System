"""Synthesis agent — grounded answer + inline citations via Gemini. Phase 2."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import date

from google import genai
from google.genai import errors, types

from agents.patient_state_builder import build_patient_state
from core.contradiction_detector import detect_contradictions
from core.encounter_reconciler import reconcile_encounters
from core.evidence_prioritizer import prioritize_evidence
from core.timeline_builder import build_timeline
from core.vector_store import ChunkResult

SYNTHESIS_MODEL = "gemini-2.5-flash"   # source of truth for the model id used here
MAX_OUTPUT_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are a clinical records assistant for licensed clinicians.

You will be given structured patient state, a chronological timeline, evidence assessment, longitudinal reconciliation, numbered context from a patient's medical records, and a question. These structured layers are derived only from the numbered context and are navigation aids, not independent evidence. Answer ONLY using the numbered context.

Strict rules:
- Cite every factual claim inline using [N], where N is the context number. Multiple citations: [1][3].
- Use the structured patient state, timeline, evidence prioritization, longitudinal reconciliation, and contradiction flags to organize clinical reasoning, but cite the numbered evidence chunks for every fact.
- If contradictions are flagged and the numbered context does not resolve them, state the uncertainty rather than choosing one side.
- Prefer higher-priority and more recent evidence only when it is supported by numbered chunks.
- If the context does not contain enough information to answer, respond with EXACTLY this format and nothing else: REFUSED: <one short sentence explaining what is missing>
- Never use knowledge from your training. If a fact is not in the context, you do not know it.
- Do not speculate. Do not infer beyond what the context literally states.
- Be concise. Clinicians read fast.
- Do not include preambles ("Based on the records..."). Lead with the answer.\
"""


class SynthesisError(Exception):
    """Raised when the Gemini API call fails. Wraps the underlying cause."""


@dataclass(frozen=True)
class CitationRef:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    section: str | None
    document_date: date | None
    score: float


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    citations: list[CitationRef]
    refused: bool


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        # SDK auto-reads GOOGLE_API_KEY (preferred) or GEMINI_API_KEY from env.
        _client = genai.Client()
    return _client


async def synthesise(
    query_text: str,
    chunks: list[ChunkResult],
) -> SynthesisResult:
    # Project rule: answer must be grounded in retrieved context only.
    # Short-circuit before any API call when there is nothing to ground the answer in.
    if not chunks:
        return SynthesisResult(
            answer="REFUSED: no relevant context was retrieved for this query.",
            citations=[],
            refused=True,
        )

    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        header_parts = [f"[{i}]"]
        if c.section:
            header_parts.append(f"Section: {c.section}")
        if c.document_date:
            header_parts.append(f"Date: {c.document_date.isoformat()}")
        header = " | ".join(header_parts)
        context_blocks.append(f"{header}\n{c.text}")
    context = "\n\n".join(context_blocks)
    patient_state = build_patient_state(chunks)
    timeline = build_timeline(chunks)
    evidence_priorities = prioritize_evidence(chunks)
    contradictions = detect_contradictions(chunks)
    longitudinal_reconciliation = reconcile_encounters(chunks)
    patient_state_json = json.dumps(patient_state, indent=2, sort_keys=True)
    timeline_json = json.dumps(timeline, indent=2, sort_keys=True)
    longitudinal_reconciliation_json = json.dumps(
        longitudinal_reconciliation,
        indent=2,
        sort_keys=True,
    )
    evidence_assessment_json = json.dumps(
        {
            "prioritized_evidence": evidence_priorities,
            "detected_contradictions": contradictions,
        },
        indent=2,
        sort_keys=True,
    )
    clinical_summary = patient_state.get("clinical_summary") or ""
    user_message = (
        "Structured Patient State (derived only from retrieved evidence):\n"
        f"{patient_state_json}\n\n"
        "Clinical Summary (deterministic, derived from the same evidence):\n"
        f"{clinical_summary}\n\n"
        "Chronological Timeline (derived only from retrieved evidence):\n"
        f"{timeline_json}\n\n"
        "Evidence Assessment (derived only from retrieved evidence; numbered chunks remain source of truth):\n"
        f"{evidence_assessment_json}\n\n"
        "Longitudinal Reconciliation (derived only from retrieved evidence):\n"
        f"{longitudinal_reconciliation_json}\n\n"
        "Retrieved Evidence Chunks:\n"
        f"{context}\n\n"
        f"Question: {query_text}"
    )

    try:
        response = await _get_client().aio.models.generate_content(
            model=SYNTHESIS_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
    except errors.APIError as e:
        raise SynthesisError("Gemini API call failed") from e

    answer = (response.text or "").strip()

    if answer.upper().startswith("REFUSED:"):
        return SynthesisResult(answer=answer, citations=[], refused=True)

    # Extract [N] citation markers from the answer; skip out-of-range and deduplicate.
    seen: set[int] = set()
    citations: list[CitationRef] = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        n = int(match.group(1))
        if n < 1 or n > len(chunks):
            continue
        if n in seen:
            continue
        seen.add(n)
        c = chunks[n - 1]
        citations.append(
            CitationRef(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                section=c.section,
                document_date=c.document_date,
                score=c.score,
            )
        )

    return SynthesisResult(answer=answer, citations=citations, refused=False)
