"""Synthesis agent — grounded answer + inline citations via Gemini. Phase 2."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date

from google import genai
from google.genai import errors, types

from core.vector_store import ChunkResult

SYNTHESIS_MODEL = "gemini-2.5-flash"   # source of truth for the model id used here
MAX_OUTPUT_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are a clinical records assistant for licensed clinicians.

You will be given numbered context from a patient's medical records and a question. Answer ONLY using the provided context.

Strict rules:
- Cite every factual claim inline using [N], where N is the context number. Multiple citations: [1][3].
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
    user_message = f"Context:\n{context}\n\nQuestion: {query_text}"

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
