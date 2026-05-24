# Claude Code — Project Kickoff Prompt

> Paste this into Claude Code at the start of every session.

---

## What we are building

A **Medical Records RAG System** — a multi-agent AI pipeline that lets clinicians query patient records in natural language and get grounded, cited answers. Every query is access-controlled and audit-logged.

Repo: https://github.com/Ameyk244/Medical-Records-RAG-System

Read `CLAUDE.md` in the repo root before doing anything else. It is the source of truth for stack, structure, rules, and model usage.

---

## Model usage (strict)

- Use **Opus (`claude-opus-4-6`)** for: architecture decisions, agent design, orchestration logic, anything that requires reasoning through tradeoffs
- Use **Sonnet (`claude-sonnet-4-6`)** for: all code writing, tests, configs, boilerplate, file creation

Do not use Opus for implementation tasks. Do not use Sonnet for architectural decisions.

---

## Our goal

Build a working end-to-end system across 5 phases:

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Ingestion pipeline — parse, chunk, embed, store | 🔲 not started |
| 2 | Query + synthesis — retrieval + grounded Claude answer | 🔲 not started |
| 3 | Orchestrator + audit — routing, access control, audit log | 🔲 not started |
| 4 | API + auth — FastAPI routes, JWT, role-based access | 🔲 not started |
| 5 | Hardening — guardrails, rate limiting, PII handling, tests | 🔲 not started |

We work one phase at a time. Do not jump ahead.

---

## Agents we are building

1. **Orchestrator** — central router, sequences all other agents
2. **Ingestion agent** — upload → parse → chunk (512 tokens, 50 overlap) → embed → upsert → register
3. **Query agent** — embed query → pgvector similarity search → filter by patient_id → top 5 chunks
4. **Synthesis agent** — Claude API call, grounded answer + inline citations, refuses if context insufficient
5. **Audit agent** — access check before retrieval, append-only response log after synthesis

---

## End product

A FastAPI service where a clinician can:

1. `POST /ingest` — upload a patient record (PDF, FHIR JSON, HL7, CSV)
2. `POST /query` — ask a natural language question about a patient
3. Get back a **cited, grounded answer** with sources
4. Every action logged to an immutable audit trail

---

## Rules (non-negotiable)

- Synthesis agent must refuse to answer if retrieved context does not support it
- Audit agent runs on every single query — no exceptions
- Embeddings: Voyage `voyage-3-large` only (dim 1024) — no other providers
- Chunk text only in vector store — no raw PII
- Audit log is append-only — no updates, no deletes

---

## How we work together

1. **Read before write** — explore existing files before generating anything
2. **Report before act** — tell me what you plan to do, wait for go-ahead on anything destructive
3. **One phase at a time** — complete and confirm each phase before moving to the next
4. **Ask if unclear** — do not assume, especially around data models and access control logic

---

## Start of session checklist

Before writing any code:
- [ ] Read `CLAUDE.md`
- [ ] Run `ls -la` and map current repo structure
- [ ] Identify which phase we are on
- [ ] State what you plan to implement this session
- [ ] Wait for confirmation before starting
