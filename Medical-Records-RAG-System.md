# Medical Records RAG System

## Repository
https://github.com/Ameyk244/Medical-Records-RAG-System

## Model usage

**For Claude Code (development assistant):**
- Opus (`claude-opus-4-6`): architecture decisions, agent orchestration logic, complex reasoning tasks
- Sonnet (`claude-sonnet-4-6`): all implementation — writing code, tests, configs, boilerplate

**For the synthesis agent (runtime LLM):**
- Gemini (`gemini-2.5-flash`): grounded answer generation with inline citations. Originally specced as Anthropic Claude; switched to Google Gemini in Phase 2 because Anthropic API access required payment. Free tier covers Phase 2/3/4 development. Synthesis is a pluggable component — provider can be swapped without changing the orchestrator, query, or audit layers.

## Project overview
Multi-agent RAG system for clinicians to query patient records in natural language.
Returns grounded answers with cited sources. Every query is audit-logged.

## Stack
- Python 3.13 (older 3.11+ also OK, except `voyageai>=0.3.3` is required for 3.13)
- Google Gemini API (`gemini-2.5-flash`) — synthesis LLM
- Voyage AI embeddings (`voyage-3-large`, dim 1024)
- pgvector (Postgres extension) + HNSW index
- pgcrypto (Postgres extension) — encrypts `audit_log.query_text_enc` / `response_text_enc`
- FastAPI with Pydantic v2
- S3-compatible object storage (MinIO locally, AWS S3 in prod)
- Celery + Redis — async ingestion queue
- structlog — JSON request logging
- prometheus-fastapi-instrumentator — `/metrics` endpoint
- slowapi — rate limiting, Redis-backed
- Alembic — schema migrations
- python-jose + passlib[bcrypt] — JWT + password hashing

## Folder structure
```
medical-rag/
├── agents/
│   ├── orchestrator.py        # routes tasks between agents
│   ├── ingestion_agent.py     # parse, chunk, embed, upsert
│   ├── ingestion_worker.py    # Celery task — async wrapper around ingest_document
│   ├── celery_app.py          # Celery application instance
│   ├── query_agent.py         # embed query, retrieve top-k chunks
│   ├── synthesis_agent.py     # Gemini call, grounded answer + citations
│   └── audit_agent.py         # access check + encrypted response logging
├── core/
│   ├── embeddings.py          # voyage-3-large wrapper
│   ├── vector_store.py        # pgvector upsert + similarity search
│   ├── document_parser.py     # 9 formats — see Data sources below
│   ├── chunker.py             # token-aware chunker
│   ├── s3.py                  # MinIO/S3 upload wrapper
│   ├── cache.py               # Redis query cache, sha256-keyed
│   ├── middleware.py          # security headers, X-Request-ID, structlog, Prometheus
│   └── auth.py                # JWT + bcrypt + users-table auth helpers
├── api/
│   └── routes.py              # FastAPI endpoints (auth, ingest, query, health)
├── db/
│   ├── models.py              # SQLAlchemy models
│   └── init.py                # LOCAL DEV ONLY — Alembic for production
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_initial.py       # full schema + append-only trigger
│       └── 0002_encrypted_audit.py  # pgcrypto + encrypted columns
├── website/
│   └── index.html             # marketing site, vanilla HTML/CSS/JS
├── tests/
├── .env.example
├── docker-compose.yml         # postgres + redis + minio
└── Medical-Records-RAG-System.md   # this file (read first)
```

## Agents
- **Orchestrator** — receives clinician query, sequences agent calls, assembles response
- **Ingestion agent** — triggered on upload; parse → chunk → embed → upsert → register. Sync core function, also invoked from a Celery worker for HTTP path.
- **Query agent** — embed query → similarity search → filter by patient_id + date → top-k chunks
- **Synthesis agent** — Gemini API call; answer must be grounded in retrieved context only
- **Audit agent** — runs before retrieval (access check) and after synthesis (response log). Writes pgcrypto-encrypted `query_text_enc` + `response_text_enc` for new rows.

## Data flow
```
HTTP request → middleware (security, X-Request-ID) → rate limit → JWT check + jti blacklist
            → orchestrator → audit (access check) → cache lookup
            → query agent → synthesis agent (Gemini) → audit (log encrypted)
            → cache populate → response
```

For ingest:
```
POST /ingest → S3 upload (eager) → Celery enqueue → 202 + task_id
                                  ↓ (worker)
                            parse → chunk → embed → upsert
                                  ↓
                        GET /ingest/{task_id}/status (polled by client)
```

## Data sources supported

Implemented in `core/document_parser.py`. `source_type` field on `POST /ingest`:

| `source_type` | Format | Parser library |
|---|---|---|
| `pdf` | PDF | pdfplumber |
| `fhir_json` | FHIR R4 JSON | stdlib json |
| `hl7` | HL7 v2 | python-hl7 |
| `csv` | CSV | stdlib csv |
| `dicom_meta` | DICOM metadata (no pixels) | pydicom |
| `ccd_xml` | CCD / C-CDA XML | stdlib ElementTree |
| `txt` | Plain text | stdlib (ALL CAPS heading heuristic) |
| `docx` | Microsoft Word | python-docx |
| `xlsx` | Excel spreadsheet | openpyxl |

## API endpoints

| Method | Path | Auth | Rate limit | Notes |
|---|---|---|---|---|
| `POST` | `/auth/login` | none | 5/min (IP) | returns `{access_token, refresh_token, token_type}` |
| `POST` | `/auth/register` | admin | 5/min (IP) | role: `clinician` / `admin` / `readonly` |
| `POST` | `/auth/refresh` | none | — | accepts `{refresh_token}` → new access token |
| `POST` | `/auth/logout` | bearer | — | blacklists `jti`, revokes refresh tokens for user |
| `POST` | `/ingest` | bearer | 10/min (IP) | enqueues Celery task, returns 202 `{task_id}` |
| `GET` | `/ingest/{task_id}/status` | bearer | — | `{state, document_id?, error?}` |
| `POST` | `/query` | bearer | 30/min (IP) | Redis-cached, `X-Cache: HIT/MISS` header |
| `GET` | `/health` | none | — | always 200; `{status, checks: {db,redis,s3}}` |
| `GET` | `/health/ready` | none | — | 200 or 503 (Kubernetes readiness) |
| `GET` | `/health/live` | none | — | always 200 (liveness) |
| `GET` | `/metrics` | none | — | Prometheus, auto-instrumented |
| `GET` | `/docs` | none | — | Swagger UI |

## Build phases
1. Ingestion pipeline — parse docs, chunk, embed, store (no agents yet) ✓
2. Query + synthesis — retrieval + Gemini grounded answer with citations ✓
3. Orchestrator + audit — routing layer, access control, append-only audit log ✓
4. API + auth — FastAPI routes, JWT, role-based access ✓
5. Hardening — auth (users table, refresh tokens, lockout), rate limiting, security headers, encryption-at-rest, async ingestion, query cache, extended parsers, observability ✓

## Phase 5 — completed in this session

- [x] Real `users` table replaces hardcoded user; bcrypt-hashed passwords; 5-attempt lockout (15min); JWT `jti` claim + `revoked_tokens` blacklist; SHA256-hashed refresh tokens, 7-day expiry, mass-revoke on logout
- [x] `POST /auth/register` (admin-only) + `POST /auth/refresh` + `POST /auth/logout` endpoints
- [x] Rate limiting via slowapi: 5/min login & register, 30/min query, 10/min ingest (Redis-backed)
- [x] Input validation: Pydantic `Field(pattern=...)` on `patient_id` (alnum + `-_`), `query_text` (≤2000 chars), file uploads validated by route
- [x] Security headers middleware: CSP, HSTS, X-Frame-Options DENY, nosniff, XSS, Referrer-Policy, Permissions-Policy
- [x] pgcrypto encryption on `audit_log.query_text_enc` + `response_text_enc` (migration 0002); plaintext columns left nullable for backwards compat
- [x] Celery async ingestion: `POST /ingest` returns task_id immediately; `GET /ingest/{task_id}/status` polls; worker drives `asyncio.run(ingest_document(...))`
- [x] Redis query cache: sha256(patient_id|query) key, 5-min TTL, per-patient invalidation on new ingest
- [x] Health endpoints: `/health` (always 200), `/health/ready` (503 on dep failure), `/health/live`
- [x] X-Request-ID middleware + structlog JSON logs + Prometheus instrumentator (`/metrics`)
- [x] Extended parsers: DICOM metadata, CCD/C-CDA XML, plain text, DOCX, XLSX (5 new formats, total 9)
- [x] Alembic initialized; migration 0001 (full schema + trigger), migration 0002 (pgcrypto)
- [x] Production marketing site at [website/index.html](website/index.html)

## Phase 5+ backlog (still open)

- [ ] vector_store: set `hnsw.iterative_scan` + `hnsw.ef_search` on the engine — HNSW + patient_id WHERE filter can return fewer than top_k results without it
- [ ] chunker: confirm whether `voyageai.Client().tokenize()` is local or network — if network, batch sentences before counting for ingestion throughput
- [ ] chunker: replace regex sentence splitter with `pysbd` / `scispaCy` for clinical-quality boundary detection (drug doses, lab values, ICD codes)
- [ ] parser: bullet lists collapse to one sentence with no terminators — emit each bullet as its own logical unit
- [ ] parser: FHIR fallback rendering does not unroll nested CodeableConcept fields (ICD-10, SNOMED, drug codes) — add structured-field renderer
- [ ] audit_log: add ON TRUNCATE statement-level trigger — TRUNCATE bypasses row-level triggers
- [ ] auth: evaluate replacing passlib (unmaintained since 2020) with bcrypt directly or argon2-cffi
- [ ] ingest: S3 orphan cleanup — when `ingest_document` fails after the S3 upload succeeded, the S3 object is leaked. Add a sweep or transactional cleanup hook.
- [ ] audit_log: backfill plaintext rows into `query_text_enc`/`response_text_enc` then drop plaintext columns
- [ ] rate limiting: per-endpoint `@limiter.limit(...)` decorators currently disabled due to slowapi 0.1.9 + FastAPI 0.115 signature-introspection bug (breaks OpenAPI + parameter parsing). slowapi infrastructure (Limiter, RateLimitExceeded handler, Redis storage) is wired and ready — re-enable decorators after pinning to a compatible combo OR migrate to `fastapi-limiter`. Also switch key from `get_remote_address` to authenticated `actor_id` once a custom key_func is wired.
- [ ] connection pool tuning: explicit `pool_size=20, max_overflow=10, pool_pre_ping=True` on `create_async_engine`
- [ ] revoked_tokens sweeper: nightly cron `DELETE FROM revoked_tokens WHERE expires_at < now()`
- [ ] write full pytest suite — current `tests/*_e2e.py` files are scripts, not pytest modules
- [ ] add `tests/test_phase5_e2e.py` covering: refresh token flow, account lockout, rate limit 429, cache hit/miss, Celery task status, health-degraded path

## Key rules
- Synthesis agent must refuse to answer if context does not support it — no hallucination
- Audit log is append-only — never update or delete rows
- Access check always runs before retrieval — no exceptions
- Chunk size: 512 tokens, 50-token overlap
- Retrieve top 5 chunks per query by default

## Environment variables
```
GOOGLE_API_KEY=                 # Gemini synthesis LLM
VOYAGE_API_KEY=                 # voyage-3-large embeddings
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medical_rag
S3_ENDPOINT_URL=http://localhost:9000   # MinIO; omit for real AWS S3
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=medical-records
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=                     # 64+ chars — `python -c "import secrets;print(secrets.token_urlsafe(48))"`
DB_ENCRYPTION_KEY=              # pgcrypto symmetric key — same generator
```

## Commands
```bash
# Install
pip install -r requirements.txt

# Start services
docker-compose up -d            # postgres + redis + minio

# DB schema
alembic upgrade head            # production

# API server
uvicorn api.routes:app --reload --port 8000
# Open http://localhost:8000/docs

# Celery worker (separate process — required for async ingest)
celery -A agents.celery_app worker --loglevel=info

# Run tests (ad-hoc scripts, not pytest)
PYTHONPATH=. python tests/test_phase2_e2e.py
PYTHONPATH=. python tests/test_phase3_e2e.py
PYTHONPATH=. python tests/test_phase4a_e2e.py
PYTHONPATH=. python tests/test_phase4b_e2e.py
```

## Do not
- Do not use embeddings from any other provider — use Voyage `voyage-3-large` only
- Do not store raw PII in the vector store — chunk text only, metadata in Postgres
- Do not skip the audit agent — every query must be logged
- Do not let synthesis agent answer outside retrieved context
- Do not write to `audit_log.query_text` / `response_text` columns directly — write encrypted via `query_text_enc` / `response_text_enc`
- Do not bypass `alembic upgrade head` in production — `db/init.py` is local-dev only
