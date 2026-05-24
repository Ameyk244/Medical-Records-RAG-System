# Medical Records RAG System

A multi-agent retrieval-augmented generation system for clinicians to query patient records in natural language. Built for structured healthcare data — PDF, FHIR R4, HL7 v2, DICOM, CCD/C-CDA XML, CSV, DOCX, XLSX, and plain text — it returns grounded answers with inline citations, refuses to speculate when the records are silent, and writes every query to an append-only, pgcrypto-encrypted audit log. Designed as a production-grade engineering portfolio demonstrating the full lifecycle from raw document upload to secure, observable RAG in Python.

[![Python](https://img.shields.io/badge/python-3.13-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What it does

Documents are uploaded via `POST /ingest`. The route writes the raw file to S3 (MinIO locally) and immediately returns a `task_id`. A Celery worker picks up the task, parses the document with a format-specific parser, splits it into 512-token chunks with 50-token overlap, embeds each chunk with Voyage AI `voyage-3-large` (1024-dimensional vectors), and stores the vectors in Postgres with pgvector.

When a clinician sends a `POST /query` request, the orchestrator runs an access check, then embeds the query with Voyage's asymmetric `query` embedding type and finds the top-5 cosine-closest chunks for that specific patient using an HNSW index. Google Gemini `gemini-2.5-flash` synthesises a grounded answer from those chunks only. Every factual claim must be cited by context number (`[1]`, `[3]`, etc.). If the retrieved context does not support an answer, the model is instructed to respond with a `REFUSED:` prefix rather than guess.

Every query writes two rows to the `audit_log` table: one at the access-check stage and one after synthesis. The query text and Gemini response are stored as pgcrypto-encrypted `bytea` columns. A Postgres trigger enforces append-only semantics — `UPDATE` and `DELETE` raise an exception before the operation can modify any row.

Authentication is JWT-based (HS256, 60-minute expiry) with opaque SHA256-hashed refresh tokens (7-day expiry). Account lockout fires after 5 failed attempts. `POST /auth/logout` blacklists the JWT `jti` claim and mass-revokes all refresh tokens for the user.

---

## Why this is different

- **Grounded-only answers.** The synthesis agent operates under a strict system prompt that prohibits answering outside the retrieved context. If `chunks` is empty, `agents/synthesis_agent.py:synthesise()` short-circuits before making any Gemini API call and returns a `refused=True` result.
- **Forensic audit trail.** Every query produces two append-only rows in `audit_log`. A Postgres `BEFORE UPDATE OR DELETE` trigger (`raise_on_audit_mutation`) raises an exception before any mutation can land. Even a superuser cannot bypass it without first dropping the trigger.
- **Nine ingestion formats.** PDF (pdfplumber with font-size heading detection), FHIR R4 JSON (Bundle and single-resource), HL7 v2, CSV, DICOM metadata (no pixel data), CCD/C-CDA XML, plain text (ALL-CAPS heading heuristic), DOCX (Word heading styles), and XLSX (one section per sheet).
- **Patient-scoped retrieval.** `patient_id` is denormalized onto the `chunks` table, so the `WHERE chunks.patient_id = :pid` filter runs before the HNSW scan, preventing cross-patient data leakage at the query layer.
- **Sub-2-second semantic search.** The HNSW index (`vector_cosine_ops`) is created in migration `0001_initial` and hit by every `search_chunks()` call via pgvector's `<=>` operator. The async SQLAlchemy engine and aioboto3 keep I/O non-blocking throughout.
- **Deploy anywhere.** `docker-compose.yml` starts Postgres 16 + pgvector, Redis 7, and MinIO. Swap `S3_ENDPOINT_URL` for a real AWS S3 endpoint and `DATABASE_URL` for RDS and the code is unchanged.

---

## Architecture at a glance

```
HTTP request
  └─ CORSMiddleware
     └─ SecurityHeadersMiddleware  (CSP, HSTS, X-Frame-Options, nosniff, XSS, Referrer)
        └─ RequestIDMiddleware     (X-Request-ID, structlog context, request timing)
           └─ slowapi Limiter      (Redis-backed rate limits — currently wired, decorators pending)
              └─ get_current_actor (HTTPBearer → verify_token → jti blacklist check)
                 └─ POST /query
                    ├─ cache lookup  (sha256(patient_id|query), 5-min TTL)
                    │   └─ X-Cache: HIT → return immediately
                    └─ orchestrator.handle_query()
                       ├─ audit_agent.check_access()   → audit_log row 1
                       ├─ query_agent.query_chunks()
                       │   ├─ embed_texts(input_type="query")   [Voyage AI]
                       │   └─ search_chunks()                   [HNSW, cosine distance]
                       ├─ synthesis_agent.synthesise()          [Gemini gemini-2.5-flash]
                       └─ audit_agent.log_response()   → audit_log row 2 (pgcrypto encrypted)
                          └─ cache.set() → X-Cache: MISS response

Ingestion path:
  POST /ingest (multipart)
    ├─ upload_document()   [aioboto3 → MinIO/S3, key: documents/{pid}/{doc_id}/{filename}]
    ├─ invalidate_patient_cache()
    └─ ingest_task.delay() [Celery enqueue → 202 + task_id]
         ↓ (Celery worker process)
       ingest_document()
         ├─ Document row inserted, status="ingesting"
         ├─ parse_document()       [format-specific parser]
         ├─ chunk_document()       [512 tokens, 50 overlap, Voyage tokenizer]
         ├─ embed_texts(input_type="document")  [Voyage voyage-3-large]
         ├─ upsert_chunks()        [pgvector INSERT]
         └─ Document status="ready"

  GET /ingest/{task_id}/status
    └─ celery_app.AsyncResult(task_id) → {state, document_id?, error?}
```

---

## Tech stack

| Component | Tech | Purpose |
|---|---|---|
| Web framework | FastAPI 0.115.6 | Async HTTP API, OpenAPI docs |
| Data validation | Pydantic 2.10.3 | Request/response schemas, field-level validation |
| ASGI server | Uvicorn 0.32.1 | Production-grade async server |
| ORM | SQLAlchemy 2.0.36 (asyncio) | Async DB access, model definitions |
| Database | Postgres 16 (`pgvector/pgvector:pg16`) | Relational store + vector search |
| Vector extension | pgvector 0.3.6 | 1024-dim embeddings, HNSW index, cosine distance |
| Encryption extension | pgcrypto (built-in) | Symmetric encryption of audit log fields |
| Embeddings | Voyage AI `voyage-3-large` (dim 1024) | Asymmetric document/query embeddings |
| Synthesis LLM | Google Gemini `gemini-2.5-flash` | Grounded answer generation with inline citations |
| Task queue | Celery 5.4.0 | Async document ingestion |
| Message broker / cache backend | Redis 7 | Celery broker+backend, query result cache, rate limiting |
| Object storage | MinIO (aioboto3 13.2.0) | Raw document storage, S3-compatible |
| Migrations | Alembic 1.14.0 | Schema versioning |
| Auth | python-jose 3.3.0 + passlib 1.7.4 + bcrypt 4.0.1 | JWT HS256, bcrypt password hashing |
| Rate limiting | slowapi 0.1.9 | Redis-backed rate limiting (infrastructure wired) |
| Structured logging | structlog 24.4.0 | JSON request logs with per-request context |
| Observability | prometheus-fastapi-instrumentator 7.0.0 | `/metrics` endpoint |
| PDF parsing | pdfplumber 0.11.4 | Font-size + ALL-CAPS heading detection |
| HL7 v2 parsing | hl7 0.4.5 | HL7 v2 segment parsing |
| DICOM parsing | pydicom 3.0.1 | DICOM metadata extraction (no pixel data) |
| DOCX parsing | python-docx 1.1.2 | Word document, heading-style section detection |
| XLSX parsing | openpyxl 3.1.5 | Excel spreadsheet, one section per sheet |
| Token counting | tiktoken 0.8.0 | Local fallback; chunker uses Voyage SDK tokenizer |
| HTTP client | httpx 0.28.1 | Test suite HTTP client |

---

## Quick start

```bash
git clone https://github.com/Ameyk244/Medical-Records-RAG-System
cd Medical-Records-RAG-System

python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env
# Edit .env — at minimum set:
#   GOOGLE_API_KEY    (Gemini)
#   VOYAGE_API_KEY    (Voyage AI)
#   JWT_SECRET        (64+ random chars)
#   DB_ENCRYPTION_KEY (64+ random chars, independent of JWT_SECRET)

docker-compose up -d
# Waits for Postgres, Redis, and MinIO to be healthy.

PYTHONPATH=. ./.venv/bin/alembic upgrade head
# Creates all tables, indexes, and the audit-log append-only trigger.

PYTHONPATH=. ./.venv/bin/uvicorn api.routes:app --reload --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

Default users are auto-bootstrapped on first run when the `users` table is empty:

```
doctor1 / password123   (role: clinician)
admin   / password123   (role: admin)
```

For async document ingestion, start a Celery worker in a separate terminal:

```bash
PYTHONPATH=. ./.venv/bin/celery -A agents.celery_app worker --loglevel=info
```

Without the worker, `POST /ingest` returns a `task_id` but the ingestion pipeline never executes.

---

## API endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| `POST` | `/auth/login` | none | `{"username", "password"}` → `{access_token, refresh_token, token_type}` |
| `POST` | `/auth/register` | bearer (admin role) | Creates a new user; role must be `clinician`, `admin`, or `readonly` |
| `POST` | `/auth/refresh` | none | `{"refresh_token"}` → new `{access_token}` |
| `POST` | `/auth/logout` | bearer | Blacklists JWT `jti`; mass-revokes all refresh tokens for the user |
| `POST` | `/ingest` | bearer | Multipart form: `patient_id`, `source_type`, `file`, `document_date?` → 202 `{task_id}` |
| `GET` | `/ingest/{task_id}/status` | bearer | Celery task state: `PENDING`, `STARTED`, `SUCCESS`, `FAILURE` |
| `POST` | `/query` | bearer | `{"patient_id", "query_text"}` → `{answer, refused, citations[]}`, `X-Cache: HIT/MISS` |
| `GET` | `/health` | none | Always 200; body: `{status, checks: {db, redis, s3}}` |
| `GET` | `/health/ready` | none | 200 when all deps healthy, 503 otherwise (Kubernetes readiness) |
| `GET` | `/health/live` | none | Always 200 (Kubernetes liveness) |
| `GET` | `/metrics` | none | Prometheus metrics (auto-instrumented, not in OpenAPI schema) |
| `GET` | `/docs` | none | Swagger UI |

---

## Repository layout

```
Medical-Records-RAG-System/
├── agents/
│   ├── orchestrator.py          # Sequences access check → retrieval → synthesis → audit
│   ├── ingestion_agent.py       # Core ingestion pipeline: parse → chunk → embed → upsert
│   ├── ingestion_worker.py      # Celery task wrapping ingest_document with asyncio.run()
│   ├── celery_app.py            # Celery app instance, broker=Redis
│   ├── query_agent.py           # Embeds query (input_type="query"), calls search_chunks()
│   ├── synthesis_agent.py       # Gemini API call, citation extraction, refusal logic
│   └── audit_agent.py           # check_access() + log_response() (pgcrypto encrypted)
├── core/
│   ├── embeddings.py            # voyage-3-large async wrapper, batch size 128
│   ├── vector_store.py          # upsert_chunks() + search_chunks() (HNSW cosine)
│   ├── document_parser.py       # 9 format parsers returning list[Section]
│   ├── chunker.py               # 512-token / 50-overlap sentence-aware chunker
│   ├── s3.py                    # aioboto3 upload_document() + ensure_bucket_exists()
│   ├── cache.py                 # Redis query cache, sha256 key, 5-min TTL
│   ├── middleware.py            # SecurityHeadersMiddleware, RequestIDMiddleware, structlog
│   └── auth.py                  # JWT create/verify, bcrypt hash/verify, authenticate_user()
├── api/
│   └── routes.py                # FastAPI app, lifespan, all 12 endpoints
├── db/
│   ├── models.py                # SQLAlchemy models: Document, Chunk, AuditLog, User,
│   │                            #   RefreshToken, RevokedToken
│   └── init.py                  # Local-dev schema init (do not use in production)
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 0001_initial.py      # Full schema + HNSW index + append-only trigger
│       └── 0002_encrypted_audit.py  # pgcrypto extension + query_text_enc/response_text_enc
├── tests/
│   └── test_phase*_e2e.py       # Ad-hoc end-to-end scripts (not pytest modules yet)
├── website/
│   ├── index.html               # Marketing site: hero, architecture diagram, animated demo
│   ├── auth.html                # Login, register, refresh, and logout panels
│   ├── try.html                 # Live ingest + query UI against the running API
│   ├── app.js                   # Shared JS for API calls
│   └── app.css                  # Shared styles
├── .env.example                 # All required env vars with comments
├── docker-compose.yml           # Postgres 16 + pgvector, Redis 7, MinIO
├── requirements.txt
├── alembic.ini
└── Medical-Records-RAG-System.md  # Original spec and build-phase log
```

---

## Configuration

All configuration is read from environment variables. Copy `.env.example` to `.env` and fill in the required values before starting the server.

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key. The SDK also accepts `GEMINI_API_KEY`. Used by `agents/synthesis_agent.py` via `genai.Client()`. |
| `VOYAGE_API_KEY` | Yes | Voyage AI API key. Used by `core/embeddings.py` via `voyageai.AsyncClient()` and by `core/chunker.py` for token counting. |
| `DATABASE_URL` | Yes | Postgres connection string in `postgresql://user:pass@host:port/db` form. The application rewrites it to `postgresql+psycopg://` (psycopg3 async dialect) at startup. |
| `JWT_SECRET` | Yes | Secret for HS256 JWT signing and verification. Use at least 64 random characters. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. |
| `DB_ENCRYPTION_KEY` | Yes | pgcrypto symmetric key passed to `pgp_sym_encrypt()` / `pgp_sym_decrypt()` for `audit_log.query_text_enc` and `response_text_enc`. Generate independently from `JWT_SECRET`. |
| `S3_ACCESS_KEY` | Yes | S3 / MinIO access key. Default for local MinIO: `minioadmin`. |
| `S3_SECRET_KEY` | Yes | S3 / MinIO secret key. Default for local MinIO: `minioadmin`. |
| `S3_BUCKET` | Yes | Bucket name for raw document storage. Created automatically on startup if it does not exist. |
| `S3_ENDPOINT_URL` | No | Override the S3 endpoint. Set to `http://localhost:9000` for MinIO. Omit for real AWS S3. |
| `REDIS_URL` | Yes | Redis connection string, e.g. `redis://localhost:6379/0`. Used for Celery broker/backend, query result cache, and rate limiter storage. |

---

## Marketing website

The repository ships a multi-page static site at `website/`:

- `website/index.html` — home page with architecture overview and animated demo
- `website/auth.html` — login, register, token refresh, and logout panels that call the real API
- `website/try.html` — live ingest (drag-and-drop file upload) and query UI

Open `website/index.html` directly in any browser. The Try and Auth pages expect the API at `http://localhost:8000`. No build step required.

---

## Development

**Run the test scripts:**

```bash
# Ad-hoc end-to-end scripts (require a running stack)
PYTHONPATH=. python tests/test_phase2_e2e.py
PYTHONPATH=. python tests/test_phase3_e2e.py
PYTHONPATH=. python tests/test_phase4a_e2e.py
PYTHONPATH=. python tests/test_phase4b_e2e.py
```

Note: the existing test files are scripts, not pytest modules. A full pytest suite is a known open item (see Known Limitations).

**Database migrations:**

```bash
# Apply all migrations (production path)
PYTHONPATH=. alembic upgrade head

# Generate a new migration after changing db/models.py
PYTHONPATH=. alembic revision --autogenerate -m "short description"

# Roll back one step
PYTHONPATH=. alembic downgrade -1
```

**CLI ingestion (bypasses HTTP + Celery):**

```bash
PYTHONPATH=. python -m agents.ingestion_agent \
  --file path/to/record.pdf \
  --patient_id P123 \
  --source_type pdf \
  --document_date 2024-03-01
```

---

## Known limitations

The following items are open from the Phase 5+ backlog:

- **HNSW + WHERE filter**: `hnsw.ef_search` and `hnsw.iterative_scan` are not configured. When a `WHERE patient_id = :pid` filter is combined with the HNSW scan, fewer than `top_k` results may be returned for patients with small document sets.
- **Rate limiting decorators disabled**: The slowapi `@limiter.limit(...)` decorators are wired but not applied to endpoints due to a signature-introspection incompatibility between slowapi 0.1.9 and FastAPI 0.115. The limiter infrastructure is ready; re-enable after resolving the dependency conflict.
- **FHIR nested fields**: `core/document_parser.py:_render_fhir_resource()` only emits top-level scalar fields. Nested `CodeableConcept` structures (ICD-10, SNOMED, drug codes) are not unrolled when a `text.div` narrative is absent.
- **S3 orphan cleanup**: If `ingest_document()` fails after the S3 upload succeeds, the S3 object is leaked. No sweep or transactional cleanup is implemented.
- **Audit log TRUNCATE**: The append-only trigger is a `FOR EACH ROW` trigger. `TRUNCATE` bypasses row-level triggers and is not covered.
- **passlib maintenance**: passlib 1.7.4 has been unmaintained since 2020. Evaluating a migration to `bcrypt` directly or `argon2-cffi` is an open backlog item.
- **Test coverage**: The current `tests/*_e2e.py` files are scripts, not pytest modules. A full pytest suite with fixtures covering auth flows, lockout, cache hit/miss, and Celery task status is not yet written.
- **Revoked token sweeper**: Expired rows in `revoked_tokens` are never purged. A nightly `DELETE FROM revoked_tokens WHERE expires_at < now()` cron job is not yet set up.

---

## License

MIT.

---

## Acknowledgements

Built end-to-end with Claude Code (Opus `claude-opus-4-6` for architecture decisions, Sonnet `claude-sonnet-4-6` for all implementation). See `Medical-Records-RAG-System.md` for the original specification and build-phase log.
