# How it works — internal deep dive

This document walks through every operation in the system end-to-end, naming the exact files, functions, libraries, and database tables involved at each step.

For the high-level architecture see `Medical-Records-RAG-System.md` and `README.md`. This file is for engineers who want to understand or modify the code.

---

## Table of contents

1. [Boot and lifespan](#1-boot-and-lifespan)
2. [Authentication](#2-authentication)
   - [Login (POST /auth/login)](#login-post-authlogin)
   - [Register (POST /auth/register)](#register-post-authregister)
   - [Refresh (POST /auth/refresh)](#refresh-post-authrefresh)
   - [Logout (POST /auth/logout)](#logout-post-authlogout)
   - [Token validation on protected routes](#token-validation-on-protected-routes)
   - [Account lockout](#account-lockout)
3. [Document ingestion](#3-document-ingestion)
   - [HTTP layer](#http-layer)
   - [S3 upload](#s3-upload)
   - [Celery enqueue](#celery-enqueue)
   - [Worker pipeline](#worker-pipeline)
   - [Status polling](#status-polling)
4. [Query](#4-query)
   - [Cache lookup](#cache-lookup)
   - [Orchestrator entry point](#orchestrator-entry-point)
   - [Access check and first audit row](#access-check-and-first-audit-row)
   - [Query embedding](#query-embedding)
   - [pgvector retrieval](#pgvector-retrieval)
   - [Synthesis](#synthesis)
   - [Citation extraction](#citation-extraction)
   - [Second audit row (encrypted)](#second-audit-row-encrypted)
   - [Cache write](#cache-write)
   - [Response shape](#response-shape)
5. [Audit logging](#5-audit-logging)
   - [Append-only trigger](#append-only-trigger)
   - [pgcrypto encryption](#pgcrypto-encryption)
6. [Middleware stack](#6-middleware-stack)
7. [Database schema](#7-database-schema)
   - [documents](#documents)
   - [chunks](#chunks)
   - [users](#users)
   - [refresh_tokens](#refresh_tokens)
   - [audit_log](#audit_log)
   - [revoked_tokens](#revoked_tokens)
   - [Indexes](#indexes)
   - [HNSW index](#hnsw-index)
8. [Document parsers (9 formats)](#8-document-parsers-9-formats)
9. [Configuration and secrets](#9-configuration-and-secrets)
10. [Failure modes and exception types](#10-failure-modes-and-exception-types)

---

## 1. Boot and lifespan

When uvicorn imports `api/routes.py`, Python executes the module-level code: the `FastAPI` app is created, middleware is added, and the `lifespan` async context manager is registered as the startup/shutdown hook.

`api/routes.py:lifespan()` runs the following steps in order:

1. **Logging initialization** — calls `core/middleware.py:configure_logging()`. This configures `structlog` with a processor chain: `merge_contextvars → add_log_level → TimeStamper(utc=True) → dict_tracebacks → JSONRenderer`. Every log line from this point is a JSON object. `configure_logging()` must run first so that subsequent startup steps (bucket creation, user seeding) produce structured output.

2. **Database URL normalization** — reads `DATABASE_URL` from the environment. If it starts with `postgresql://` (the standard form in `.env`) and has no `+driver` suffix, it is rewritten to `postgresql+psycopg://` — the dialect token that SQLAlchemy 2.x requires for the psycopg3 async driver.

3. **Engine and session factory creation** — calls `create_async_engine(db_url, future=True)` and `async_sessionmaker(engine, expire_on_commit=False)`. Both are stored on `app.state` so route handlers can reach them via `request.app.state.session_factory`.

4. **S3 bucket initialization** — calls `core/s3.py:ensure_bucket_exists()`. It issues a `head_bucket` request via aioboto3. A 404 / `NoSuchBucket` response triggers `create_bucket`. A `BucketAlreadyOwnedByYou` or `BucketAlreadyExists` error on the create call is treated as success (safe under concurrent startup). Any other error raises `S3Error`.

5. **Default user bootstrapping** — runs `SELECT count(*) FROM users`. If the count is 0, it calls `core/auth.py:register_user()` twice to create `doctor1` (role `clinician`) and `admin` (role `admin`), both with password `password123`. This is a one-time seed on a fresh database. An `AuthError` from a concurrent startup race is silently swallowed.

6. **Yield** — the `lifespan` yields, handing control to the request-serving loop.

7. **Shutdown** — when uvicorn receives a shutdown signal, `lifespan` resumes from the `finally` block and calls `engine.dispose()` to close all pooled connections.

Reference files:
- `api/routes.py:lifespan()` (lines 58-98)
- `core/middleware.py:configure_logging()` (lines 64-76)
- `core/s3.py:ensure_bucket_exists()` (lines 79-99)
- `core/auth.py:register_user()` (lines 148-176)

---

## 2. Authentication

### Login (POST /auth/login)

The route handler is `api/routes.py:login()`. It accepts a JSON body validated by the `LoginRequest` Pydantic model (`username`: 1-64 chars, `password`: min 1 char).

**Step 1 — Rate limit.** The `@limiter.limit("5/minute")` decorator is wired in the spec but currently disabled due to a slowapi 0.1.9 + FastAPI 0.115 signature-introspection incompatibility (documented in `Medical-Records-RAG-System.md` backlog). The limiter infrastructure is initialized and attached to `app.state.limiter`, but individual endpoint decorators are not active.

**Step 2 — Authenticate.** `core/auth.py:authenticate_user(username, password, session)` is called:

1. Executes `SELECT * FROM users WHERE username = :username` via `session.execute(select(User).where(User.username == username))`.
2. If no row is found, or `user.is_active` is `False`, raises `AuthError("invalid credentials")`. The same message for both cases prevents user enumeration.
3. Checks `user.locked_until > now`. If the lockout window is active, raises `AuthError("account locked")` without checking the password.
4. Calls `core/auth.py:verify_password(plain, user.password_hash)`, which delegates to `_pwd_context.verify()` (passlib's bcrypt context, initialized once at import time as `_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")`).
5. On password failure: increments `user.failed_login_attempts`. If `failed_login_attempts >= 5`, sets `user.locked_until = now + 15 minutes`. Commits and raises `AuthError("invalid credentials")`.
6. On password success: resets `user.failed_login_attempts = 0`, clears `user.locked_until`, sets `user.last_login = now`, commits, and returns the `User` object.

**Step 3 — Mint access token.** `core/auth.py:create_access_token(actor_id=user.username, actor_role=user.role)`:

- Builds a claims dict: `{sub, role, iat, exp, jti}`. `exp = now + 60 minutes`. `jti = uuid.uuid4().hex` (a 32-char hex string unique per token, used as the revocation key).
- Signs with `jwt.encode(claims, JWT_SECRET, algorithm="HS256")` using python-jose.

**Step 4 — Mint refresh token.** `raw_refresh = uuid.uuid4().hex` is generated in the route handler. The server stores `sha256(raw_refresh)` as `refresh_tokens.token_hash` — the raw value is never persisted. The `RefreshToken` row has `expires_at = now + 7 days`.

**Step 5 — Response.** Returns `{"access_token": "<jwt>", "refresh_token": "<opaque_hex>", "token_type": "bearer"}`.

The opaque design for refresh tokens means a database breach does not expose replayable tokens — only SHA256 digests are stored.

### Register (POST /auth/register)

Protected by `require_admin` dependency, which delegates to `get_current_actor` then checks `actor_role == "admin"`. Returns 403 if the role is not `admin`.

The request body is `RegisterRequest`: `username` (alphanumeric plus `_-`, 1-64 chars), `password` (min 8 chars), `role` (pattern `^(clinician|admin|readonly)$`).

Calls `core/auth.py:register_user(username, password, role, session)`:
1. Validates `role` is in `_VALID_ROLES = frozenset({"clinician", "admin", "readonly"})`.
2. Calls `hash_password(plain)` → `_pwd_context.hash(plain)` (bcrypt).
3. Inserts a `User` row. On `IntegrityError` (unique constraint on `username`), rolls back and raises `AuthError("username already taken")`.

Returns 201 `{"user_id", "username", "role"}` on success, 409 if the username is taken.

### Refresh (POST /auth/refresh)

No authentication required. Accepts `{"refresh_token": "<opaque>"}`.

1. Computes `token_hash = sha256(body.refresh_token)`.
2. Queries `SELECT * FROM refresh_tokens WHERE token_hash = :hash`.
3. If no row, or `row.revoked` is `True`, or `row.expires_at < now` → returns 401 with `"invalid refresh token"`. Same message for all three cases.
4. Fetches the associated `User`; returns 401 if the user is inactive.
5. Issues a new access token via `create_access_token()`. The refresh token row is not rotated — the same opaque token can be reused until it expires or the user logs out.

### Logout (POST /auth/logout)

Protected by bearer auth. Two side effects:

1. **Blacklist the JWT.** Creates a `RevokedToken(jti=actor["jti"], expires_at=<token exp>)` row. On `IntegrityError` (already blacklisted), rolls back and continues — logout is idempotent.
2. **Revoke all refresh tokens.** Executes `UPDATE refresh_tokens SET revoked = true WHERE user_id = (SELECT id FROM users WHERE username = :actor_id)`.

After logout, any subsequent use of the access token or any refresh token for that user is rejected.

### Token validation on protected routes

Every protected endpoint depends on `api/routes.py:get_current_actor()`:

1. The `HTTPBearer(auto_error=False)` scheme extracts the `Authorization: Bearer <token>` header. If absent or empty, raises `HTTPException(401)`.
2. Calls `core/auth.py:verify_token(token)`:
   - `jwt.decode(token, JWT_SECRET, algorithms=["HS256"])`. `ExpiredSignatureError` is caught before the parent `JWTError` class — order matters because `ExpiredSignatureError` is a subclass.
   - Verifies `sub` and `jti` claims are present.
3. Queries `SELECT 1 FROM revoked_tokens WHERE jti = :jti`. If a row exists, raises 401 `"token has been revoked"`.
4. Calls `structlog.contextvars.bind_contextvars(actor_id=claims["sub"])` so all subsequent log lines in this request carry the actor identity.
5. Returns `{"actor_id", "actor_role", "jti", "exp"}` to the route function.

`RequestIDMiddleware` calls `structlog.contextvars.unbind_contextvars("request_id", "actor_id")` after the response is sent to prevent context leaking across requests in the same worker.

### Account lockout

Constants in `core/auth.py`:

```
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_MINUTES = 15
```

The counter (`users.failed_login_attempts`) is incremented only when a user row exists and the password check fails. Attempts with an unknown username do not touch the counter. On the 5th failure, `locked_until = now + 15 minutes` is written. During the lockout window, all login attempts return `"account locked"` without executing the bcrypt verification. The counter resets to 0 only on a successful login.

---

## 3. Document ingestion

### HTTP layer

`api/routes.py:ingest()` accepts a multipart form with fields `patient_id`, `source_type`, `file` (UploadFile), and optional `document_date` (YYYY-MM-DD string).

Validation:
- File content is read with `await file.read()`. An empty file raises 400.
- `document_date` is parsed with `date.fromisoformat()`. An invalid format raises 400.
- A `document_id = uuid.uuid4()` is generated in the route. This UUID is used for both the S3 key and the `documents` table row, ensuring they stay correlated even though the pipeline runs asynchronously.

### S3 upload

`core/s3.py:upload_document(patient_id, document_id, filename, content)` is called synchronously (it is async) before Celery enqueue:

- Constructs the key: `documents/{patient_id}/{document_id}/{filename}`.
- Opens an aioboto3 session (lazily initialized singleton `_session`).
- Issues `s3.put_object(Bucket=bucket, Key=key, Body=content)`.
- On `botocore.exceptions.ClientError`, raises `S3Error`.

The upload happens before the Celery task is enqueued. This means the raw document is in S3 even if the Celery worker is down. S3 orphan cleanup (when the worker subsequently fails) is a known open item.

### Celery enqueue

`ingest_task.delay(source_type, content, patient_id, ...)` serializes all arguments and pushes a message to the Redis broker. The `content` bytes field is included directly in the message (Celery serializes it as base64 via the default JSON serializer).

The route immediately calls `core/cache.py:invalidate_patient_cache(patient_id)` to purge any cached query results for this patient. This is a best-effort fire-and-forget — `CacheError` is caught and swallowed.

The route returns 202 `{"task_id": "<celery-task-uuid>"}`.

### Worker pipeline

`agents/ingestion_worker.py:ingest_task` is the Celery task (bound with `bind=True`, registered as `agents.ingestion_worker.ingest_task`). It is a synchronous Celery task that drives the async pipeline with `asyncio.run(_run_ingest(...))`.

`_run_ingest()` creates a fresh `create_async_engine` and `async_sessionmaker` per task invocation (the worker is a separate process with no shared connection pool) and calls `agents/ingestion_agent.py:ingest_document()`.

`ingest_document()` runs in three phases designed for observability:

**Phase A — Document row insertion.**

A `Document` row is inserted with `status="ingesting"` and `chunk_count=0`. The `document_id` passed from the route is re-used so the S3 key and DB row are correlated. This row is committed immediately — operational dashboards can see the in-progress document even if Phase B fails.

**Phase B — Full pipeline (atomic commit).**

1. `core/document_parser.py:parse_document(source_type, content)` dispatches to the format-specific parser and returns `list[Section]`. Each `Section` has a `heading` and `text` field.

2. `core/chunker.py:chunk_document(sections)` converts sections to `list[Chunk]`. Each `Chunk` has `text`, `token_count`, `section`, and `chunk_index`. The chunker uses the Voyage SDK's synchronous `voyageai.Client().count_tokens([text], model="voyage-3-large")` for exact token counts. The 512-token limit and 50-token overlap are constants `CHUNK_TOKEN_LIMIT` and `CHUNK_OVERLAP_TOKENS`.

3. `core/embeddings.py:embed_texts([tc.text for tc in text_chunks], input_type="document")` calls the Voyage AI async client in batches of 128. The `input_type="document"` produces document-side vectors in Voyage's asymmetric embedding space.

4. `core/vector_store.py:upsert_chunks(chunk_rows, session)` inserts all chunk rows (including the 1024-float embedding vector) with a single `insert(Chunk).values(rows)`. The insert is not an upsert — there is no `ON CONFLICT DO UPDATE`. Re-ingesting a document assigns a new `document_id`; old chunks are deleted by the `CASCADE` on the `document_id` foreign key.

5. Updates `documents.chunk_count` and `documents.status = "ready"` in the same transaction as the chunk insert.

**Phase C — Error handling.**

If Phase B raises any exception (other than `ValueError`, which propagates untouched), the session is rolled back, and a fresh `UPDATE documents SET status = "failed" WHERE id = :doc_id` is committed so operators can identify failed documents.

`ingestion_agent.py` validates only the four original source types (`pdf`, `fhir_json`, `hl7`, `csv`) at the top of `ingest_document()`. The five extended types (`dicom_meta`, `ccd_xml`, `txt`, `docx`, `xlsx`) accepted by the HTTP route reach the parser without hitting this guard — the validator in `ingest_document()` predates the extended parser set and is a known inconsistency.

### Status polling

`api/routes.py:ingest_status(task_id)` calls `celery_app.AsyncResult(task_id).state`. Celery stores task results in the Redis backend (configured as `backend=REDIS_URL` in `agents/celery_app.py`). Results expire after 3600 seconds (`result_expires=3600`).

States returned: `PENDING` (queued or unknown), `STARTED` (worker has started), `SUCCESS` (includes `document_id` in the response), `FAILURE` (includes the exception string as `error`).

---

## 4. Query

### Cache lookup

`api/routes.py:query()` calls `core/cache.py:get_cached_query_result(patient_id, query_text)` before entering the pipeline.

The cache key is `query:{patient_id}:{sha256(patient_id + "|" + query_text)}`. The `patient_id` is included in the digest, not just the prefix, so two patients with the same query text get different cache keys and cannot see each other's results.

On a Redis read failure, `CacheError` is caught and `cached = None` is used. Cache is best-effort: a Redis outage degrades to uncached queries, not failures.

A cache hit returns a `Response` with `X-Cache: HIT` header immediately, bypassing Voyage, Gemini, and the audit pipeline entirely.

### Orchestrator entry point

On a cache miss, `agents/orchestrator.py:handle_query(query_text, patient_id, actor_id, actor_role, session)` is called. The orchestrator validates its inputs, then sequences four operations: access check, retrieval, synthesis, and audit log.

If both the pipeline and the audit write fail, the orchestrator raises `OrchestratorError` with the audit failure as the primary cause — an audit gap is treated as a harder failure than a synthesis error.

### Access check and first audit row

`agents/audit_agent.py:check_access(actor_id, actor_role, patient_id, session)`:

- Current implementation is permissive: any non-empty `actor_id` is allowed (Phase 4 stub). The comment in the code notes that real RBAC with role and patient-consent checks is planned for a future phase.
- Writes a row to `audit_log` with `action="query"` and `decision="allowed"` (or `action="access_denied"` and `decision="denied"`).
- Commits the audit row immediately in its own transaction. The commit happens before retrieval so the row is durable even if the calling orchestrator transaction rolls back.

If `check_access()` returns `False` (access denied), the orchestrator returns a `SynthesisResult(answer="Access denied.", refused=True)` immediately without touching Voyage or Gemini.

### Query embedding

`agents/query_agent.py:query_chunks()` calls `core/embeddings.py:embed_texts([query_text.strip()], input_type="query")`.

The `input_type="query"` is critical: Voyage `voyage-3-large` produces asymmetric embeddings. Document vectors and query vectors live in different subspaces. Using `input_type="document"` for a query would silently produce lower-quality retrieval without raising an error.

`embed_texts()` wraps the call in a `voyageai.AsyncClient()` (lazy singleton `_client`). It batches at 128 texts per call, passing `truncation=True` to handle any input that exceeds the model's context limit.

### pgvector retrieval

`core/vector_store.py:search_chunks(query_embedding, patient_id, top_k, session)` builds:

```sql
SELECT id, document_id, text, section, document_date,
       embedding <=> :query_vec AS distance
FROM chunks
WHERE patient_id = :patient_id
[AND document_date >= :from]
[AND document_date <= :to]
ORDER BY distance
LIMIT :top_k
```

The `<=>` operator is pgvector's cosine distance. The HNSW index (`ix_chunks_embedding_hnsw`, `vector_cosine_ops`) is used for the `ORDER BY distance` scan. The `WHERE patient_id` filter runs before the HNSW scan in the query planner, scoping the search to a single patient.

The raw `distance` (0 = identical, 2 = maximally dissimilar) is converted to a similarity score with `score = 1 - distance`. Returned as `list[ChunkResult]`.

Default `top_k = 5` as specified in project rules.

### Synthesis

`agents/synthesis_agent.py:synthesise(query_text, chunks)`:

1. If `chunks` is empty, returns `SynthesisResult(refused=True)` without calling Gemini. This is the primary hallucination guard — no API call is made when there is nothing to ground the answer.

2. Builds a numbered context string. Each chunk becomes `[N] | Section: <name> | Date: <date>\n<text>`. Sections and dates are omitted when `None`.

3. Constructs `user_message = "Context:\n{context}\n\nQuestion: {query_text}"`.

4. Calls `genai.Client().aio.models.generate_content(model="gemini-2.5-flash", contents=user_message, config=GenerateContentConfig(system_instruction=_SYSTEM_PROMPT, max_output_tokens=1024))`.

The system prompt (`_SYSTEM_PROMPT` constant in `synthesis_agent.py`) enforces:
- Every factual claim cited with `[N]` inline.
- If context is insufficient: respond with exactly `REFUSED: <reason>` and nothing else.
- No training-data knowledge.
- No speculation or inference beyond literal context.
- No preambles.

### Citation extraction

After the Gemini response arrives, the answer string is checked for a `REFUSED:` prefix (case-insensitive). If found, `refused=True` and no citations are extracted.

Otherwise, `re.finditer(r"\[(\d+)\]", answer)` extracts all `[N]` references. Out-of-range indices (N < 1 or N > len(chunks)) are skipped. Duplicates are deduplicated with a `seen` set. For each valid unique N, a `CitationRef(chunk_id, document_id, section, document_date, score)` is built from `chunks[N-1]`.

### Second audit row (encrypted)

`agents/audit_agent.py:log_response(actor_id, actor_role, patient_id, query_text, chunks, result, session)` inserts the response audit row:

```python
insert(AuditLog).values(
    ...
    query_text=None,               # plaintext column left NULL
    response_text=None,            # plaintext column left NULL
    query_text_enc=func.pgp_sym_encrypt(query_text, key),
    response_text_enc=func.pgp_sym_encrypt(result.answer, key),
    chunks_retrieved=[{"chunk_id": str(c.chunk_id), "score": c.score} for c in chunks],
    decision="answered" or "refused_no_context",
)
```

`func.pgp_sym_encrypt()` is a SQLAlchemy `func` expression that passes the encryption to Postgres. The key is read from `DB_ENCRYPTION_KEY` at call time via `_encryption_key()`. If `DB_ENCRYPTION_KEY` is not set, `AuditError` is raised immediately before any DB write.

The audit row is committed in its own transaction (same pattern as `check_access`).

### Cache write

`core/cache.py:set_cached_query_result(patient_id, query_text, serialized, ttl=300)` calls `redis.set(key, json.dumps(result), ex=300)`. TTL is 5 minutes. `CacheError` from Redis is caught and swallowed — cache failures are non-fatal.

### Response shape

```json
{
  "answer": "The patient was prescribed metformin 500mg twice daily [1][2].",
  "refused": false,
  "citations": [
    {
      "chunk_id": "3fa85f64-...",
      "document_id": "1c9de3b2-...",
      "section": "Medications",
      "document_date": "2024-01-15",
      "score": 0.91
    }
  ]
}
```

The `X-Cache: MISS` header is set on all non-cached responses. `X-Cache: HIT` is set when served from Redis.

---

## 5. Audit logging

### Append-only trigger

Migration `alembic/versions/0001_initial.py` creates a Postgres trigger function and trigger:

```sql
CREATE OR REPLACE FUNCTION raise_on_audit_mutation()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_no_mutation
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION raise_on_audit_mutation();
```

`BEFORE ... FOR EACH ROW` means the exception is raised before the storage engine sees the mutation. The operation is fully prevented — no partial write occurs. The trigger cannot be bypassed by any role short of a superuser who first drops the trigger, which would be visible in the DDL audit trail.

Known gap: `TRUNCATE` bypasses `FOR EACH ROW` triggers. A statement-level `ON TRUNCATE` trigger is listed in the open backlog.

### pgcrypto encryption

Migration `alembic/versions/0002_encrypted_audit.py` adds:

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
ALTER TABLE audit_log ADD COLUMN query_text_enc bytea;
ALTER TABLE audit_log ADD COLUMN response_text_enc bytea;
```

The existing plaintext `query_text` and `response_text` columns remain in the schema as nullable. New rows written by `audit_agent.log_response()` leave those columns `NULL` and write the encrypted values to `query_text_enc` and `response_text_enc` via `pgp_sym_encrypt(value, key)`. Old rows (pre-migration) retain their plaintext values.

To decrypt for compliance review:

```sql
SELECT
    id,
    actor_id,
    created_at,
    pgp_sym_decrypt(query_text_enc, 'your-key') AS query_text,
    pgp_sym_decrypt(response_text_enc, 'your-key') AS response_text
FROM audit_log
WHERE query_text_enc IS NOT NULL;
```

Backfilling old plaintext rows into the encrypted columns and dropping the plaintext columns is an open backlog item.

---

## 6. Middleware stack

Middleware is added to the FastAPI app in `api/routes.py` (lines 109-132). Starlette processes middleware in reverse-addition order (last added = outermost). The effective order for an incoming request is:

| Order | Middleware | File | What it does |
|---|---|---|---|
| 1 (outermost) | `CORSMiddleware` | fastapi built-in | Sets `Access-Control-Allow-*` headers. Dev config allows all origins. Exposes `X-Cache` and `X-Request-ID` to browser JS. |
| 2 | `SecurityHeadersMiddleware` | `core/middleware.py` | Adds `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Strict-Transport-Security: max-age=31536000`, `Content-Security-Policy: default-src 'self'` (bypassed for `/docs` paths), `Referrer-Policy: no-referrer`, `Permissions-Policy: geolocation=(), camera=(), microphone=()`. |
| 3 | `RequestIDMiddleware` | `core/middleware.py` | Reads `X-Request-ID` from the incoming request or generates `uuid4().hex`. Binds it to structlog context. Records start time with `time.perf_counter()`. After the response: logs `request_complete` with `path`, `method`, `status_code`, `duration_ms`; clears structlog context vars; sets `X-Request-ID` on the response. |
| 4 (innermost) | slowapi `Limiter` | slowapi | Registered via `app.state.limiter`. Uses `get_remote_address` as the key function. Redis-backed when `REDIS_URL` is set, in-memory otherwise. Individual `@limiter.limit()` decorators are wired but disabled (see known limitations). |

Prometheus instrumentation is registered via `setup_metrics(app)` in `core/middleware.py`, which wraps the app with `prometheus-fastapi-instrumentator`. Metrics exclude `/metrics`, `/health`, `/health/ready`, `/health/live` from instrumentation. The `/metrics` endpoint is added to the app but excluded from the OpenAPI schema.

---

## 7. Database schema

All models are in `db/models.py`. The schema is created by `alembic upgrade head` (migrations `0001_initial` and `0002_encrypted_audit`).

### documents

Tracks every ingested document. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` PK | Default `uuid4`. Passed from the HTTP route to correlate with S3 key. |
| `patient_id` | `VARCHAR(64)` | Indexed. Foreign-key-like coupling to all `chunks` for this document. |
| `source_type` | `VARCHAR(16)` | One of the 9 parser keys. |
| `source_uri` | `TEXT` | S3 key (`documents/{pid}/{doc_id}/{filename}`) or `file://` path for CLI ingestion. |
| `original_filename` | `TEXT` | Client-supplied filename from the upload. |
| `document_date` | `DATE` | Clinical event date. Optional; used for date-range filtering in queries. |
| `ingested_at` | `TIMESTAMPTZ` | `server_default=now()`. |
| `ingested_by` | `VARCHAR(128)` | Actor username who triggered ingestion. |
| `metadata` (ORM: `doc_metadata`) | `JSONB` | Reserved for future use. |
| `chunk_count` | `INTEGER` | Updated to the actual count at end of Phase B. |
| `embedding_model` | `VARCHAR(64)` | Stores `"voyage-3-large"` from `core/embeddings.py:EMBEDDING_MODEL`. |
| `status` | `VARCHAR(16)` | `pending` → `ingesting` → `ready` or `failed`. |

Relationship: `chunks` (one-to-many, `cascade="all, delete-orphan"`).

### chunks

Stores each text chunk and its embedding. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` PK | Default `uuid4`. Used as `chunk_id` in `CitationRef` and audit log. |
| `document_id` | `UUID` FK | `ON DELETE CASCADE` from `documents.id`. |
| `patient_id` | `VARCHAR(64)` | Denormalized from `documents` to enable hot-path `WHERE patient_id = :pid` without a join. |
| `chunk_index` | `INTEGER` | Sequential position within the document. |
| `text` | `TEXT` | Raw chunk text sent to Gemini as context. |
| `token_count` | `INTEGER` | Exact count from `voyageai.Client().count_tokens()`. |
| `section` | `VARCHAR(128)` | Section heading from the parser (e.g., `"Medications"`, `"Observations"`). |
| `document_date` | `DATE` | Denormalized from `documents` for date-range filtering without a join. |
| `embedding` | `VECTOR(1024)` | voyage-3-large document embedding. |
| `created_at` | `TIMESTAMPTZ` | `server_default=now()`. |

### users

Stores clinician and admin accounts. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` PK | |
| `username` | `VARCHAR(64)` | Unique index. |
| `password_hash` | `VARCHAR(128)` | bcrypt hash. |
| `role` | `VARCHAR(32)` | `clinician`, `admin`, or `readonly`. |
| `is_active` | `BOOLEAN` | Soft-delete flag. Inactive users cannot log in. |
| `failed_login_attempts` | `INTEGER` | Reset to 0 on success. |
| `locked_until` | `TIMESTAMPTZ` | Null when not locked. |
| `last_login` | `TIMESTAMPTZ` | Updated on every successful login. |
| `created_at` | `TIMESTAMPTZ` | `server_default=now()`. |

### refresh_tokens

Stores SHA256 digests of opaque refresh tokens. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` PK | |
| `user_id` | `UUID` FK | `ON DELETE CASCADE` from `users.id`. |
| `token_hash` | `VARCHAR(128)` | `sha256(raw_token).hexdigest()`. Indexed. |
| `expires_at` | `TIMESTAMPTZ` | 7 days from creation. |
| `revoked` | `BOOLEAN` | Set to `True` on logout for all tokens belonging to the user. |
| `created_at` | `TIMESTAMPTZ` | |

### audit_log

Append-only. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGINT` | Auto-increment PK. |
| `actor_id` | `VARCHAR(128)` | Username of the requesting clinician. |
| `actor_role` | `VARCHAR(64)` | Role at the time of the request. |
| `action` | `VARCHAR(32)` | `"query"` or `"access_denied"`. |
| `patient_id` | `VARCHAR(64)` | Patient whose records were queried. |
| `query_text` | `TEXT` | Plaintext query. `NULL` for new rows (use `query_text_enc`). |
| `chunks_retrieved` | `JSONB` | `[{"chunk_id": str, "score": float}]` for the response row. |
| `response_text` | `TEXT` | Plaintext response. `NULL` for new rows (use `response_text_enc`). |
| `query_text_enc` | `BYTEA` | pgcrypto-encrypted query text. Written by `audit_agent.log_response()`. |
| `response_text_enc` | `BYTEA` | pgcrypto-encrypted response. Written by `audit_agent.log_response()`. |
| `decision` | `VARCHAR(32)` | `"allowed"`, `"denied"`, `"answered"`, or `"refused_no_context"`. |
| `created_at` | `TIMESTAMPTZ` | `server_default=now()`. |

### revoked_tokens

Access-token blacklist. Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGINT` | Auto-increment PK. |
| `jti` | `VARCHAR(64)` | JWT ID claim. Unique index. |
| `expires_at` | `TIMESTAMPTZ` | Token's natural expiry. Used for future sweep job. |
| `revoked_at` | `TIMESTAMPTZ` | `server_default=now()`. |

### Indexes

| Index name | Table | Columns | Type | Purpose |
|---|---|---|---|---|
| `ix_documents_patient_id` | `documents` | `patient_id` | btree | Fast lookup of all documents for a patient. |
| `ix_documents_patient_date` | `documents` | `(patient_id, document_date)` | btree | Date-range filtered document queries. |
| `ix_chunks_patient_date` | `chunks` | `(patient_id, document_date)` | btree | Combined patient + date filter before HNSW scan. |
| `ix_chunks_embedding_hnsw` | `chunks` | `embedding` | HNSW (`vector_cosine_ops`) | ANN search for cosine similarity. |
| `ix_users_username` | `users` | `username` | btree (unique) | Login lookup. |
| `ix_refresh_tokens_token_hash` | `refresh_tokens` | `token_hash` | btree | Token validation on refresh. |
| `ix_refresh_tokens_user_expires` | `refresh_tokens` | `(user_id, expires_at)` | btree | Mass-revoke on logout. |
| `ix_audit_log_actor_time` | `audit_log` | `(actor_id, created_at)` | btree | Per-actor audit trail query. |
| `ix_audit_log_patient_time` | `audit_log` | `(patient_id, created_at)` | btree | Per-patient audit trail query. |
| `ix_revoked_tokens_jti` | `revoked_tokens` | `jti` | btree (unique) | Per-request blacklist check. |

### HNSW index

The HNSW (Hierarchical Navigable Small World) index is created in `alembic/versions/0001_initial.py` via raw DDL (SQLAlchemy's `Index` with `postgresql_using="hnsw"` also declares it in `db/models.py:Chunk.__table_args__`):

```sql
CREATE INDEX ix_chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
```

HNSW builds a multi-layer graph during index creation. At query time, it performs approximate nearest-neighbor search in `O(log n)` probe steps rather than scanning every vector. `vector_cosine_ops` means the graph is optimized for cosine distance (the `<=>` operator).

Known limitation: when a `WHERE patient_id = :pid` filter is combined with the HNSW scan, the planner may apply the filter post-scan, returning fewer than `top_k` rows for patients with few documents. Setting `SET hnsw.ef_search = 40` (or enabling `hnsw.iterative_scan`) before the query would mitigate this. This is an open backlog item.

---

## 8. Document parsers (9 formats)

All parsers are in `core/document_parser.py`. The entry point is `parse_document(source_type, content, filename=None)` which dispatches to the format-specific function. All parsers return `list[Section]` where `Section(heading: str | None, text: str)`.

**pdf** (`_parse_pdf`): Uses `pdfplumber.open()`. Words are extracted with `extra_attrs=["size"]` to capture font sizes. Words are grouped into lines by `top` coordinate (within 3 points). The body font size is computed as the median of all non-zero line sizes — using median rather than mean avoids large headings skewing the baseline. A line is a heading if it has ≥3 alpha chars, ≤8 words, and either its font size exceeds the body size by 15% or all alphabetic characters are uppercase. If no heading is found, the entire document is returned as a single `Section(heading=None)`. Raises `ParseError` if the PDF contains no extractable text.

**fhir_json** (`_parse_fhir_json`): Accepts a FHIR R4 Bundle (`resourceType: "Bundle"`), a list of resources, or a single resource dict. Each resource is mapped to a section heading via `_FHIR_RESOURCE_TO_SECTION` (15 resource types covered; unknown types use the `resourceType` string directly). Rendering prefers `resource.text.div` (the FHIR human-readable narrative), stripping HTML tags with `re.sub(r"<[^>]+>", " ", ...)`. The fallback renderer only emits top-level scalar fields — nested `CodeableConcept` structures are not unrolled (documented as a known limitation in the source with a `TODO(Phase 5)` comment at line 217).

**hl7** (`_parse_hl7`): Decodes bytes as `latin-1` (safe superset of ASCII for 8-bit HL7 streams). Normalizes `\r\n` and `\n` to `\r` (HL7 canonical segment terminator). Parses with the `hl7.parse()` library. Segments are grouped by type using `_HL7_SEGMENT_TO_SECTION` (14 segment types covered). Each section's `text` is the raw segment string joined with newlines.

**csv** (`_parse_csv`): Decodes with `utf-8-sig` (strips BOM) or falls back to `latin-1`. Uses `csv.DictReader`. Each row is rendered as `"col1: val1, col2: val2, ..."` (empty values omitted). The section heading is derived from the filename stem (`"lab_results.csv"` → `"Lab Results"`) if available; otherwise from the first field name. Returns a single `Section`.

**dicom_meta** (`_parse_dicom_meta`): Lazy-imports `pydicom`. Reads with `stop_before_pixels=True` to avoid loading potentially gigabyte-sized pixel data. Extracts 15 named DICOM tags (patient ID, name, birth date, sex, study/series metadata, modality, institution). Returns a single `Section(heading="DICOM Metadata")`. Raises `ParseError` if none of the 15 tags are present.

**ccd_xml** (`_parse_ccd_xml`): Parses with `xml.etree.ElementTree`. Uses a `_local(tag)` helper to strip namespace prefixes so the code matches `section`, `title`, and `text` elements regardless of which HL7 namespace the document uses. Each `<section>` element becomes a `Section`. The `<text>` content is flattened with `"".join(child.itertext())` to handle nested tables and lists. Falls back to a full-text dump if no `<section>` elements are found.

**txt** (`_parse_txt`): Decodes with `utf-8-sig` or `latin-1`. Splits on blank lines (`\n\s*\n`) to get paragraphs. A paragraph's first line is treated as a heading if all alphabetic characters are uppercase, it has ≥3 alpha chars, and ≤8 words. The ALL-CAPS heuristic mimics how many legacy clinical plain-text reports denote section headers. If no heading is found, the entire text is collapsed to a single section.

**docx** (`_parse_docx`): Lazy-imports `python-docx`. Uses `paragraph.style.name` to detect headings (`"Heading 1"`, `"Heading 2"`, `"Heading 3"`). Non-heading paragraphs accumulate in the current section's body. Falls back to a single section if no heading styles are present.

**xlsx** (`_parse_xlsx`): Lazy-imports `openpyxl`. Opens with `read_only=True, data_only=True` (evaluated cell values, not formulas). The first row of each worksheet is treated as column headers. Subsequent rows are rendered as `"header: value, ..."`. Each worksheet becomes one `Section(heading=sheet.title)`. Raises `ParseError` if all worksheets are empty.

---

## 9. Configuration and secrets

| Variable | Read by | Why required |
|---|---|---|
| `GOOGLE_API_KEY` | `agents/synthesis_agent.py:_get_client()` via `genai.Client()` auto-read | Authenticates all Gemini API calls for synthesis. |
| `VOYAGE_API_KEY` | `core/embeddings.py:_get_client()` via `voyageai.AsyncClient()` auto-read; also by `core/chunker.py:_get_tokenizer_client()` | Used for both document and query embedding, and for token counting in the chunker. |
| `DATABASE_URL` | `api/routes.py:lifespan()`, `agents/ingestion_agent.py:_cli()`, `agents/ingestion_worker.py:_db_url()` | Postgres connection string. Rewritten to `postgresql+psycopg://` at startup. |
| `JWT_SECRET` | `core/auth.py:_get_secret()` | Signs and verifies HS256 JWTs. Required at every token issue and verify call. |
| `DB_ENCRYPTION_KEY` | `agents/audit_agent.py:_encryption_key()` | pgcrypto symmetric key for `pgp_sym_encrypt` / `pgp_sym_decrypt`. Required by every `log_response()` call. |
| `S3_ACCESS_KEY` | `core/s3.py:_client_kwargs()` | S3 / MinIO credentials. |
| `S3_SECRET_KEY` | `core/s3.py:_client_kwargs()` | S3 / MinIO credentials. |
| `S3_BUCKET` | `core/s3.py:_bucket()` | Target bucket for raw document storage. Auto-created on startup. |
| `S3_ENDPOINT_URL` | `core/s3.py:_client_kwargs()` | Override the S3 endpoint. Optional; omit for real AWS S3. |
| `REDIS_URL` | `agents/celery_app.py:_get_broker_url()`, `core/cache.py:_get_client()`, `api/routes.py:_redis_storage_uri()` | Celery broker + backend, query result cache, and slowapi rate limiter. Falls back to in-memory for the limiter if unset. |

Secrets (`JWT_SECRET`, `DB_ENCRYPTION_KEY`) should be at least 64 characters of cryptographic randomness. Generate with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

These two keys must be independent. Rotating `JWT_SECRET` invalidates all issued JWTs. Rotating `DB_ENCRYPTION_KEY` makes existing encrypted audit rows unreadable without a re-encryption migration.

---

## 10. Failure modes and exception types

| Exception | Defined in | Raised when | How callers handle it |
|---|---|---|---|
| `AuthError` | `core/auth.py` | Token signing/verification fails; password hash/verify fails; invalid credentials; account locked; username already taken; `JWT_SECRET` missing. | `api/routes.py` catches and maps to `HTTPException(401)` or `HTTPException(409)` depending on context. |
| `IngestionError` | `agents/ingestion_agent.py` | Zero chunks after parsing; embedding count mismatch; any Phase B pipeline failure. | `agents/ingestion_worker.py` propagates to Celery, which marks the task `FAILURE`. The `ingest_document()` `except` block also marks `documents.status = "failed"`. |
| `QueryError` | `agents/query_agent.py` | Voyage embedding fails; `search_chunks()` raises `VectorStoreError`. | `agents/orchestrator.py` catches `QueryError` and `SynthesisError` together, builds a refused stub result, and ensures `log_response()` still runs. |
| `SynthesisError` | `agents/synthesis_agent.py` | `google.genai.errors.APIError` from the Gemini call. | Same as `QueryError` — orchestrator builds stub and logs. |
| `OrchestratorError` | `agents/orchestrator.py` | Access-check audit write fails; both pipeline and audit writes fail; pipeline fails after audit was written. | `api/routes.py` catches and raises `HTTPException(500, "query failed")`. |
| `S3Error` | `core/s3.py` | `botocore.exceptions.ClientError` on `put_object` or `head_bucket` / `create_bucket`; required env vars missing. | `api/routes.py:ingest()` catches and raises `HTTPException(500, "S3 upload failed")`. Startup failure in `ensure_bucket_exists()` propagates as a fatal startup error. |
| `CacheError` | `core/cache.py` | `redis.RedisError` on get or set; `REDIS_URL` env var missing; corrupted JSON in cache is silently treated as a miss. | `api/routes.py:query()` catches `CacheError` on both read and write, treating cache as best-effort. The query proceeds without caching. |
| `AuditError` | `agents/audit_agent.py` | `SQLAlchemyError` on audit row insert; `DB_ENCRYPTION_KEY` not set. | `agents/orchestrator.py` treats audit failures as primary failures — they are surfaced over pipeline failures. `OrchestratorError` is raised with the `AuditError` as the cause. |
| `ParseError` | `core/document_parser.py` | PDF has no extractable text; FHIR JSON is not a valid structure; HL7 parse failure; CSV has no rows; DICOM has no recognised tags; CCD XML has no text; DOCX has no paragraphs; XLSX has no rows; required libraries not installed (pydicom, python-docx, openpyxl). | Propagates through `ingest_document()` Phase B, triggers Phase C failure handling, marks document `failed`. |
| `EmbeddingError` | `core/embeddings.py` | `voyageai.error.RateLimitError`, `ServiceUnavailableError`, `Timeout`, or `APIConnectionError` after SDK internal retries. | `agents/query_agent.py` catches and re-raises as `QueryError`. `agents/ingestion_agent.py` lets it propagate to Phase C (document marked failed). |
| `VectorStoreError` | `core/vector_store.py` | `sqlalchemy.exc.SQLAlchemyError` on `INSERT` (upsert_chunks) or `SELECT` (search_chunks). | `agents/query_agent.py` catches and re-raises as `QueryError`. `agents/ingestion_agent.py` lets it propagate to Phase C. |

---
