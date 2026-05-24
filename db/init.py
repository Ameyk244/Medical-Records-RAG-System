# LOCAL DEV ONLY — use `alembic upgrade head` for production.
"""Phase 1 smoke-test bootstrap — enable pgvector + create_all + audit trigger. Replaced by Alembic in Phase 4."""

# Phase 1 only — replace with Alembic migrations in Phase 4.

from __future__ import annotations
import asyncio
import os
import sys
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from db.models import Base


# Append-only enforcement for audit_log. CREATE OR REPLACE makes the function idempotent;
# DROP TRIGGER IF EXISTS + CREATE TRIGGER makes the trigger idempotent (Postgres has no
# CREATE TRIGGER IF NOT EXISTS as of PG 16).
_AUDIT_TRIGGER_FN = """
CREATE OR REPLACE FUNCTION raise_on_audit_mutation()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;
"""

_AUDIT_TRIGGER_DROP = "DROP TRIGGER IF EXISTS audit_log_no_mutation ON audit_log;"

_AUDIT_TRIGGER_CREATE = """
CREATE TRIGGER audit_log_no_mutation
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION raise_on_audit_mutation();
"""


async def _init() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("error: DATABASE_URL env var is required", file=sys.stderr)
        sys.exit(1)

    if db_url.startswith("postgresql://") and "+" not in db_url.split("://", 1)[0]:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_async_engine(db_url, future=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("✓ extension vector enabled")
            await conn.run_sync(Base.metadata.create_all)
            print("✓ tables created")
            await conn.execute(text(_AUDIT_TRIGGER_FN))
            await conn.execute(text(_AUDIT_TRIGGER_DROP))
            await conn.execute(text(_AUDIT_TRIGGER_CREATE))
            print("✓ audit_log append-only trigger installed")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_init())
