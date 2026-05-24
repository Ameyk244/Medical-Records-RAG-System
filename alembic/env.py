"""Alembic migrations env — sync psycopg3, reads DATABASE_URL from .env."""
from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Make project root importable so `from db.models import Base` works.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env from the project root (one level up from /alembic/env.py).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from db.models import Base  # noqa: E402 — must come after sys.path manipulation

config = context.config

# Standard logging setup from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve the database URL: prefer DATABASE_URL env, then alembic.ini's sqlalchemy.url.
db_url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
if not db_url:
    raise RuntimeError("DATABASE_URL not set (env or alembic.ini)")

# Normalise to psycopg3 sync driver. App uses the same URL but opts into async by
# passing it to create_async_engine; here we use a sync engine, so the same URL works.
if db_url.startswith("postgresql://") and "+" not in db_url.split("://", 1)[0]:
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

# Inject the resolved URL back into alembic config so engine_from_config picks it up.
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emits SQL to stdout without a live DB."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a real DB."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
