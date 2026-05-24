"""Celery application — broker + backend on Redis (REDIS_URL env)."""
from __future__ import annotations

import os

from celery import Celery


def _get_broker_url() -> str:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL env var is required for Celery")
    return url


celery_app = Celery(
    "medical_rag",
    broker=_get_broker_url(),
    backend=_get_broker_url(),
    include=["agents.ingestion_worker"],
)

# Conservative defaults — these can be tuned in Phase 5b.
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    result_expires=3600,  # 1 hour
    timezone="UTC",
)
