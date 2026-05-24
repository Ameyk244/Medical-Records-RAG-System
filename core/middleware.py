"""HTTP middleware — security headers, X-Request-ID, structured logging, Prometheus."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from starlette.middleware.base import BaseHTTPMiddleware

_log = structlog.get_logger()


# Docs paths bypass the strict CSP — Swagger UI needs inline scripts + CDN assets.
# Real API responses still get strict CSP. Tighten further once docs is self-hosted.
_CSP_BYPASS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path not in _CSP_BYPASS_PATHS:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            # Clear request-scoped context vars so they don't bleed into
            # the next request handled by the same worker process/coroutine.
            structlog.contextvars.unbind_contextvars("request_id", "actor_id")

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        _log.info(
            "request_complete",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def setup_metrics(app: FastAPI) -> None:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/health", "/health/ready", "/health/live"],
    )
    instrumentator.add(metrics.default())
    instrumentator.add(metrics.latency())
    instrumentator.add(metrics.requests())
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
