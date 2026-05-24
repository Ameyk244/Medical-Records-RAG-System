"""Phase 4a HTTP end-to-end — auth, ingest, query, auth failures. ASGI in-process."""
import asyncio
import sys
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from api.routes import app, lifespan

BASE_URL = "http://test"
SAMPLE_CSV = Path("tests/sample_labs.csv")


def _check(label: str, ok: bool, details: str, failures: list[str]) -> None:
    if ok:
        print(f"✓ PASS — {label}")
    else:
        print(f"✗ FAIL — {label}: {details}")
        failures.append(f"{label}: {details}")


async def main() -> None:
    failures: list[str] = []

    # The lifespan() context manager initialises the engine + session factory on
    # app.state. ASGITransport does not run lifespan itself; calling it explicitly
    # is the supported pattern for in-process tests.
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
            # ----- Test 1: login valid -----
            r = await client.post(
                "/auth/login",
                json={"username": "doctor1", "password": "password123"},
            )
            ok = (
                r.status_code == 200
                and isinstance(r.json().get("access_token"), str)
                and len(r.json().get("access_token", "")) > 20
                and r.json().get("token_type") == "bearer"
            )
            _check(
                "login valid",
                ok,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )
            token = r.json().get("access_token", "") if r.status_code == 200 else ""

            # ----- Test 2: login wrong password -----
            r = await client.post(
                "/auth/login",
                json={"username": "doctor1", "password": "wrong"},
            )
            _check(
                "login wrong password → 401",
                r.status_code == 401,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )

            # ----- Test 3: ingest with valid JWT -----
            csv_bytes = SAMPLE_CSV.read_bytes()
            r = await client.post(
                "/ingest",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("sample_labs.csv", csv_bytes, "text/csv")},
                data={
                    "patient_id": "P123",
                    "source_type": "csv",
                    "document_date": "2025-11-03",
                },
            )
            ok = r.status_code == 200 and "document_id" in r.json()
            doc_id = r.json().get("document_id", "") if r.status_code == 200 else ""
            _check(
                "ingest valid → 200, document_id returned",
                ok,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )
            if ok:
                print(f"  document_id: {doc_id}")

            # ----- Test 4: ingest with no JWT -----
            r = await client.post(
                "/ingest",
                files={"file": ("sample_labs.csv", csv_bytes, "text/csv")},
                data={"patient_id": "P123", "source_type": "csv"},
            )
            _check(
                "ingest no JWT → 401",
                r.status_code == 401,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )

            # ----- Test 5: query with valid JWT -----
            r = await client.post(
                "/query",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "patient_id": "P123",
                    "query_text": "What were the elevated lab values?",
                },
            )
            ok = (
                r.status_code == 200
                and "answer" in r.json()
                and "citations" in r.json()
                and "refused" in r.json()
            )
            _check(
                "query valid → 200, full payload",
                ok,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )
            if ok:
                body = r.json()
                print(f"  refused={body['refused']}, citations={len(body['citations'])}")
                print(f"  answer: {body['answer'][:200]}")

            # ----- Test 6: query with no JWT -----
            r = await client.post(
                "/query",
                json={"patient_id": "P123", "query_text": "anything"},
            )
            _check(
                "query no JWT → 401",
                r.status_code == 401,
                f"status={r.status_code} body={r.text[:200]}",
                failures,
            )

    print("\n========== SUMMARY ==========")
    if failures:
        print(f"✗ {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✓ All 6 HTTP tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
