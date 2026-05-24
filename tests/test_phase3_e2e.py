"""Phase 3 end-to-end verification — audit rows, denial path, append-only trigger."""
import asyncio
import os
import sys

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agents.audit_agent import check_access
from agents.orchestrator import handle_query
from db.models import AuditLog


def _db_url() -> str:
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def _count_audit(session) -> int:
    result = await session.execute(select(func.count()).select_from(AuditLog))
    return result.scalar_one()


async def _count_audit_by_action(session, action: str) -> int:
    result = await session.execute(
        select(func.count()).select_from(AuditLog).where(AuditLog.action == action)
    )
    return result.scalar_one()


async def main() -> None:
    engine = create_async_engine(_db_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    failures: list[str] = []

    try:
        # --- Scenario 1: allowed query via orchestrator writes 2 audit rows ---
        async with session_factory() as session:
            print("\n========== SCENARIO 1: allowed query writes 2 audit rows ==========")
            before = await _count_audit(session)
            result = await handle_query(
                query_text="What lab values were flagged?",
                patient_id="P123",
                actor_id="doctor_smith",
                actor_role="physician",
                session=session,
            )
            after = await _count_audit(session)
            delta = after - before
            print(f"audit rows: {before} → {after} (delta = {delta})")
            print(f"orchestrator returned refused={result.refused}, citations={len(result.citations)}")
            if delta == 2:
                print("✓ PASS — exactly 2 audit rows written")
            else:
                failures.append(f"Scenario 1: expected delta=2, got {delta}")
                print(f"✗ FAIL — expected delta=2, got {delta}")

        # --- Scenario 2: denied check_access writes 1 access_denied row ---
        async with session_factory() as session:
            print("\n========== SCENARIO 2: denied access writes 1 access_denied row ==========")
            print("(Calling check_access directly — orchestrator's ValueError on empty")
            print(" actor_id intentionally blocks this from being reachable via handle_query.)")
            before_total = await _count_audit(session)
            before_denied = await _count_audit_by_action(session, "access_denied")
            allowed = await check_access(
                actor_id="",
                actor_role=None,
                patient_id="P123",
                session=session,
            )
            after_total = await _count_audit(session)
            after_denied = await _count_audit_by_action(session, "access_denied")
            print(f"check_access returned allowed={allowed}")
            print(f"total audit rows: {before_total} → {after_total} (delta = {after_total - before_total})")
            print(f"access_denied rows: {before_denied} → {after_denied} (delta = {after_denied - before_denied})")
            if allowed is False and (after_total - before_total) == 1 and (after_denied - before_denied) == 1:
                print("✓ PASS — exactly 1 access_denied row written, allowed=False")
            else:
                failures.append(
                    f"Scenario 2: allowed={allowed}, total_delta={after_total - before_total}, "
                    f"denied_delta={after_denied - before_denied}"
                )
                print("✗ FAIL — see above")

        # --- Scenario 3: UPDATE on audit_log is blocked by the trigger ---
        async with session_factory() as session:
            print("\n========== SCENARIO 3: UPDATE on audit_log blocked by trigger ==========")
            try:
                await session.execute(
                    text("UPDATE audit_log SET response_text = 'tampered'")
                )
                await session.commit()
                print("✗ FAIL — UPDATE was NOT blocked")
                failures.append("Scenario 3: UPDATE succeeded — trigger missing or broken")
            except Exception as e:
                msg = str(e)
                print(f"got exception type: {type(e).__name__}")
                print(f"message tail: ...{msg[-200:]}")
                if "audit_log is append-only" in msg:
                    print("✓ PASS — trigger raised the expected message")
                else:
                    failures.append(f"Scenario 3: unexpected exception message: {msg[:200]}")
                    print("✗ FAIL — trigger raised but message did not match expected")
                await session.rollback()

    finally:
        await engine.dispose()

    print("\n========== SUMMARY ==========")
    if failures:
        print(f"✗ {len(failures)} scenario(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✓ All 3 scenarios passed.")


if __name__ == "__main__":
    asyncio.run(main())
