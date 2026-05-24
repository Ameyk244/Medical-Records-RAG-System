"""Add pgcrypto + encrypted audit_log columns.

Revision ID: 0002_encrypted_audit
Revises: 0001_initial
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_encrypted_audit"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.add_column("audit_log", sa.Column("query_text_enc", sa.LargeBinary(), nullable=True))
    op.add_column("audit_log", sa.Column("response_text_enc", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    op.drop_column("audit_log", "response_text_enc")
    op.drop_column("audit_log", "query_text_enc")
    # Extension drop is intentionally omitted — pgcrypto may be used by other tables in future.
