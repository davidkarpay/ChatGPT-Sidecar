"""PostgreSQL compatibility update

Revision ID: e2b1179e2d52
Revises: 001
Create Date: 2025-08-19 20:22:07.745284

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e2b1179e2d52'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL specific updates - these operations are compatible with both SQLite and PostgreSQL
    # No changes needed for now since the schema is already compatible
    pass


def downgrade() -> None:
    # No changes needed for compatibility migration
    pass