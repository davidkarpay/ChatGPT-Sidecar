"""Add alternative auth methods

Revision ID: 7f67210111fe
Revises: ea9be782be5e
Create Date: 2025-08-19 21:02:12.267665

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7f67210111fe'
down_revision = 'ea9be782be5e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Update user_oauth to support multiple providers
    op.add_column('user_oauth', sa.Column('zoho_id', sa.String(), nullable=True))
    op.add_column('user_oauth', sa.Column('access_token', sa.String(), nullable=True))
    op.add_column('user_oauth', sa.Column('refresh_token', sa.String(), nullable=True))
    
    # Password authentication table
    op.create_table('user_passwords',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'])
    )


def downgrade() -> None:
    op.drop_table('user_passwords')
    op.drop_column('user_oauth', 'refresh_token')
    op.drop_column('user_oauth', 'access_token')
    op.drop_column('user_oauth', 'zoho_id')