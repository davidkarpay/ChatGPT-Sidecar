"""Add sync service tables

Revision ID: ea9be782be5e
Revises: bda5b9a93bfb
Create Date: 2025-08-19 20:26:54.942721

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ea9be782be5e'
down_revision = 'bda5b9a93bfb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # User sync configuration
    op.create_table('user_sync_config',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('sync_enabled', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('chatgpt_export_url', sa.String(), nullable=True),
        sa.Column('sync_frequency_hours', sa.Integer(), nullable=True, server_default='24'),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'])
    )

    # Sync history tracking
    op.create_table('sync_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sync_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('sync_type', sa.String(), nullable=True, server_default="'scheduled'"),  # 'scheduled', 'manual'
        sa.Column('status', sa.String(), nullable=False),  # 'running', 'completed', 'failed'
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('files_processed', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('conversations_added', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('conversations_updated', sa.Integer(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sync_id'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'])
    )
    op.create_index('idx_sync_history_user', 'sync_history', ['user_id'])
    op.create_index('idx_sync_history_status', 'sync_history', ['status'])
    op.create_index('idx_sync_history_started', 'sync_history', ['started_at'])

    # File processing history for deduplication
    op.create_table('sync_file_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('file_hash', sa.String(), nullable=False),
        sa.Column('sync_id', sa.String(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'file_hash'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'])
    )
    op.create_index('idx_sync_file_user_hash', 'sync_file_history', ['user_id', 'file_hash'])


def downgrade() -> None:
    op.drop_table('sync_file_history')
    op.drop_table('sync_history')
    op.drop_table('user_sync_config')