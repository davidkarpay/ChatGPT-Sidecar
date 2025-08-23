"""Initial schema migration

Revision ID: 001
Revises: 
Create Date: 2025-08-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Users
    op.create_table('user',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('display_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # Projects
    op.create_table('project',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'name')
    )

    # Documents
    op.create_table('document',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('doc_type', sa.String(), nullable=False),
        sa.Column('mime_type', sa.String(), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('fingerprint', sa.String(), nullable=True),
        sa.Column('metadata_json', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_document_type', 'document', ['doc_type'])
    op.create_index('idx_document_fp', 'document', ['fingerprint'])

    # Project <-> Document
    op.create_table('project_document',
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('project_id', 'document_id')
    )

    # Chunks
    op.create_table('chunk',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('start_char', sa.Integer(), nullable=True),
        sa.Column('end_char', sa.Integer(), nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_chunk_doc', 'chunk', ['document_id'])

    # Embedding references
    op.create_table('embedding_ref',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('chunk_id', sa.Integer(), nullable=False),
        sa.Column('index_name', sa.String(), nullable=False),
        sa.Column('vector_dim', sa.Integer(), nullable=False),
        sa.Column('faiss_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('index_name', 'chunk_id')
    )

    # Tags
    op.create_table('tag',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    op.create_table('document_tag',
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('tag_id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('document_id', 'tag_id')
    )

    # Facts
    op.create_table('fact',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', sa.String(), nullable=False),
        sa.Column('notes', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'key')
    )

    op.create_table('fact_tag',
        sa.Column('fact_id', sa.Integer(), nullable=False),
        sa.Column('tag_id', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('fact_id', 'tag_id')
    )

    # Citations: fact <-> chunk
    op.create_table('citation',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('fact_id', sa.Integer(), nullable=False),
        sa.Column('chunk_id', sa.Integer(), nullable=False),
        sa.Column('excerpt', sa.String(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True, server_default='1.0'),
        sa.PrimaryKeyConstraint('id')
    )

    # Conversation normalization
    op.create_table('conversation',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id')
    )

    op.create_table('message',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('created_at', sa.String(), nullable=True),
        sa.Column('metadata_json', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Source links
    op.create_table('source_link',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('link_type', sa.String(), nullable=False),
        sa.Column('href', sa.String(), nullable=False),
        sa.Column('notes', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # User identity management for personalized model responses
    op.create_table('user_identity',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('preferences_json', sa.String(), nullable=True),
        sa.Column('is_active', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'full_name'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'])
    )

    # User sessions linked to user identity for personalized responses
    op.create_table('user_session',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('user_identity_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_active_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('session_data_json', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id'),
        sa.ForeignKeyConstraint(['user_identity_id'], ['user_identity.id'])
    )
    op.create_index('idx_user_session_session_id', 'user_session', ['session_id'])
    op.create_index('idx_user_session_identity', 'user_session', ['user_identity_id'])
    op.create_index('idx_user_session_active', 'user_session', ['last_active_at'])

    # Training data collection for local model fine-tuning
    op.create_table('training_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('user_query', sa.String(), nullable=False),
        sa.Column('model_response', sa.String(), nullable=False),
        sa.Column('context_json', sa.String(), nullable=True),
        sa.Column('feedback_rating', sa.Integer(), nullable=True),
        sa.Column('feedback_correction', sa.String(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('user_identity_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_identity_id'], ['user_identity.id'])
    )
    op.create_index('idx_training_data_session', 'training_data', ['session_id'])
    op.create_index('idx_training_data_rating', 'training_data', ['feedback_rating'])
    op.create_index('idx_training_data_model', 'training_data', ['model_name'])

    # Model versions and fine-tuning tracking
    op.create_table('model_version',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('base_model', sa.String(), nullable=False),
        sa.Column('version_name', sa.String(), nullable=False),
        sa.Column('adapter_path', sa.String(), nullable=True),
        sa.Column('training_config_json', sa.String(), nullable=True),
        sa.Column('training_data_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('is_active', sa.Integer(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('base_model', 'version_name')
    )

    # Training sessions for tracking fine-tuning runs
    op.create_table('training_session',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('training_data_filter', sa.String(), nullable=True),
        sa.Column('config_json', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=True, server_default="'pending'"),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('metrics_json', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_version.id'])
    )

    # Benchmark prompts for consistent model evaluation
    op.create_table('benchmark_prompt',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('prompt', sa.String(), nullable=False),
        sa.Column('context_json', sa.String(), nullable=True),
        sa.Column('expected_qualities_json', sa.String(), nullable=True),
        sa.Column('scoring_criteria_json', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )

    # Evaluation results for benchmark prompt responses
    op.create_table('evaluation_result',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('prompt_id', sa.String(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('response', sa.String(), nullable=False),
        sa.Column('scores_json', sa.String(), nullable=False),
        sa.Column('metadata_json', sa.String(), nullable=True),
        sa.Column('evaluated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['prompt_id'], ['benchmark_prompt.id']),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_version.id'])
    )
    op.create_index('idx_evaluation_result_prompt', 'evaluation_result', ['prompt_id'])
    op.create_index('idx_evaluation_result_model', 'evaluation_result', ['model_version_id'])
    op.create_index('idx_evaluation_result_date', 'evaluation_result', ['evaluated_at'])

    # Model snapshots for tracking parameter and performance changes
    op.create_table('model_snapshot',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('parameter_stats_json', sa.String(), nullable=False),
        sa.Column('embedding_analysis_json', sa.String(), nullable=False),
        sa.Column('performance_metrics_json', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_version.id'])
    )
    op.create_index('idx_model_snapshot_version', 'model_snapshot', ['model_version_id'])
    op.create_index('idx_model_snapshot_date', 'model_snapshot', ['created_at'])

    # Monthly check-in reports
    op.create_table('monthly_checkin',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_period', sa.String(), nullable=False),
        sa.Column('training_stats_json', sa.String(), nullable=False),
        sa.Column('performance_comparison_json', sa.String(), nullable=True),
        sa.Column('recommendations_json', sa.String(), nullable=True),
        sa.Column('report_data_json', sa.String(), nullable=True),
        sa.Column('generated_at', sa.DateTime(), nullable=True, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('report_period')
    )
    op.create_index('idx_monthly_checkin_period', 'monthly_checkin', ['report_period'])


def downgrade() -> None:
    # Drop tables in reverse order to avoid foreign key constraints
    op.drop_table('monthly_checkin')
    op.drop_table('model_snapshot')
    op.drop_table('evaluation_result')
    op.drop_table('benchmark_prompt')
    op.drop_table('training_session')
    op.drop_table('model_version')
    op.drop_table('training_data')
    op.drop_table('user_session')
    op.drop_table('user_identity')
    op.drop_table('source_link')
    op.drop_table('message')
    op.drop_table('conversation')
    op.drop_table('citation')
    op.drop_table('fact_tag')
    op.drop_table('fact')
    op.drop_table('document_tag')
    op.drop_table('tag')
    op.drop_table('embedding_ref')
    op.drop_table('chunk')
    op.drop_table('project_document')
    op.drop_table('document')
    op.drop_table('project')
    op.drop_table('user')