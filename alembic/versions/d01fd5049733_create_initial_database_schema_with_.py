"""Create initial database schema with documents, tasks, billing_records, and quality_issues tables

Revision ID: d01fd5049733
Revises: 
Create Date: 2025-12-21 00:05:27.354393

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd01fd5049733'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('source_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', sa.String(length=100), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED', name='taskstatus'), nullable=True),
        sa.Column('annotations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ai_predictions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create billing_records table
    op.create_table(
        'billing_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('annotation_count', sa.Integer(), nullable=True),
        sa.Column('time_spent', sa.Integer(), nullable=True),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('billing_date', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_DATE'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create quality_issues table
    op.create_table(
        'quality_issues',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('issue_type', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='issueseverity'), nullable=True),
        sa.Column('status', sa.Enum('OPEN', 'IN_PROGRESS', 'RESOLVED', 'CLOSED', name='issuestatus'), nullable=True),
        sa.Column('assignee_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['task_id'], ['tasks.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create GIN indexes for JSONB columns to optimize queries
    op.create_index('idx_documents_metadata_gin', 'documents', ['metadata'], postgresql_using='gin')
    op.create_index('idx_documents_source_config_gin', 'documents', ['source_config'], postgresql_using='gin')
    op.create_index('idx_tasks_annotations_gin', 'tasks', ['annotations'], postgresql_using='gin')
    op.create_index('idx_tasks_ai_predictions_gin', 'tasks', ['ai_predictions'], postgresql_using='gin')
    
    # Create additional indexes for performance
    op.create_index('idx_tasks_document_id', 'tasks', ['document_id'])
    op.create_index('idx_tasks_project_id', 'tasks', ['project_id'])
    op.create_index('idx_tasks_status', 'tasks', ['status'])
    op.create_index('idx_billing_records_tenant_id', 'billing_records', ['tenant_id'])
    op.create_index('idx_billing_records_user_id', 'billing_records', ['user_id'])
    op.create_index('idx_billing_records_billing_date', 'billing_records', ['billing_date'])
    op.create_index('idx_quality_issues_task_id', 'quality_issues', ['task_id'])
    op.create_index('idx_quality_issues_status', 'quality_issues', ['status'])
    op.create_index('idx_quality_issues_assignee_id', 'quality_issues', ['assignee_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index('idx_quality_issues_assignee_id', table_name='quality_issues')
    op.drop_index('idx_quality_issues_status', table_name='quality_issues')
    op.drop_index('idx_quality_issues_task_id', table_name='quality_issues')
    op.drop_index('idx_billing_records_billing_date', table_name='billing_records')
    op.drop_index('idx_billing_records_user_id', table_name='billing_records')
    op.drop_index('idx_billing_records_tenant_id', table_name='billing_records')
    op.drop_index('idx_tasks_status', table_name='tasks')
    op.drop_index('idx_tasks_project_id', table_name='tasks')
    op.drop_index('idx_tasks_document_id', table_name='tasks')
    op.drop_index('idx_tasks_ai_predictions_gin', table_name='tasks')
    op.drop_index('idx_tasks_annotations_gin', table_name='tasks')
    op.drop_index('idx_documents_source_config_gin', table_name='documents')
    op.drop_index('idx_documents_metadata_gin', table_name='documents')
    
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table('quality_issues')
    op.drop_table('billing_records')
    op.drop_table('tasks')
    op.drop_table('documents')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS issuestatus')
    op.execute('DROP TYPE IF EXISTS issueseverity')
    op.execute('DROP TYPE IF EXISTS taskstatus')
