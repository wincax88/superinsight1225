"""Add data sync system tables and extend existing models

Revision ID: sync_001
Revises: security_001
Create Date: 2024-12-24 10:00:00.000000

This migration creates all tables required for the data synchronization system:
- data_sources: Data source configuration
- sync_jobs: Sync job configuration
- sync_executions: Sync execution history
- data_conflicts: Data conflict records
- sync_rules: Sync rules configuration
- transformation_rules: Data transformation rules
- industry_datasets: Industry dataset metadata
- sync_audit_logs: Sync operation audit logs
- data_quality_scores: Data quality tracking

It also extends existing tables (documents, tasks) with sync-related fields.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'sync_001'
down_revision = 'security_001'
branch_labels = None
depends_on = None


def upgrade():
    # ========================================================================
    # Create ENUM types for sync system
    # ========================================================================

    # SyncDirection enum
    op.execute("""
        CREATE TYPE syncdirection AS ENUM (
            'local_to_cloud', 'cloud_to_local', 'bidirectional', 'pull', 'push'
        )
    """)

    # SyncFrequency enum
    op.execute("""
        CREATE TYPE syncfrequency AS ENUM (
            'real_time', 'scheduled', 'manual', 'on_change'
        )
    """)

    # SyncJobStatus enum
    op.execute("""
        CREATE TYPE syncjobstatus AS ENUM (
            'draft', 'active', 'paused', 'disabled', 'archived'
        )
    """)

    # SyncExecutionStatus enum
    op.execute("""
        CREATE TYPE syncexecutionstatus AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled', 'partial'
        )
    """)

    # ConflictType enum
    op.execute("""
        CREATE TYPE conflicttype AS ENUM (
            'version_conflict', 'content_conflict', 'schema_conflict',
            'delete_conflict', 'constraint_conflict'
        )
    """)

    # ConflictResolutionStrategy enum
    op.execute("""
        CREATE TYPE conflictresolutionstrategy AS ENUM (
            'timestamp_based', 'source_wins', 'target_wins',
            'manual', 'field_merge', 'business_rule'
        )
    """)

    # ConflictStatus enum
    op.execute("""
        CREATE TYPE conflictstatus AS ENUM (
            'pending', 'auto_resolved', 'manually_resolved', 'escalated', 'ignored'
        )
    """)

    # DataSourceType enum
    op.execute("""
        CREATE TYPE datasourcetype AS ENUM (
            'mysql', 'postgresql', 'oracle', 'mongodb', 'sqlserver',
            'rest_api', 'graphql_api', 'soap_api',
            'ftp', 'sftp', 's3', 'local_file', 'webhook'
        )
    """)

    # DataSourceStatus enum
    op.execute("""
        CREATE TYPE datasourcestatus AS ENUM (
            'active', 'inactive', 'error', 'testing'
        )
    """)

    # TransformationType enum
    op.execute("""
        CREATE TYPE transformationtype AS ENUM (
            'field_mapping', 'data_type_conversion', 'value_transformation',
            'aggregation', 'filtering', 'enrichment', 'normalization', 'custom_script'
        )
    """)

    # DatasetCategory enum
    op.execute("""
        CREATE TYPE datasetcategory AS ENUM (
            'finance', 'healthcare', 'legal', 'general',
            'technology', 'retail', 'manufacturing', 'education'
        )
    """)

    # DatasetStatus enum
    op.execute("""
        CREATE TYPE datasetstatus AS ENUM (
            'available', 'downloading', 'processing', 'integrated', 'error', 'deprecated'
        )
    """)

    # AuditAction enum
    op.execute("""
        CREATE TYPE auditaction AS ENUM (
            'job_created', 'job_updated', 'job_deleted',
            'job_started', 'job_stopped', 'job_paused', 'job_resumed',
            'sync_started', 'sync_completed', 'sync_failed',
            'conflict_detected', 'conflict_resolved',
            'data_pushed', 'data_pulled',
            'source_connected', 'source_disconnected'
        )
    """)

    # SyncStatus enum for documents/tasks
    op.execute("""
        CREATE TYPE syncstatus AS ENUM (
            'pending', 'syncing', 'synced', 'conflict', 'failed'
        )
    """)

    # ========================================================================
    # Extend existing tables with sync-related fields
    # ========================================================================

    # Extend documents table
    op.add_column('documents', sa.Column('tenant_id', sa.String(100), nullable=True))
    op.add_column('documents', sa.Column('sync_status', postgresql.ENUM('pending', 'syncing', 'synced', 'conflict', 'failed', name='syncstatus', create_type=False), nullable=True))
    op.add_column('documents', sa.Column('sync_version', sa.Integer(), nullable=True, server_default='1'))
    op.add_column('documents', sa.Column('sync_hash', sa.String(64), nullable=True))
    op.add_column('documents', sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents', sa.Column('sync_source_id', sa.String(200), nullable=True))
    op.add_column('documents', sa.Column('sync_job_id', sa.String(36), nullable=True))
    op.add_column('documents', sa.Column('is_from_sync', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('documents', sa.Column('sync_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'))

    # Add index for tenant_id on documents
    op.create_index('idx_documents_tenant_id', 'documents', ['tenant_id'])

    # Extend tasks table
    op.add_column('tasks', sa.Column('tenant_id', sa.String(100), nullable=True))
    op.add_column('tasks', sa.Column('sync_status', postgresql.ENUM('pending', 'syncing', 'synced', 'conflict', 'failed', name='syncstatus', create_type=False), nullable=True))
    op.add_column('tasks', sa.Column('sync_version', sa.Integer(), nullable=True, server_default='1'))
    op.add_column('tasks', sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('sync_execution_id', sa.String(36), nullable=True))
    op.add_column('tasks', sa.Column('is_from_sync', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('tasks', sa.Column('sync_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'))

    # Add index for tenant_id on tasks
    op.create_index('idx_tasks_tenant_id', 'tasks', ['tenant_id'])

    # ========================================================================
    # Create data_sources table
    # ========================================================================
    op.create_table('data_sources',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_type', postgresql.ENUM('mysql', 'postgresql', 'oracle', 'mongodb', 'sqlserver', 'rest_api', 'graphql_api', 'soap_api', 'ftp', 'sftp', 's3', 'local_file', 'webhook', name='datasourcetype', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('active', 'inactive', 'error', 'testing', name='datasourcestatus', create_type=False), nullable=True, server_default='inactive'),
        sa.Column('connection_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('schema_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('pool_size', sa.Integer(), nullable=True, server_default='5'),
        sa.Column('max_overflow', sa.Integer(), nullable=True, server_default='10'),
        sa.Column('connection_timeout', sa.Integer(), nullable=True, server_default='30'),
        sa.Column('last_health_check', sa.DateTime(timezone=True), nullable=True),
        sa.Column('health_check_status', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_data_sources_tenant_type', 'data_sources', ['tenant_id', 'source_type'])
    op.create_index('idx_data_sources_tenant_id', 'data_sources', ['tenant_id'])

    # ========================================================================
    # Create sync_jobs table
    # ========================================================================
    op.create_table('sync_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('direction', postgresql.ENUM('local_to_cloud', 'cloud_to_local', 'bidirectional', 'pull', 'push', name='syncdirection', create_type=False), nullable=True, server_default="'pull'"),
        sa.Column('frequency', postgresql.ENUM('real_time', 'scheduled', 'manual', 'on_change', name='syncfrequency', create_type=False), nullable=True, server_default="'manual'"),
        sa.Column('status', postgresql.ENUM('draft', 'active', 'paused', 'disabled', 'archived', name='syncjobstatus', create_type=False), nullable=True, server_default="'draft'"),
        sa.Column('schedule_cron', sa.String(100), nullable=True),
        sa.Column('schedule_timezone', sa.String(50), nullable=True, server_default="'UTC'"),
        sa.Column('batch_size', sa.Integer(), nullable=True, server_default='1000'),
        sa.Column('max_retries', sa.Integer(), nullable=True, server_default='3'),
        sa.Column('retry_delay', sa.Integer(), nullable=True, server_default='60'),
        sa.Column('timeout', sa.Integer(), nullable=True, server_default='3600'),
        sa.Column('enable_incremental', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('incremental_field', sa.String(100), nullable=True),
        sa.Column('last_sync_value', sa.String(500), nullable=True),
        sa.Column('conflict_resolution', postgresql.ENUM('timestamp_based', 'source_wins', 'target_wins', 'manual', 'field_merge', 'business_rule', name='conflictresolutionstrategy', create_type=False), nullable=True, server_default="'timestamp_based'"),
        sa.Column('transformation_rules', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('filter_conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('enable_encryption', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('enable_compression', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('total_executions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('successful_executions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('failed_executions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_records_synced', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('last_executed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(100), nullable=True),
        sa.ForeignKeyConstraint(['source_id'], ['data_sources.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sync_jobs_tenant_status', 'sync_jobs', ['tenant_id', 'status'])
    op.create_index('idx_sync_jobs_next_scheduled', 'sync_jobs', ['next_scheduled_at'])
    op.create_index('idx_sync_jobs_tenant_id', 'sync_jobs', ['tenant_id'])

    # ========================================================================
    # Create sync_executions table
    # ========================================================================
    op.create_table('sync_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'cancelled', 'partial', name='syncexecutionstatus', create_type=False), nullable=True, server_default="'pending'"),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('records_total', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_processed', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_inserted', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_updated', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_deleted', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_skipped', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('records_failed', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('bytes_transferred', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('checkpoint_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('triggered_by', sa.String(100), nullable=True),
        sa.Column('trigger_type', sa.String(50), nullable=True, server_default="'manual'"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['sync_jobs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sync_executions_job_status', 'sync_executions', ['job_id', 'status'])
    op.create_index('idx_sync_executions_tenant_time', 'sync_executions', ['tenant_id', 'created_at'])
    op.create_index('idx_sync_executions_job_id', 'sync_executions', ['job_id'])
    op.create_index('idx_sync_executions_tenant_id', 'sync_executions', ['tenant_id'])

    # ========================================================================
    # Create data_conflicts table
    # ========================================================================
    op.create_table('data_conflicts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('conflict_type', postgresql.ENUM('version_conflict', 'content_conflict', 'schema_conflict', 'delete_conflict', 'constraint_conflict', name='conflicttype', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'auto_resolved', 'manually_resolved', 'escalated', 'ignored', name='conflictstatus', create_type=False), nullable=True, server_default="'pending'"),
        sa.Column('record_id', sa.String(500), nullable=False),
        sa.Column('table_name', sa.String(200), nullable=True),
        sa.Column('field_name', sa.String(200), nullable=True),
        sa.Column('source_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('target_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('source_version', sa.String(100), nullable=True),
        sa.Column('target_version', sa.String(100), nullable=True),
        sa.Column('source_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('target_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_strategy', postgresql.ENUM('timestamp_based', 'source_wins', 'target_wins', 'manual', 'field_merge', 'business_rule', name='conflictresolutionstrategy', create_type=False), nullable=True),
        sa.Column('resolved_value', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('resolved_by', sa.String(100), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True, server_default='5'),
        sa.Column('detected_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['execution_id'], ['sync_executions.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_data_conflicts_execution_status', 'data_conflicts', ['execution_id', 'status'])
    op.create_index('idx_data_conflicts_tenant_pending', 'data_conflicts', ['tenant_id', 'status'])
    op.create_index('idx_data_conflicts_execution_id', 'data_conflicts', ['execution_id'])
    op.create_index('idx_data_conflicts_tenant_id', 'data_conflicts', ['tenant_id'])

    # ========================================================================
    # Create sync_rules table
    # ========================================================================
    op.create_table('sync_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('table_pattern', sa.String(200), nullable=True),
        sa.Column('field_pattern', sa.String(200), nullable=True),
        sa.Column('direction', postgresql.ENUM('local_to_cloud', 'cloud_to_local', 'bidirectional', 'pull', 'push', name='syncdirection', create_type=False), nullable=True),
        sa.Column('enabled', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('priority', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('sync_deletes', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('sync_updates', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('sync_inserts', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('filter_conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('conflict_resolution', postgresql.ENUM('timestamp_based', 'source_wins', 'target_wins', 'manual', 'field_merge', 'business_rule', name='conflictresolutionstrategy', create_type=False), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['sync_jobs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sync_rules_job_id', 'sync_rules', ['job_id'])
    op.create_index('idx_sync_rules_tenant_id', 'sync_rules', ['tenant_id'])

    # ========================================================================
    # Create transformation_rules table
    # ========================================================================
    op.create_table('transformation_rules',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('transformation_type', postgresql.ENUM('field_mapping', 'data_type_conversion', 'value_transformation', 'aggregation', 'filtering', 'enrichment', 'normalization', 'custom_script', name='transformationtype', create_type=False), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('priority', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('source_field', sa.String(200), nullable=True),
        sa.Column('target_field', sa.String(200), nullable=True),
        sa.Column('transformation_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('custom_script', sa.Text(), nullable=True),
        sa.Column('script_language', sa.String(50), nullable=True, server_default="'python'"),
        sa.Column('validation_rules', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_transformation_rules_tenant_id', 'transformation_rules', ['tenant_id'])

    # ========================================================================
    # Create industry_datasets table
    # ========================================================================
    op.create_table('industry_datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.String(50), nullable=True, server_default="'1.0.0'"),
        sa.Column('source_platform', sa.String(100), nullable=False),
        sa.Column('source_url', sa.String(500), nullable=False),
        sa.Column('source_identifier', sa.String(200), nullable=False),
        sa.Column('category', postgresql.ENUM('finance', 'healthcare', 'legal', 'general', 'technology', 'retail', 'manufacturing', 'education', name='datasetcategory', create_type=False), nullable=False),
        sa.Column('domain_tags', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('language', sa.String(50), nullable=True, server_default="'zh'"),
        sa.Column('status', postgresql.ENUM('available', 'downloading', 'processing', 'integrated', 'error', 'deprecated', name='datasetstatus', create_type=False), nullable=True, server_default="'available'"),
        sa.Column('is_public', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('total_records', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True, server_default='0'),
        sa.Column('quality_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('quality_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('local_path', sa.String(500), nullable=True),
        sa.Column('integration_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('recommended_dilution_ratio', sa.Float(), nullable=True, server_default='0.3'),
        sa.Column('min_dilution_ratio', sa.Float(), nullable=True, server_default='0.1'),
        sa.Column('max_dilution_ratio', sa.Float(), nullable=True, server_default='0.5'),
        sa.Column('download_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('integration_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('last_downloaded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_integrated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('license_type', sa.String(100), nullable=True),
        sa.Column('license_url', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_industry_datasets_category', 'industry_datasets', ['category'])
    op.create_index('idx_industry_datasets_status', 'industry_datasets', ['status'])
    op.create_index('idx_industry_datasets_platform', 'industry_datasets', ['source_platform'])
    op.create_index('idx_industry_datasets_tenant_id', 'industry_datasets', ['tenant_id'])

    # ========================================================================
    # Create sync_audit_logs table
    # ========================================================================
    op.create_table('sync_audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('action', postgresql.ENUM('job_created', 'job_updated', 'job_deleted', 'job_started', 'job_stopped', 'job_paused', 'job_resumed', 'sync_started', 'sync_completed', 'sync_failed', 'conflict_detected', 'conflict_resolved', 'data_pushed', 'data_pulled', 'source_connected', 'source_disconnected', name='auditaction', create_type=False), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('source_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('actor_id', sa.String(100), nullable=True),
        sa.Column('actor_type', sa.String(50), nullable=True, server_default="'user'"),
        sa.Column('actor_ip', sa.String(50), nullable=True),
        sa.Column('action_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('state_before', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('state_after', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sync_audit_logs_tenant_action', 'sync_audit_logs', ['tenant_id', 'action'])
    op.create_index('idx_sync_audit_logs_tenant_time', 'sync_audit_logs', ['tenant_id', 'created_at'])
    op.create_index('idx_sync_audit_logs_job', 'sync_audit_logs', ['job_id'])
    op.create_index('idx_sync_audit_logs_tenant_id', 'sync_audit_logs', ['tenant_id'])

    # ========================================================================
    # Create data_quality_scores table
    # ========================================================================
    op.create_table('data_quality_scores',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.String(100), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('batch_id', sa.String(200), nullable=True),
        sa.Column('stage', sa.String(50), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('completeness_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('consistency_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('accuracy_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('relevancy_score', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('noise_ratio', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('duplicate_ratio', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('total_samples', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('valid_samples', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('invalid_samples', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('ai_accuracy_improvement', sa.Float(), nullable=True),
        sa.Column('ai_response_time_improvement', sa.Float(), nullable=True),
        sa.Column('metrics_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='{}'),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_data_quality_scores_tenant_stage', 'data_quality_scores', ['tenant_id', 'stage'])
    op.create_index('idx_data_quality_scores_tenant_id', 'data_quality_scores', ['tenant_id'])


def downgrade():
    # Drop tables in reverse order
    op.drop_table('data_quality_scores')
    op.drop_table('sync_audit_logs')
    op.drop_table('industry_datasets')
    op.drop_table('transformation_rules')
    op.drop_table('sync_rules')
    op.drop_table('data_conflicts')
    op.drop_table('sync_executions')
    op.drop_table('sync_jobs')
    op.drop_table('data_sources')

    # Remove sync-related columns from existing tables
    op.drop_index('idx_tasks_tenant_id', table_name='tasks')
    op.drop_column('tasks', 'sync_metadata')
    op.drop_column('tasks', 'is_from_sync')
    op.drop_column('tasks', 'sync_execution_id')
    op.drop_column('tasks', 'last_synced_at')
    op.drop_column('tasks', 'sync_version')
    op.drop_column('tasks', 'sync_status')
    op.drop_column('tasks', 'tenant_id')

    op.drop_index('idx_documents_tenant_id', table_name='documents')
    op.drop_column('documents', 'sync_metadata')
    op.drop_column('documents', 'is_from_sync')
    op.drop_column('documents', 'sync_job_id')
    op.drop_column('documents', 'sync_source_id')
    op.drop_column('documents', 'last_synced_at')
    op.drop_column('documents', 'sync_hash')
    op.drop_column('documents', 'sync_version')
    op.drop_column('documents', 'sync_status')
    op.drop_column('documents', 'tenant_id')

    # Drop ENUM types
    op.execute('DROP TYPE IF EXISTS syncstatus')
    op.execute('DROP TYPE IF EXISTS auditaction')
    op.execute('DROP TYPE IF EXISTS datasetstatus')
    op.execute('DROP TYPE IF EXISTS datasetcategory')
    op.execute('DROP TYPE IF EXISTS transformationtype')
    op.execute('DROP TYPE IF EXISTS datasourcestatus')
    op.execute('DROP TYPE IF EXISTS datasourcetype')
    op.execute('DROP TYPE IF EXISTS conflictstatus')
    op.execute('DROP TYPE IF EXISTS conflictresolutionstrategy')
    op.execute('DROP TYPE IF EXISTS conflicttype')
    op.execute('DROP TYPE IF EXISTS syncexecutionstatus')
    op.execute('DROP TYPE IF EXISTS syncjobstatus')
    op.execute('DROP TYPE IF EXISTS syncfrequency')
    op.execute('DROP TYPE IF EXISTS syncdirection')
