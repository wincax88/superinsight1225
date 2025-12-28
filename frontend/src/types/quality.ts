// Quality management type definitions

export type QualityRuleType = 'format' | 'content' | 'consistency' | 'custom';

export type QualitySeverity = 'warning' | 'error';

export type QualityIssueStatus = 'open' | 'fixed' | 'ignored';

export interface QualityRule {
  id: string;
  name: string;
  type: QualityRuleType;
  description: string;
  enabled: boolean;
  severity: QualitySeverity;
  violations_count: number;
  last_run?: string;
  config?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface QualityIssue {
  id: string;
  rule_id: string;
  rule_name: string;
  task_id: string;
  task_name: string;
  severity: QualitySeverity;
  description: string;
  status: QualityIssueStatus;
  annotation_id?: string;
  sample_content?: string;
  expected_value?: string;
  actual_value?: string;
  assigned_to?: string;
  assigned_to_name?: string;
  created_at: string;
  updated_at?: string;
  resolved_at?: string;
  resolved_by?: string;
}

export interface QualityStats {
  total_rules: number;
  active_rules: number;
  total_violations: number;
  open_issues: number;
  fixed_issues: number;
  ignored_issues: number;
  quality_score: number;
  violations_by_severity: {
    warning: number;
    error: number;
  };
  violations_by_type: Record<QualityRuleType, number>;
}

export interface CreateQualityRulePayload {
  name: string;
  type: QualityRuleType;
  description?: string;
  severity: QualitySeverity;
  config?: Record<string, unknown>;
}

export interface UpdateQualityRulePayload {
  name?: string;
  description?: string;
  severity?: QualitySeverity;
  enabled?: boolean;
  config?: Record<string, unknown>;
}

export interface QualityCheckResult {
  rule_id: string;
  rule_name: string;
  passed: boolean;
  violations_found: number;
  issues: Array<{
    description: string;
    annotation_id?: string;
    sample_content?: string;
  }>;
}
