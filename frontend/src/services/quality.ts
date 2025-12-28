// Quality management service
import { apiClient } from './api/client';
import { API_ENDPOINTS } from '@/constants';

export interface QualityRule {
  id: string;
  name: string;
  type: 'format' | 'content' | 'consistency' | 'custom';
  description: string;
  enabled: boolean;
  severity: 'warning' | 'error';
  violations_count: number;
  last_run?: string;
  config?: Record<string, unknown>;
}

export interface QualityIssue {
  id: string;
  rule_id: string;
  rule_name: string;
  task_id: string;
  task_name: string;
  severity: 'warning' | 'error';
  description: string;
  status: 'open' | 'fixed' | 'ignored';
  created_at: string;
  updated_at?: string;
  assigned_to?: string;
  annotation_id?: string;
}

export interface CreateRulePayload {
  name: string;
  type: 'format' | 'content' | 'consistency' | 'custom';
  description?: string;
  severity: 'warning' | 'error';
  config?: Record<string, unknown>;
}

export interface UpdateRulePayload {
  name?: string;
  description?: string;
  severity?: 'warning' | 'error';
  enabled?: boolean;
  config?: Record<string, unknown>;
}

export interface RuleListParams {
  page?: number;
  page_size?: number;
  type?: string;
  enabled?: boolean;
}

export interface RuleListResponse {
  items: QualityRule[];
  total: number;
  page: number;
  page_size: number;
}

export interface IssueListParams {
  page?: number;
  page_size?: number;
  status?: 'open' | 'fixed' | 'ignored';
  severity?: 'warning' | 'error';
  rule_id?: string;
  task_id?: string;
}

export interface IssueListResponse {
  items: QualityIssue[];
  total: number;
  page: number;
  page_size: number;
}

export const qualityService = {
  // Rule management
  async getRules(params?: RuleListParams): Promise<RuleListResponse> {
    const response = await apiClient.get<RuleListResponse>(
      API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules',
      { params }
    );
    return response.data;
  },

  async getRule(id: string): Promise<QualityRule> {
    const response = await apiClient.get<QualityRule>(
      `${API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules'}/${id}`
    );
    return response.data;
  },

  async createRule(payload: CreateRulePayload): Promise<QualityRule> {
    const response = await apiClient.post<QualityRule>(
      API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules',
      payload
    );
    return response.data;
  },

  async updateRule(id: string, payload: UpdateRulePayload): Promise<QualityRule> {
    const response = await apiClient.put<QualityRule>(
      `${API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules'}/${id}`,
      payload
    );
    return response.data;
  },

  async deleteRule(id: string): Promise<void> {
    await apiClient.delete(
      `${API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules'}/${id}`
    );
  },

  async toggleRule(id: string, enabled: boolean): Promise<QualityRule> {
    const response = await apiClient.patch<QualityRule>(
      `${API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules'}/${id}/toggle`,
      { enabled }
    );
    return response.data;
  },

  async runRule(id: string): Promise<{ violations_found: number }> {
    const response = await apiClient.post<{ violations_found: number }>(
      `${API_ENDPOINTS.QUALITY?.RULES || '/api/quality/rules'}/${id}/run`
    );
    return response.data;
  },

  async runAllRules(): Promise<{ total_violations: number }> {
    const response = await apiClient.post<{ total_violations: number }>(
      API_ENDPOINTS.QUALITY?.RUN_ALL || '/api/quality/rules/run-all'
    );
    return response.data;
  },

  // Issue management
  async getIssues(params?: IssueListParams): Promise<IssueListResponse> {
    const response = await apiClient.get<IssueListResponse>(
      API_ENDPOINTS.QUALITY?.ISSUES || '/api/quality/issues',
      { params }
    );
    return response.data;
  },

  async getIssue(id: string): Promise<QualityIssue> {
    const response = await apiClient.get<QualityIssue>(
      `${API_ENDPOINTS.QUALITY?.ISSUES || '/api/quality/issues'}/${id}`
    );
    return response.data;
  },

  async updateIssueStatus(
    id: string,
    status: 'open' | 'fixed' | 'ignored'
  ): Promise<QualityIssue> {
    const response = await apiClient.patch<QualityIssue>(
      `${API_ENDPOINTS.QUALITY?.ISSUES || '/api/quality/issues'}/${id}/status`,
      { status }
    );
    return response.data;
  },

  async assignIssue(id: string, userId: string): Promise<QualityIssue> {
    const response = await apiClient.patch<QualityIssue>(
      `${API_ENDPOINTS.QUALITY?.ISSUES || '/api/quality/issues'}/${id}/assign`,
      { user_id: userId }
    );
    return response.data;
  },

  // Statistics
  async getStats(): Promise<{
    total_rules: number;
    active_rules: number;
    total_violations: number;
    open_issues: number;
    quality_score: number;
  }> {
    const response = await apiClient.get(
      API_ENDPOINTS.QUALITY?.STATS || '/api/quality/stats'
    );
    return response.data;
  },
};
