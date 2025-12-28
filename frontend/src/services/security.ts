// Security audit service
import { apiClient } from './api/client';
import { API_ENDPOINTS } from '@/constants';

export interface AuditLog {
  id: string;
  user_id: string;
  user_name: string;
  action: string;
  resource: string;
  ip_address: string;
  user_agent: string;
  status: 'success' | 'failed';
  details?: string;
  created_at: string;
}

export interface SecurityEvent {
  id: string;
  type: 'login_attempt' | 'permission_change' | 'data_access' | 'config_change';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  user_name?: string;
  ip_address?: string;
  created_at: string;
  resolved: boolean;
  resolved_at?: string;
  resolved_by?: string;
}

export interface AuditLogListParams {
  page?: number;
  page_size?: number;
  user_id?: string;
  action?: string;
  status?: 'success' | 'failed';
  start_date?: string;
  end_date?: string;
  resource?: string;
}

export interface AuditLogListResponse {
  items: AuditLog[];
  total: number;
  page: number;
  page_size: number;
}

export interface SecurityEventListParams {
  page?: number;
  page_size?: number;
  type?: string;
  severity?: string;
  resolved?: boolean;
  start_date?: string;
  end_date?: string;
}

export interface SecurityEventListResponse {
  items: SecurityEvent[];
  total: number;
  page: number;
  page_size: number;
}

export const securityService = {
  // Audit logs
  async getAuditLogs(params?: AuditLogListParams): Promise<AuditLogListResponse> {
    const response = await apiClient.get<AuditLogListResponse>(
      API_ENDPOINTS.SECURITY?.AUDIT_LOGS || '/api/security/audit-logs',
      { params }
    );
    return response.data;
  },

  async getAuditLog(id: string): Promise<AuditLog> {
    const response = await apiClient.get<AuditLog>(
      `${API_ENDPOINTS.SECURITY?.AUDIT_LOGS || '/api/security/audit-logs'}/${id}`
    );
    return response.data;
  },

  async exportAuditLogs(params?: AuditLogListParams): Promise<Blob> {
    const response = await apiClient.get(
      API_ENDPOINTS.SECURITY?.EXPORT_LOGS || '/api/security/audit-logs/export',
      {
        params,
        responseType: 'blob',
      }
    );
    return response.data;
  },

  // Security events
  async getSecurityEvents(
    params?: SecurityEventListParams
  ): Promise<SecurityEventListResponse> {
    const response = await apiClient.get<SecurityEventListResponse>(
      API_ENDPOINTS.SECURITY?.EVENTS || '/api/security/events',
      { params }
    );
    return response.data;
  },

  async getSecurityEvent(id: string): Promise<SecurityEvent> {
    const response = await apiClient.get<SecurityEvent>(
      `${API_ENDPOINTS.SECURITY?.EVENTS || '/api/security/events'}/${id}`
    );
    return response.data;
  },

  async resolveSecurityEvent(id: string): Promise<SecurityEvent> {
    const response = await apiClient.post<SecurityEvent>(
      `${API_ENDPOINTS.SECURITY?.EVENTS || '/api/security/events'}/${id}/resolve`
    );
    return response.data;
  },

  // Statistics
  async getStats(): Promise<{
    total_logs_today: number;
    failed_attempts: number;
    unresolved_events: number;
    active_users: number;
    suspicious_ips: string[];
  }> {
    const response = await apiClient.get(
      API_ENDPOINTS.SECURITY?.STATS || '/api/security/stats'
    );
    return response.data;
  },

  // IP management
  async getBlockedIPs(): Promise<string[]> {
    const response = await apiClient.get<string[]>(
      API_ENDPOINTS.SECURITY?.BLOCKED_IPS || '/api/security/blocked-ips'
    );
    return response.data;
  },

  async blockIP(ip: string, reason?: string): Promise<void> {
    await apiClient.post(
      API_ENDPOINTS.SECURITY?.BLOCKED_IPS || '/api/security/blocked-ips',
      { ip, reason }
    );
  },

  async unblockIP(ip: string): Promise<void> {
    await apiClient.delete(
      `${API_ENDPOINTS.SECURITY?.BLOCKED_IPS || '/api/security/blocked-ips'}/${encodeURIComponent(ip)}`
    );
  },

  // Session management
  async getActiveSessions(): Promise<
    Array<{
      id: string;
      user_id: string;
      user_name: string;
      ip_address: string;
      user_agent: string;
      created_at: string;
      last_activity: string;
    }>
  > {
    const response = await apiClient.get(
      API_ENDPOINTS.SECURITY?.SESSIONS || '/api/security/sessions'
    );
    return response.data;
  },

  async terminateSession(sessionId: string): Promise<void> {
    await apiClient.delete(
      `${API_ENDPOINTS.SECURITY?.SESSIONS || '/api/security/sessions'}/${sessionId}`
    );
  },

  async terminateAllUserSessions(userId: string): Promise<void> {
    await apiClient.delete(
      `${API_ENDPOINTS.SECURITY?.SESSIONS || '/api/security/sessions'}/user/${userId}`
    );
  },
};
