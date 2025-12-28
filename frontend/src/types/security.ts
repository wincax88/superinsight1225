// Security audit type definitions

export type AuditLogStatus = 'success' | 'failed';

export type AuditAction =
  | 'LOGIN'
  | 'LOGOUT'
  | 'CREATE'
  | 'UPDATE'
  | 'DELETE'
  | 'VIEW'
  | 'EXPORT'
  | 'IMPORT'
  | 'ASSIGN'
  | 'APPROVE'
  | 'REJECT';

export type SecurityEventType =
  | 'login_attempt'
  | 'permission_change'
  | 'data_access'
  | 'config_change'
  | 'suspicious_activity'
  | 'rate_limit_exceeded';

export type SecuritySeverity = 'low' | 'medium' | 'high' | 'critical';

export interface AuditLog {
  id: string;
  user_id: string;
  user_name: string;
  action: AuditAction | string;
  resource: string;
  resource_id?: string;
  ip_address: string;
  user_agent: string;
  status: AuditLogStatus;
  details?: string;
  metadata?: Record<string, unknown>;
  tenant_id?: string;
  created_at: string;
}

export interface SecurityEvent {
  id: string;
  type: SecurityEventType;
  severity: SecuritySeverity;
  description: string;
  user_id?: string;
  user_name?: string;
  ip_address?: string;
  user_agent?: string;
  metadata?: Record<string, unknown>;
  resolved: boolean;
  resolved_at?: string;
  resolved_by?: string;
  resolution_notes?: string;
  tenant_id?: string;
  created_at: string;
}

export interface SecurityStats {
  total_logs_today: number;
  total_logs_week: number;
  failed_attempts: number;
  failed_attempts_today: number;
  unresolved_events: number;
  critical_events: number;
  active_users: number;
  active_sessions: number;
  suspicious_ips: string[];
  top_actions: Array<{
    action: string;
    count: number;
  }>;
}

export interface ActiveSession {
  id: string;
  user_id: string;
  user_name: string;
  ip_address: string;
  user_agent: string;
  location?: string;
  device_type?: string;
  created_at: string;
  last_activity: string;
  expires_at?: string;
}

export interface BlockedIP {
  ip: string;
  reason?: string;
  blocked_at: string;
  blocked_by?: string;
  expires_at?: string;
}

export interface IPWhitelistEntry {
  ip: string;
  description?: string;
  added_at: string;
  added_by?: string;
}

export interface AuditLogFilter {
  user_id?: string;
  action?: AuditAction | string;
  status?: AuditLogStatus;
  resource?: string;
  start_date?: string;
  end_date?: string;
  ip_address?: string;
}

export interface SecurityEventFilter {
  type?: SecurityEventType;
  severity?: SecuritySeverity;
  resolved?: boolean;
  start_date?: string;
  end_date?: string;
  user_id?: string;
}
