// User-related types

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  avatar?: string;
  role: string;
  tenant_id: string;
  is_active: boolean;
  last_login?: string;
  created_at: string;
  updated_at?: string;
}

export interface UserActivity {
  total_actions: number;
  actions_by_type: Record<string, number>;
  resources_accessed: Record<string, number>;
  daily_activity: Record<string, number>;
  suspicious_patterns: unknown[];
  analysis_period_days: number;
}

export interface UserSession {
  id: string;
  user_id: string;
  started_at: string;
  last_activity?: string;
  ip_address?: string;
  user_agent?: string;
}
