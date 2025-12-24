// Authentication types

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_id: string;
  username: string;
  role: string;
  tenant_id: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  full_name: string;
  tenant_id?: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  user: User | null;
  currentTenant: Tenant | null;
}

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  role: UserRole;
  tenant_id: string;
  is_active: boolean;
  last_login?: string;
  created_at: string;
}

export type UserRole = 'admin' | 'manager' | 'annotator' | 'viewer';

export interface Tenant {
  id: string;
  name: string;
  logo?: string;
}

export interface Permission {
  id: string;
  user_id: string;
  project_id: string;
  permission_type: PermissionType;
}

export type PermissionType = 'read' | 'write' | 'admin' | 'quality_control';
