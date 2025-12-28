// Authentication service
import apiClient from './api/client';
import { API_ENDPOINTS } from '@/constants';
import type { LoginCredentials, LoginResponse, User, Tenant } from '@/types';

export interface RegisterPayload {
  username: string;
  email: string;
  password: string;
  tenant_name?: string;
  invite_code?: string;
}

export const authService = {
  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    const response = await apiClient.post<LoginResponse>(API_ENDPOINTS.AUTH.LOGIN, credentials);
    return response.data;
  },

  async register(payload: RegisterPayload): Promise<void> {
    await apiClient.post(API_ENDPOINTS.AUTH.REGISTER, payload);
  },

  async logout(): Promise<void> {
    await apiClient.post(API_ENDPOINTS.AUTH.LOGOUT);
  },

  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>(API_ENDPOINTS.AUTH.CURRENT_USER);
    return response.data;
  },

  async getTenants(): Promise<Tenant[]> {
    const response = await apiClient.get<Tenant[]>(API_ENDPOINTS.ADMIN.TENANTS);
    return response.data;
  },

  async refreshToken(refreshToken: string): Promise<LoginResponse> {
    const response = await apiClient.post<LoginResponse>(API_ENDPOINTS.AUTH.REFRESH, {
      refresh_token: refreshToken,
    });
    return response.data;
  },
};
