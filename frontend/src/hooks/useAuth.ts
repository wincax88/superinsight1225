// Authentication hook
import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import { useAuthStore } from '@/stores/authStore';
import { authService } from '@/services/auth';
import { ROUTES } from '@/constants';
import type { LoginCredentials } from '@/types';

export function useAuth() {
  const navigate = useNavigate();
  const { user, token, currentTenant, isAuthenticated, setAuth, clearAuth } = useAuthStore();

  const login = useCallback(
    async (credentials: LoginCredentials) => {
      try {
        const response = await authService.login(credentials);
        const userInfo = await authService.getCurrentUser();

        setAuth(userInfo, response.access_token, {
          id: response.tenant_id,
          name: response.tenant_id,
        });

        message.success('登录成功');
        navigate(ROUTES.DASHBOARD);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : '登录失败，请检查用户名和密码';
        message.error(errorMessage);
        throw error;
      }
    },
    [navigate, setAuth]
  );

  const logout = useCallback(async () => {
    try {
      await authService.logout();
    } catch {
      // Ignore logout errors
    } finally {
      clearAuth();
      navigate(ROUTES.LOGIN);
      message.success('已退出登录');
    }
  }, [navigate, clearAuth]);

  const checkAuth = useCallback(async () => {
    if (!token) {
      return false;
    }
    try {
      const userInfo = await authService.getCurrentUser();
      setAuth(userInfo, token, currentTenant || undefined);
      return true;
    } catch {
      clearAuth();
      return false;
    }
  }, [token, currentTenant, setAuth, clearAuth]);

  return {
    user,
    token,
    currentTenant,
    isAuthenticated,
    login,
    logout,
    checkAuth,
  };
}
