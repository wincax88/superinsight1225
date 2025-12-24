// Authentication store
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User, Tenant } from '@/types';
import { setToken, clearAuthTokens } from '@/utils/token';

interface AuthState {
  user: User | null;
  token: string | null;
  currentTenant: Tenant | null;
  isAuthenticated: boolean;

  // Actions
  setAuth: (user: User, token: string, tenant?: Tenant) => void;
  setUser: (user: User) => void;
  setTenant: (tenant: Tenant) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      currentTenant: null,
      isAuthenticated: false,

      setAuth: (user, token, tenant) => {
        setToken(token);
        set({
          user,
          token,
          currentTenant: tenant || { id: user.tenant_id, name: user.tenant_id },
          isAuthenticated: true,
        });
      },

      setUser: (user) => {
        set({ user });
      },

      setTenant: (tenant) => {
        set({ currentTenant: tenant });
      },

      clearAuth: () => {
        clearAuthTokens();
        set({
          user: null,
          token: null,
          currentTenant: null,
          isAuthenticated: false,
        });
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        currentTenant: state.currentTenant,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
