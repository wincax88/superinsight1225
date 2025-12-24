// UI state store
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { THEMES, DEFAULT_LANGUAGE, type Theme, type SupportedLanguage } from '@/constants';

interface UIState {
  theme: Theme;
  language: SupportedLanguage;
  sidebarCollapsed: boolean;
  loading: boolean;

  // Actions
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  setLanguage: (language: SupportedLanguage) => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleSidebar: () => void;
  setLoading: (loading: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      theme: THEMES.LIGHT,
      language: DEFAULT_LANGUAGE,
      sidebarCollapsed: false,
      loading: false,

      setTheme: (theme) => {
        set({ theme });
        document.documentElement.setAttribute('data-theme', theme);
      },

      toggleTheme: () => {
        const newTheme = get().theme === THEMES.LIGHT ? THEMES.DARK : THEMES.LIGHT;
        set({ theme: newTheme });
        document.documentElement.setAttribute('data-theme', newTheme);
      },

      setLanguage: (language) => {
        set({ language });
      },

      setSidebarCollapsed: (collapsed) => {
        set({ sidebarCollapsed: collapsed });
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
      },

      setLoading: (loading) => {
        set({ loading });
      },
    }),
    {
      name: 'ui-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        theme: state.theme,
        language: state.language,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);
