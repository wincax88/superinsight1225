// Export all constants
export * from './routes';
export * from './api';

// App constants
export const APP_NAME = 'SuperInsight';
export const DEFAULT_LANGUAGE = 'zh';
export const SUPPORTED_LANGUAGES = ['zh', 'en'] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

// Storage keys
export const STORAGE_KEYS = {
  TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER: 'user',
  TENANT: 'tenant',
  THEME: 'theme',
  LANGUAGE: 'language',
} as const;

// Theme
export const THEMES = {
  LIGHT: 'light',
  DARK: 'dark',
} as const;
export type Theme = (typeof THEMES)[keyof typeof THEMES];

// Pagination defaults
export const DEFAULT_PAGE_SIZE = 20;
export const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];
