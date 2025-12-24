// Token utilities
import { storage } from './storage';
import { STORAGE_KEYS } from '@/constants';

interface TokenPayload {
  sub: string;
  exp: number;
  tenant_id: string;
  [key: string]: unknown;
}

export function getToken(): string | null {
  return storage.get<string>(STORAGE_KEYS.TOKEN);
}

export function setToken(token: string): void {
  storage.set(STORAGE_KEYS.TOKEN, token);
}

export function removeToken(): void {
  storage.remove(STORAGE_KEYS.TOKEN);
}

export function getRefreshToken(): string | null {
  return storage.get<string>(STORAGE_KEYS.REFRESH_TOKEN);
}

export function setRefreshToken(token: string): void {
  storage.set(STORAGE_KEYS.REFRESH_TOKEN, token);
}

export function removeRefreshToken(): void {
  storage.remove(STORAGE_KEYS.REFRESH_TOKEN);
}

export function decodeToken(token: string): TokenPayload | null {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}

export function isTokenExpired(token: string): boolean {
  const payload = decodeToken(token);
  if (!payload || !payload.exp) {
    return true;
  }
  // Check if token expires in less than 5 minutes
  const expirationTime = payload.exp * 1000;
  const bufferTime = 5 * 60 * 1000; // 5 minutes
  return Date.now() > expirationTime - bufferTime;
}

export function clearAuthTokens(): void {
  removeToken();
  removeRefreshToken();
  storage.remove(STORAGE_KEYS.USER);
  storage.remove(STORAGE_KEYS.TENANT);
}
