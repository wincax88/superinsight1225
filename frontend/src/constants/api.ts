// API endpoint constants

export const API_ENDPOINTS = {
  // Authentication
  AUTH: {
    LOGIN: '/api/security/login',
    LOGOUT: '/api/security/logout',
    CURRENT_USER: '/api/security/users/me',
    REFRESH: '/api/security/refresh',
  },

  // Users
  USERS: {
    BASE: '/api/security/users',
    BY_ID: (id: string) => `/api/security/users/${id}`,
    ROLE: (id: string) => `/api/security/users/${id}/role`,
  },

  // Business metrics
  METRICS: {
    SUMMARY: '/api/business-metrics/summary',
    ANNOTATION_EFFICIENCY: '/api/business-metrics/annotation-efficiency',
    USER_ACTIVITY: '/api/business-metrics/user-activity',
    AI_MODELS: '/api/business-metrics/ai-models',
    PROJECTS: '/api/business-metrics/projects',
  },

  // Admin
  ADMIN: {
    TENANTS: '/api/admin/tenants',
    TENANT_BY_ID: (id: string) => `/api/admin/tenants/${id}`,
  },

  // Billing
  BILLING: {
    RECORDS: (tenantId: string) => `/api/billing/records/${tenantId}`,
    ANALYSIS: (tenantId: string) => `/api/billing/analysis/${tenantId}`,
    TRENDS: (tenantId: string) => `/api/billing/analytics/trends/${tenantId}`,
  },

  // Quality
  QUALITY: {
    DASHBOARD: '/api/quality/dashboard/summary',
    RULES: '/api/quality/rules',
  },

  // Security / Audit
  SECURITY: {
    AUDIT_LOGS: '/api/security/audit-logs',
    AUDIT_SUMMARY: '/api/security/audit/summary',
    PERMISSIONS: '/api/security/permissions',
    IP_WHITELIST: '/api/security/ip-whitelist',
  },

  // System
  SYSTEM: {
    HEALTH: '/health',
    STATUS: '/system/status',
    METRICS: '/system/metrics',
    SERVICES: '/system/services',
  },
} as const;
