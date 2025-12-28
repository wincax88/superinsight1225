// API endpoint constants

export const API_ENDPOINTS = {
  // Authentication
  AUTH: {
    LOGIN: '/api/security/login',
    LOGOUT: '/api/security/logout',
    REGISTER: '/api/security/register',
    CURRENT_USER: '/api/security/users/me',
    REFRESH: '/api/security/refresh',
  },

  // Label Studio
  LABEL_STUDIO: {
    PROJECTS: '/api/label-studio/projects',
    PROJECT_BY_ID: (id: string) => `/api/label-studio/projects/${id}`,
    TASKS: (projectId: string) => `/api/label-studio/projects/${projectId}/tasks`,
    ANNOTATIONS: (projectId: string, taskId: string) =>
      `/api/label-studio/projects/${projectId}/tasks/${taskId}/annotations`,
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
    ISSUES: '/api/quality/issues',
    RUN_ALL: '/api/quality/rules/run-all',
    STATS: '/api/quality/stats',
  },

  // Security / Audit
  SECURITY: {
    AUDIT_LOGS: '/api/security/audit-logs',
    AUDIT_SUMMARY: '/api/security/audit/summary',
    PERMISSIONS: '/api/security/permissions',
    IP_WHITELIST: '/api/security/ip-whitelist',
    EVENTS: '/api/security/events',
    EXPORT_LOGS: '/api/security/audit-logs/export',
    BLOCKED_IPS: '/api/security/blocked-ips',
    SESSIONS: '/api/security/sessions',
    STATS: '/api/security/stats',
  },

  // Data Augmentation
  AUGMENTATION: {
    JOBS: '/api/augmentation/jobs',
    SAMPLES: '/api/augmentation/samples',
    UPLOAD: '/api/augmentation/upload',
    STATS: '/api/augmentation/stats',
  },

  // Tasks
  TASKS: {
    BASE: '/api/tasks',
    BY_ID: (id: string) => `/api/tasks/${id}`,
    STATS: '/api/tasks/stats',
    ASSIGN: (id: string) => `/api/tasks/${id}/assign`,
    BATCH: '/api/tasks/batch',
  },

  // System
  SYSTEM: {
    HEALTH: '/health',
    STATUS: '/system/status',
    METRICS: '/system/metrics',
    SERVICES: '/system/services',
  },
} as const;
