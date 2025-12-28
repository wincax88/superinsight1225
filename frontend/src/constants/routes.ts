// Route path constants

export const ROUTES = {
  // Public routes
  LOGIN: '/login',
  REGISTER: '/register',

  // Protected routes
  HOME: '/',
  DASHBOARD: '/dashboard',

  // Task management (Phase 2)
  TASKS: '/tasks',
  TASK_DETAIL: '/tasks/:id',
  TASK_CREATE: '/tasks/create',

  // Billing (Phase 2)
  BILLING: '/billing',
  BILLING_DETAIL: '/billing/:id',

  // Settings
  SETTINGS: '/settings',
  PROFILE: '/settings/profile',

  // Admin
  ADMIN: '/admin',
  ADMIN_TENANTS: '/admin/tenants',
  ADMIN_USERS: '/admin/users',

  // Data Augmentation
  AUGMENTATION: '/augmentation',

  // Quality Management
  QUALITY: '/quality',

  // Security Audit
  SECURITY: '/security',

  // Error pages
  NOT_FOUND: '/404',
  FORBIDDEN: '/403',
  SERVER_ERROR: '/500',
} as const;

export type RoutePath = (typeof ROUTES)[keyof typeof ROUTES];
