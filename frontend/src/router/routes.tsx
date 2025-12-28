// Route configuration
import { lazy, Suspense } from 'react';
import type { RouteObject } from 'react-router-dom';
import { Navigate } from 'react-router-dom';
import { MainLayout } from '@/components/Layout/MainLayout';
import { ProtectedRoute } from './ProtectedRoute';
import { Loading } from '@/components/Common/Loading';
import { ROUTES } from '@/constants';

// Lazy load pages
const LoginPage = lazy(() => import('@/pages/Login'));
const RegisterPage = lazy(() => import('@/pages/Register'));
const DashboardPage = lazy(() => import('@/pages/Dashboard'));
const TasksPage = lazy(() => import('@/pages/Tasks'));
const TaskDetailPage = lazy(() => import('@/pages/Tasks/TaskDetail'));
const BillingPage = lazy(() => import('@/pages/Billing'));
const SettingsPage = lazy(() => import('@/pages/Settings'));
const AdminPage = lazy(() => import('@/pages/Admin'));
const AugmentationPage = lazy(() => import('@/pages/Augmentation'));
const QualityPage = lazy(() => import('@/pages/Quality'));
const SecurityPage = lazy(() => import('@/pages/Security'));
const NotFoundPage = lazy(() => import('@/pages/Error/404'));
const ForbiddenPage = lazy(() => import('@/pages/Error/403'));
const ServerErrorPage = lazy(() => import('@/pages/Error/500'));

const withSuspense = (Component: React.ComponentType) => (
  <Suspense fallback={<Loading fullScreen />}>
    <Component />
  </Suspense>
);

export const routes: RouteObject[] = [
  {
    path: ROUTES.LOGIN,
    element: withSuspense(LoginPage),
  },
  {
    path: ROUTES.REGISTER,
    element: withSuspense(RegisterPage),
  },
  {
    path: '/',
    element: (
      <ProtectedRoute>
        <MainLayout />
      </ProtectedRoute>
    ),
    children: [
      {
        index: true,
        element: <Navigate to={ROUTES.DASHBOARD} replace />,
      },
      {
        path: 'dashboard',
        element: withSuspense(DashboardPage),
      },
      {
        path: 'tasks',
        element: withSuspense(TasksPage),
      },
      {
        path: 'tasks/:id',
        element: withSuspense(TaskDetailPage),
      },
      {
        path: 'billing',
        element: withSuspense(BillingPage),
      },
      {
        path: 'settings',
        element: withSuspense(SettingsPage),
      },
      {
        path: 'admin',
        element: withSuspense(AdminPage),
      },
      {
        path: 'augmentation',
        element: withSuspense(AugmentationPage),
      },
      {
        path: 'quality',
        element: withSuspense(QualityPage),
      },
      {
        path: 'security',
        element: withSuspense(SecurityPage),
      },
    ],
  },
  {
    path: ROUTES.FORBIDDEN,
    element: withSuspense(ForbiddenPage),
  },
  {
    path: ROUTES.SERVER_ERROR,
    element: withSuspense(ServerErrorPage),
  },
  {
    path: '*',
    element: withSuspense(NotFoundPage),
  },
];
