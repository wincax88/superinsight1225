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
const DashboardPage = lazy(() => import('@/pages/Dashboard'));
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
