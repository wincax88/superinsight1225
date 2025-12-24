// Permission guard component
import type { ReactNode } from 'react';
import { useAuthStore } from '@/stores/authStore';

interface PermissionGuardProps {
  permissions?: string[];
  roles?: string[];
  children: ReactNode;
  fallback?: ReactNode;
}

export const PermissionGuard: React.FC<PermissionGuardProps> = ({
  permissions = [],
  roles = [],
  children,
  fallback = null,
}) => {
  const { user } = useAuthStore();

  if (!user) {
    return <>{fallback}</>;
  }

  // Check roles
  if (roles.length > 0 && !roles.includes(user.role)) {
    return <>{fallback}</>;
  }

  // For now, we just check roles since permissions are not fully implemented
  // In a full implementation, you would check against user.permissions
  if (permissions.length > 0) {
    // Placeholder: admin has all permissions
    if (user.role !== 'admin') {
      return <>{fallback}</>;
    }
  }

  return <>{children}</>;
};
