// Dashboard data hook
import { useQuery } from '@tanstack/react-query';
import { dashboardService } from '@/services/dashboard';

export function useDashboard() {
  const summaryQuery = useQuery({
    queryKey: ['dashboard', 'summary'],
    queryFn: () => dashboardService.getSummary(),
    refetchInterval: 60000, // Refresh every minute
    staleTime: 30000, // Data is fresh for 30 seconds
  });

  const annotationEfficiencyQuery = useQuery({
    queryKey: ['dashboard', 'annotation-efficiency'],
    queryFn: () => dashboardService.getAnnotationEfficiency(24),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  const userActivityQuery = useQuery({
    queryKey: ['dashboard', 'user-activity'],
    queryFn: () => dashboardService.getUserActivity(24),
    refetchInterval: 60000,
    staleTime: 30000,
  });

  return {
    summary: summaryQuery.data,
    annotationEfficiency: annotationEfficiencyQuery.data,
    userActivity: userActivityQuery.data,
    isLoading: summaryQuery.isLoading || annotationEfficiencyQuery.isLoading || userActivityQuery.isLoading,
    error: summaryQuery.error || annotationEfficiencyQuery.error || userActivityQuery.error,
    refetch: () => {
      summaryQuery.refetch();
      annotationEfficiencyQuery.refetch();
      userActivityQuery.refetch();
    },
  };
}
