// React Query hooks for security audit
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { message } from 'antd';
import {
  securityService,
  type AuditLogListParams,
  type SecurityEventListParams,
} from '@/services/security';

const QUERY_KEYS = {
  auditLogs: 'security-audit-logs',
  auditLog: 'security-audit-log',
  events: 'security-events',
  event: 'security-event',
  stats: 'security-stats',
  blockedIPs: 'security-blocked-ips',
  sessions: 'security-sessions',
} as const;

// Audit Logs
export function useAuditLogs(params?: AuditLogListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.auditLogs, params],
    queryFn: () => securityService.getAuditLogs(params),
  });
}

export function useAuditLog(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.auditLog, id],
    queryFn: () => securityService.getAuditLog(id),
    enabled: !!id,
  });
}

export function useExportAuditLogs() {
  return useMutation({
    mutationFn: (params?: AuditLogListParams) => securityService.exportAuditLogs(params),
    onSuccess: (blob) => {
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `audit-logs-${new Date().toISOString().split('T')[0]}.xlsx`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success('Audit logs exported successfully');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to export audit logs');
    },
  });
}

// Security Events
export function useSecurityEvents(params?: SecurityEventListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.events, params],
    queryFn: () => securityService.getSecurityEvents(params),
  });
}

export function useSecurityEvent(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.event, id],
    queryFn: () => securityService.getSecurityEvent(id),
    enabled: !!id,
  });
}

export function useResolveSecurityEvent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => securityService.resolveSecurityEvent(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.events] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.event, id] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Security event resolved');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to resolve security event');
    },
  });
}

// Stats
export function useSecurityStats() {
  return useQuery({
    queryKey: [QUERY_KEYS.stats],
    queryFn: () => securityService.getStats(),
    refetchInterval: 60000, // Refresh every minute
  });
}

// IP Management
export function useBlockedIPs() {
  return useQuery({
    queryKey: [QUERY_KEYS.blockedIPs],
    queryFn: () => securityService.getBlockedIPs(),
  });
}

export function useBlockIP() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ ip, reason }: { ip: string; reason?: string }) =>
      securityService.blockIP(ip, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.blockedIPs] });
      message.success('IP address blocked');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to block IP address');
    },
  });
}

export function useUnblockIP() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ip: string) => securityService.unblockIP(ip),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.blockedIPs] });
      message.success('IP address unblocked');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to unblock IP address');
    },
  });
}

// Session Management
export function useActiveSessions() {
  return useQuery({
    queryKey: [QUERY_KEYS.sessions],
    queryFn: () => securityService.getActiveSessions(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

export function useTerminateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => securityService.terminateSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.sessions] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Session terminated');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to terminate session');
    },
  });
}

export function useTerminateAllUserSessions() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => securityService.terminateAllUserSessions(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.sessions] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('All user sessions terminated');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to terminate user sessions');
    },
  });
}
