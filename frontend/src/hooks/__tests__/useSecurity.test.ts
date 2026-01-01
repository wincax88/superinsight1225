/**
 * useSecurity Hooks Tests
 * Task 11.5: Additional module unit tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import {
  useAuditLogs,
  useAuditLog,
  useSecurityEvents,
  useSecurityEvent,
  useSecurityStats,
  useBlockedIPs,
  useActiveSessions,
} from '../useSecurity'

// Mock antd message
vi.mock('antd', () => ({
  message: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

// Mock security service
vi.mock('@/services/security', () => ({
  securityService: {
    getAuditLogs: vi.fn().mockResolvedValue({
      items: [
        {
          id: 'log-1',
          action: 'user.login',
          user_id: 'user-1',
          ip_address: '192.168.1.1',
          timestamp: '2024-01-01T10:00:00Z',
          status: 'success',
        },
        {
          id: 'log-2',
          action: 'data.export',
          user_id: 'user-2',
          ip_address: '192.168.1.2',
          timestamp: '2024-01-01T11:00:00Z',
          status: 'success',
        },
      ],
      total: 2,
    }),
    getAuditLog: vi.fn().mockResolvedValue({
      id: 'log-1',
      action: 'user.login',
      user_id: 'user-1',
      ip_address: '192.168.1.1',
      timestamp: '2024-01-01T10:00:00Z',
      status: 'success',
      details: { browser: 'Chrome', os: 'Windows' },
    }),
    getSecurityEvents: vi.fn().mockResolvedValue({
      items: [
        {
          id: 'event-1',
          type: 'failed_login',
          severity: 'medium',
          status: 'unresolved',
          ip_address: '10.0.0.1',
          created_at: '2024-01-01T12:00:00Z',
        },
        {
          id: 'event-2',
          type: 'suspicious_activity',
          severity: 'high',
          status: 'resolved',
          ip_address: '10.0.0.2',
          created_at: '2024-01-01T13:00:00Z',
        },
      ],
      total: 2,
    }),
    getSecurityEvent: vi.fn().mockResolvedValue({
      id: 'event-1',
      type: 'failed_login',
      severity: 'medium',
      status: 'unresolved',
      ip_address: '10.0.0.1',
      created_at: '2024-01-01T12:00:00Z',
      details: { attempts: 5, blocked: false },
    }),
    getStats: vi.fn().mockResolvedValue({
      total_events: 100,
      unresolved_events: 15,
      high_severity_events: 5,
      blocked_ips: 10,
      active_sessions: 50,
      events_last_24h: 25,
    }),
    getBlockedIPs: vi.fn().mockResolvedValue({
      items: [
        {
          ip: '10.0.0.100',
          reason: 'Brute force attempt',
          blocked_at: '2024-01-01T00:00:00Z',
          blocked_by: 'system',
        },
        {
          ip: '10.0.0.101',
          reason: 'Suspicious activity',
          blocked_at: '2024-01-01T01:00:00Z',
          blocked_by: 'admin',
        },
      ],
      total: 2,
    }),
    getActiveSessions: vi.fn().mockResolvedValue({
      items: [
        {
          id: 'session-1',
          user_id: 'user-1',
          ip_address: '192.168.1.1',
          user_agent: 'Chrome/120',
          created_at: '2024-01-01T08:00:00Z',
          last_activity: '2024-01-01T10:00:00Z',
        },
        {
          id: 'session-2',
          user_id: 'user-2',
          ip_address: '192.168.1.2',
          user_agent: 'Firefox/121',
          created_at: '2024-01-01T09:00:00Z',
          last_activity: '2024-01-01T10:30:00Z',
        },
      ],
      total: 2,
    }),
    exportAuditLogs: vi.fn().mockResolvedValue(new Blob()),
    resolveSecurityEvent: vi.fn().mockResolvedValue(undefined),
    blockIP: vi.fn().mockResolvedValue(undefined),
    unblockIP: vi.fn().mockResolvedValue(undefined),
    terminateSession: vi.fn().mockResolvedValue(undefined),
    terminateAllUserSessions: vi.fn().mockResolvedValue(undefined),
  },
}))

// Helper to create wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
  return ({ children }: { children: React.ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children)
}

describe('useAuditLogs', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches audit logs successfully', async () => {
    const { result } = renderHook(() => useAuditLogs(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.items).toHaveLength(2)
    expect(result.current.data?.items[0]).toEqual({
      id: 'log-1',
      action: 'user.login',
      user_id: 'user-1',
      ip_address: '192.168.1.1',
      timestamp: '2024-01-01T10:00:00Z',
      status: 'success',
    })
  })

  it('fetches audit logs with filters', async () => {
    const { result } = renderHook(
      () => useAuditLogs({ action: 'user.login', user_id: 'user-1' }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeDefined()
  })
})

describe('useAuditLog', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches single audit log successfully', async () => {
    const { result } = renderHook(() => useAuditLog('log-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      id: 'log-1',
      action: 'user.login',
      user_id: 'user-1',
      ip_address: '192.168.1.1',
      timestamp: '2024-01-01T10:00:00Z',
      status: 'success',
      details: { browser: 'Chrome', os: 'Windows' },
    })
  })

  it('does not fetch when id is empty', () => {
    const { result } = renderHook(() => useAuditLog(''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useSecurityEvents', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches security events successfully', async () => {
    const { result } = renderHook(() => useSecurityEvents(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.items).toHaveLength(2)
    expect(result.current.data?.items[0]).toMatchObject({
      id: 'event-1',
      type: 'failed_login',
      severity: 'medium',
      status: 'unresolved',
    })
  })

  it('fetches security events with severity filter', async () => {
    const { result } = renderHook(
      () => useSecurityEvents({ severity: 'high', status: 'unresolved' }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeDefined()
  })
})

describe('useSecurityEvent', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches single security event successfully', async () => {
    const { result } = renderHook(() => useSecurityEvent('event-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      id: 'event-1',
      type: 'failed_login',
      severity: 'medium',
      status: 'unresolved',
      ip_address: '10.0.0.1',
      created_at: '2024-01-01T12:00:00Z',
      details: { attempts: 5, blocked: false },
    })
  })

  it('does not fetch when id is empty', () => {
    const { result } = renderHook(() => useSecurityEvent(''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useSecurityStats', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches security statistics successfully', async () => {
    const { result } = renderHook(() => useSecurityStats(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      total_events: 100,
      unresolved_events: 15,
      high_severity_events: 5,
      blocked_ips: 10,
      active_sessions: 50,
      events_last_24h: 25,
    })
  })

  it('provides correct data types', async () => {
    const { result } = renderHook(() => useSecurityStats(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    const stats = result.current.data
    expect(stats?.total_events).toBeTypeOf('number')
    expect(stats?.unresolved_events).toBeTypeOf('number')
    expect(stats?.blocked_ips).toBeTypeOf('number')
  })
})

describe('useBlockedIPs', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches blocked IPs successfully', async () => {
    const { result } = renderHook(() => useBlockedIPs(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.items).toHaveLength(2)
    expect(result.current.data?.items[0]).toEqual({
      ip: '10.0.0.100',
      reason: 'Brute force attempt',
      blocked_at: '2024-01-01T00:00:00Z',
      blocked_by: 'system',
    })
  })

  it('includes block reason in data', async () => {
    const { result } = renderHook(() => useBlockedIPs(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    result.current.data?.items.forEach((item) => {
      expect(item.reason).toBeDefined()
      expect(item.blocked_by).toBeDefined()
    })
  })
})

describe('useActiveSessions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches active sessions successfully', async () => {
    const { result } = renderHook(() => useActiveSessions(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.items).toHaveLength(2)
    expect(result.current.data?.items[0]).toMatchObject({
      id: 'session-1',
      user_id: 'user-1',
      ip_address: '192.168.1.1',
    })
  })

  it('includes session activity information', async () => {
    const { result } = renderHook(() => useActiveSessions(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    result.current.data?.items.forEach((session) => {
      expect(session.created_at).toBeDefined()
      expect(session.last_activity).toBeDefined()
      expect(session.user_agent).toBeDefined()
    })
  })
})
