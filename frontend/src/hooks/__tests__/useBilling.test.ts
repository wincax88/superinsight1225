/**
 * useBilling Hooks Tests
 * Task 11.5: Additional module unit tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import {
  billingKeys,
  useBillingList,
  useBillingDetail,
  useBillingAnalysis,
  useWorkHoursRanking,
  useWorkHoursStatistics,
  useProjectBreakdown,
  useDepartmentAllocation,
  useCostTrends,
  useBillingDashboard,
} from '../useBilling'

// Mock billing service
vi.mock('@/services/billing', () => ({
  billingService: {
    getList: vi.fn().mockResolvedValue({
      items: [
        { id: 'bill-1', amount: 1000, status: 'paid' },
        { id: 'bill-2', amount: 2000, status: 'pending' },
      ],
      total: 2,
    }),
    getById: vi.fn().mockResolvedValue({
      id: 'bill-1',
      amount: 1000,
      status: 'paid',
      created_at: '2024-01-01',
    }),
    getAnalysis: vi.fn().mockResolvedValue({
      total_cost: 50000,
      total_annotations: 10000,
      average_cost_per_annotation: 5,
    }),
    getWorkHoursRanking: vi.fn().mockResolvedValue({
      rankings: [
        { user_id: 'user-1', hours: 160, rank: 1 },
        { user_id: 'user-2', hours: 140, rank: 2 },
      ],
    }),
    getWorkHoursStatistics: vi.fn().mockResolvedValue({
      user_count: 10,
      statistics: [
        { user_id: 'user-1', total_hours: 40, total_cost: 2000 },
      ],
    }),
    getProjectBreakdown: vi.fn().mockResolvedValue({
      breakdowns: [
        { project_id: 'proj-1', project_name: 'Project A', total_cost: 5000 },
      ],
    }),
    getDepartmentAllocation: vi.fn().mockResolvedValue({
      allocations: [
        { department_id: 'dept-1', department_name: 'Engineering', total_cost: 4000 },
      ],
    }),
    getCostTrends: vi.fn().mockResolvedValue({
      total_cost: 10000,
      average_daily_cost: 333,
      daily_costs: [
        { date: '2024-01-01', cost: 300 },
        { date: '2024-01-02', cost: 350 },
      ],
    }),
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

describe('billingKeys', () => {
  it('generates correct query keys for all', () => {
    expect(billingKeys.all).toEqual(['billing'])
  })

  it('generates correct query keys for lists', () => {
    expect(billingKeys.lists()).toEqual(['billing', 'list'])
  })

  it('generates correct query keys for specific list', () => {
    expect(billingKeys.list('tenant-1', { page: 1 })).toEqual([
      'billing',
      'list',
      'tenant-1',
      { page: 1 },
    ])
  })

  it('generates correct query keys for detail', () => {
    expect(billingKeys.detail('tenant-1', 'bill-1')).toEqual([
      'billing',
      'detail',
      'tenant-1',
      'bill-1',
    ])
  })

  it('generates correct query keys for analysis', () => {
    expect(billingKeys.analysis('tenant-1')).toEqual(['billing', 'analysis', 'tenant-1'])
  })

  it('generates correct query keys for project breakdown', () => {
    expect(billingKeys.projectBreakdown('tenant-1', '2024-01-01', '2024-01-31')).toEqual([
      'billing',
      'projectBreakdown',
      'tenant-1',
      '2024-01-01',
      '2024-01-31',
    ])
  })

  it('generates correct query keys for cost trends', () => {
    expect(billingKeys.costTrends('tenant-1', 30)).toEqual([
      'billing',
      'costTrends',
      'tenant-1',
      30,
    ])
  })
})

describe('useBillingList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches billing list successfully', async () => {
    const { result } = renderHook(() => useBillingList('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      items: [
        { id: 'bill-1', amount: 1000, status: 'paid' },
        { id: 'bill-2', amount: 2000, status: 'pending' },
      ],
      total: 2,
    })
  })

  it('does not fetch when tenantId is empty', () => {
    const { result } = renderHook(() => useBillingList(''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useBillingDetail', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches billing detail successfully', async () => {
    const { result } = renderHook(() => useBillingDetail('tenant-1', 'bill-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      id: 'bill-1',
      amount: 1000,
      status: 'paid',
      created_at: '2024-01-01',
    })
  })

  it('does not fetch when id is empty', () => {
    const { result } = renderHook(() => useBillingDetail('tenant-1', ''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useBillingAnalysis', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches billing analysis successfully', async () => {
    const { result } = renderHook(() => useBillingAnalysis('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      total_cost: 50000,
      total_annotations: 10000,
      average_cost_per_annotation: 5,
    })
  })
})

describe('useWorkHoursRanking', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches work hours ranking successfully', async () => {
    const { result } = renderHook(() => useWorkHoursRanking('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      rankings: [
        { user_id: 'user-1', hours: 160, rank: 1 },
        { user_id: 'user-2', hours: 140, rank: 2 },
      ],
    })
  })
})

describe('useWorkHoursStatistics', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches work hours statistics successfully', async () => {
    const { result } = renderHook(
      () => useWorkHoursStatistics('tenant-1', '2024-01-01', '2024-01-31'),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      user_count: 10,
      statistics: [{ user_id: 'user-1', total_hours: 40, total_cost: 2000 }],
    })
  })

  it('does not fetch when dates are missing', () => {
    const { result } = renderHook(() => useWorkHoursStatistics('tenant-1', '', ''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useProjectBreakdown', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches project breakdown successfully', async () => {
    const { result } = renderHook(
      () => useProjectBreakdown('tenant-1', '2024-01-01', '2024-01-31'),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      breakdowns: [
        { project_id: 'proj-1', project_name: 'Project A', total_cost: 5000 },
      ],
    })
  })
})

describe('useDepartmentAllocation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches department allocation successfully', async () => {
    const { result } = renderHook(
      () => useDepartmentAllocation('tenant-1', '2024-01-01', '2024-01-31'),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      allocations: [
        { department_id: 'dept-1', department_name: 'Engineering', total_cost: 4000 },
      ],
    })
  })
})

describe('useCostTrends', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches cost trends successfully', async () => {
    const { result } = renderHook(() => useCostTrends('tenant-1', 30), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      total_cost: 10000,
      average_daily_cost: 333,
      daily_costs: [
        { date: '2024-01-01', cost: 300 },
        { date: '2024-01-02', cost: 350 },
      ],
    })
  })

  it('uses default days value of 30', async () => {
    const { result } = renderHook(() => useCostTrends('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeDefined()
  })
})

describe('useBillingDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('combines multiple queries for dashboard', async () => {
    const { result } = renderHook(() => useBillingDashboard('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.records).toBeDefined()
    expect(result.current.analysis).toBeDefined()
    expect(result.current.ranking).toBeDefined()
    expect(result.current.trends).toBeDefined()
  })

  it('provides refetch function', async () => {
    const { result } = renderHook(() => useBillingDashboard('tenant-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(typeof result.current.refetch).toBe('function')
  })
})
