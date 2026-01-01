/**
 * useQuality Hooks Tests
 * Task 11.5: Additional module unit tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createElement } from 'react'
import {
  useQualityRules,
  useQualityRule,
  useQualityIssues,
  useQualityIssue,
  useQualityStats,
} from '../useQuality'

// Mock antd message
vi.mock('antd', () => ({
  message: {
    success: vi.fn(),
    error: vi.fn(),
  },
}))

// Mock quality service
vi.mock('@/services/quality', () => ({
  qualityService: {
    getRules: vi.fn().mockResolvedValue({
      items: [
        {
          id: 'rule-1',
          name: 'Length Check',
          type: 'text_length',
          enabled: true,
          severity: 'warning',
        },
        {
          id: 'rule-2',
          name: 'Format Check',
          type: 'format_validation',
          enabled: false,
          severity: 'error',
        },
      ],
      total: 2,
    }),
    getRule: vi.fn().mockResolvedValue({
      id: 'rule-1',
      name: 'Length Check',
      type: 'text_length',
      enabled: true,
      severity: 'warning',
      config: { min_length: 10, max_length: 1000 },
    }),
    getIssues: vi.fn().mockResolvedValue({
      items: [
        {
          id: 'issue-1',
          rule_id: 'rule-1',
          status: 'open',
          severity: 'warning',
          annotation_id: 'ann-1',
        },
        {
          id: 'issue-2',
          rule_id: 'rule-2',
          status: 'fixed',
          severity: 'error',
          annotation_id: 'ann-2',
        },
      ],
      total: 2,
    }),
    getIssue: vi.fn().mockResolvedValue({
      id: 'issue-1',
      rule_id: 'rule-1',
      status: 'open',
      severity: 'warning',
      annotation_id: 'ann-1',
      details: { violation: 'Text too short' },
    }),
    getStats: vi.fn().mockResolvedValue({
      total_rules: 10,
      enabled_rules: 8,
      total_issues: 50,
      open_issues: 20,
      fixed_issues: 25,
      ignored_issues: 5,
      quality_score: 85,
    }),
    createRule: vi.fn().mockResolvedValue({ id: 'rule-new' }),
    updateRule: vi.fn().mockResolvedValue({ id: 'rule-1' }),
    deleteRule: vi.fn().mockResolvedValue(undefined),
    toggleRule: vi.fn().mockResolvedValue(undefined),
    runRule: vi.fn().mockResolvedValue({ violations_found: 5 }),
    runAllRules: vi.fn().mockResolvedValue({ total_violations: 15 }),
    updateIssueStatus: vi.fn().mockResolvedValue(undefined),
    assignIssue: vi.fn().mockResolvedValue(undefined),
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

describe('useQualityRules', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches quality rules successfully', async () => {
    const { result } = renderHook(() => useQualityRules(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      items: [
        {
          id: 'rule-1',
          name: 'Length Check',
          type: 'text_length',
          enabled: true,
          severity: 'warning',
        },
        {
          id: 'rule-2',
          name: 'Format Check',
          type: 'format_validation',
          enabled: false,
          severity: 'error',
        },
      ],
      total: 2,
    })
  })

  it('fetches quality rules with params', async () => {
    const { result } = renderHook(
      () => useQualityRules({ type: 'text_length', enabled: true }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeDefined()
  })
})

describe('useQualityRule', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches single quality rule successfully', async () => {
    const { result } = renderHook(() => useQualityRule('rule-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      id: 'rule-1',
      name: 'Length Check',
      type: 'text_length',
      enabled: true,
      severity: 'warning',
      config: { min_length: 10, max_length: 1000 },
    })
  })

  it('does not fetch when id is empty', () => {
    const { result } = renderHook(() => useQualityRule(''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useQualityIssues', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches quality issues successfully', async () => {
    const { result } = renderHook(() => useQualityIssues(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data?.items).toHaveLength(2)
    expect(result.current.data?.items[0]).toEqual({
      id: 'issue-1',
      rule_id: 'rule-1',
      status: 'open',
      severity: 'warning',
      annotation_id: 'ann-1',
    })
  })

  it('fetches quality issues with filters', async () => {
    const { result } = renderHook(
      () => useQualityIssues({ status: 'open', severity: 'warning' }),
      { wrapper: createWrapper() }
    )

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toBeDefined()
  })
})

describe('useQualityIssue', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches single quality issue successfully', async () => {
    const { result } = renderHook(() => useQualityIssue('issue-1'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      id: 'issue-1',
      rule_id: 'rule-1',
      status: 'open',
      severity: 'warning',
      annotation_id: 'ann-1',
      details: { violation: 'Text too short' },
    })
  })

  it('does not fetch when id is empty', () => {
    const { result } = renderHook(() => useQualityIssue(''), {
      wrapper: createWrapper(),
    })

    expect(result.current.isFetching).toBe(false)
  })
})

describe('useQualityStats', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('fetches quality statistics successfully', async () => {
    const { result } = renderHook(() => useQualityStats(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    expect(result.current.data).toEqual({
      total_rules: 10,
      enabled_rules: 8,
      total_issues: 50,
      open_issues: 20,
      fixed_issues: 25,
      ignored_issues: 5,
      quality_score: 85,
    })
  })

  it('provides correct data structure', async () => {
    const { result } = renderHook(() => useQualityStats(), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    const stats = result.current.data
    expect(stats?.total_rules).toBeTypeOf('number')
    expect(stats?.quality_score).toBeGreaterThanOrEqual(0)
    expect(stats?.quality_score).toBeLessThanOrEqual(100)
  })
})
