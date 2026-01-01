/**
 * BillingReports Component Tests
 * Task 11.5: Additional module unit tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@/test/test-utils'
import userEvent from '@testing-library/user-event'
import { BillingReports } from '../BillingReports'

// Mock billing hooks
const mockGenerateReport = vi.fn()
vi.mock('@/hooks/useBilling', () => ({
  useGenerateReport: () => ({
    mutateAsync: mockGenerateReport,
    isPending: false,
  }),
  useProjectBreakdown: () => ({
    data: {
      breakdowns: [
        {
          project_id: 'proj-1',
          project_name: 'Test Project 1',
          total_cost: 5000,
          total_annotations: 1000,
          total_time_spent: 50,
          avg_cost_per_annotation: 5,
          percentage_of_total: 50,
        },
        {
          project_id: 'proj-2',
          project_name: 'Test Project 2',
          total_cost: 3000,
          total_annotations: 600,
          total_time_spent: 30,
          avg_cost_per_annotation: 5,
          percentage_of_total: 30,
        },
      ],
    },
    isLoading: false,
  }),
  useDepartmentAllocation: () => ({
    data: {
      allocations: [
        {
          department_id: 'dept-1',
          department_name: 'Engineering',
          total_cost: 4000,
          user_count: 10,
          projects: ['Project A', 'Project B'],
          percentage_of_total: 40,
        },
      ],
    },
    isLoading: false,
  }),
  useCostTrends: () => ({
    data: {
      total_cost: 10000,
      average_daily_cost: 333.33,
      trend_percentage: 5.5,
      daily_costs: [
        { date: '2024-01-01', cost: 300, annotations: 60 },
        { date: '2024-01-02', cost: 350, annotations: 70 },
        { date: '2024-01-03', cost: 320, annotations: 64 },
      ],
    },
    isLoading: false,
  }),
  useWorkHoursStatistics: () => ({
    data: {
      user_count: 5,
      statistics: [
        {
          user_id: 'user-1',
          user_name: 'Alice',
          total_hours: 40,
          billable_hours: 38,
          total_annotations: 800,
          annotations_per_hour: 20,
          total_cost: 2000,
          efficiency_score: 85,
        },
        {
          user_id: 'user-2',
          user_name: 'Bob',
          total_hours: 35,
          billable_hours: 32,
          total_annotations: 600,
          annotations_per_hour: 17.14,
          total_cost: 1500,
          efficiency_score: 75,
        },
      ],
    },
    isLoading: false,
  }),
}))

// Mock recharts to avoid rendering issues in tests
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  ComposedChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="composed-chart">{children}</div>
  ),
  PieChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="pie-chart">{children}</div>
  ),
  BarChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="bar-chart">{children}</div>
  ),
  Area: () => <div data-testid="area" />,
  Line: () => <div data-testid="line" />,
  Bar: () => <div data-testid="bar" />,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}))

// Mock dayjs
vi.mock('dayjs', async () => {
  const actual = await vi.importActual('dayjs')
  const dayjs = actual.default as typeof import('dayjs').default
  return {
    default: dayjs,
    ...actual,
  }
})

describe('BillingReports', () => {
  const defaultProps = {
    tenantId: 'tenant-123',
    currentUserId: 'user-123',
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders billing reports with all sections', () => {
    render(<BillingReports {...defaultProps} />)

    // Check for main controls
    expect(screen.getByText('日期范围:')).toBeInTheDocument()
    expect(screen.getByText('报表类型:')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /生成报表/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /导出报表/i })).toBeInTheDocument()
  })

  it('displays summary statistics correctly', () => {
    render(<BillingReports {...defaultProps} />)

    // Check for statistics cards
    expect(screen.getByText('总成本')).toBeInTheDocument()
    expect(screen.getByText('日均成本')).toBeInTheDocument()
    expect(screen.getByText('项目数量')).toBeInTheDocument()
    expect(screen.getByText('成本趋势')).toBeInTheDocument()
  })

  it('shows all tabs', () => {
    render(<BillingReports {...defaultProps} />)

    // Check for tab labels
    expect(screen.getByText('成本趋势')).toBeInTheDocument()
    expect(screen.getByText('项目分析')).toBeInTheDocument()
    expect(screen.getByText('部门分析')).toBeInTheDocument()
    expect(screen.getByText('工时统计')).toBeInTheDocument()
    expect(screen.getByText('报表详情')).toBeInTheDocument()
  })

  it('can switch between tabs', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    // Click on project analysis tab
    await user.click(screen.getByText('项目分析'))

    // Check for project-related content
    await waitFor(() => {
      expect(screen.getByText('项目成本分布')).toBeInTheDocument()
      expect(screen.getByText('项目成本明细')).toBeInTheDocument()
    })
  })

  it('can switch to department analysis tab', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('部门分析'))

    await waitFor(() => {
      expect(screen.getByText('部门成本分布')).toBeInTheDocument()
      expect(screen.getByText('部门成本明细')).toBeInTheDocument()
    })
  })

  it('can switch to work hours statistics tab', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('工时统计'))

    await waitFor(() => {
      expect(screen.getByText('统计人数')).toBeInTheDocument()
      expect(screen.getByText('总工时')).toBeInTheDocument()
    })
  })

  it('displays project breakdown data in table', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('项目分析'))

    await waitFor(() => {
      expect(screen.getByText('Test Project 1')).toBeInTheDocument()
      expect(screen.getByText('Test Project 2')).toBeInTheDocument()
    })
  })

  it('calls generate report when button is clicked', async () => {
    const user = userEvent.setup()
    mockGenerateReport.mockResolvedValueOnce({
      id: 'report-1',
      report_type: 'summary',
      start_date: '2024-01-01',
      end_date: '2024-01-31',
      generated_at: '2024-01-31T12:00:00Z',
      total_cost: 10000,
      total_annotations: 2000,
      total_time_spent: 100,
      user_breakdown: {},
    })

    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByRole('button', { name: /生成报表/i }))

    await waitFor(() => {
      expect(mockGenerateReport).toHaveBeenCalledWith(
        expect.objectContaining({
          tenant_id: 'tenant-123',
          report_type: 'summary',
        })
      )
    })
  })

  it('export button is disabled when no report is generated', () => {
    render(<BillingReports {...defaultProps} />)

    const exportButton = screen.getByRole('button', { name: /导出报表/i })
    expect(exportButton).toBeDisabled()
  })

  it('shows empty state for report details before generation', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('报表详情'))

    await waitFor(() => {
      expect(screen.getByText('请先生成报表查看详情')).toBeInTheDocument()
    })
  })

  it('displays work hours statistics with user data', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('工时统计'))

    await waitFor(() => {
      expect(screen.getByText('Alice')).toBeInTheDocument()
      expect(screen.getByText('Bob')).toBeInTheDocument()
    })
  })

  it('shows department data correctly', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    await user.click(screen.getByText('部门分析'))

    await waitFor(() => {
      expect(screen.getByText('Engineering')).toBeInTheDocument()
    })
  })

  it('renders report type selector with all options', async () => {
    const user = userEvent.setup()
    render(<BillingReports {...defaultProps} />)

    // Find and click the select to open dropdown
    const select = screen.getByText('概览报表')
    await user.click(select)

    // Check for options in dropdown
    await waitFor(() => {
      expect(screen.getByText('详细报表')).toBeInTheDocument()
      expect(screen.getByText('用户分析')).toBeInTheDocument()
      expect(screen.getByText('项目分析')).toBeInTheDocument()
    })
  })
})
