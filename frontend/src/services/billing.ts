// Billing API service
import apiClient from './api/client';
import { API_ENDPOINTS } from '@/constants';
import type {
  BillingListParams,
  BillingListResponse,
  BillingRecord,
  BillingAnalysis,
  WorkHoursRanking,
  WorkHoursStatistics,
  ProjectCostBreakdown,
  DepartmentCostAllocation,
  BillingRuleVersion,
  EnhancedBillingReport,
  EnhancedReportRequest,
  BillingRuleVersionRequest,
  ExcelExportData,
} from '@/types/billing';

const BILLING_BASE = '/api/billing';

export const billingService = {
  // Get billing records list
  async getList(tenantId: string, params: BillingListParams = {}): Promise<BillingListResponse> {
    const response = await apiClient.get<BillingListResponse>(
      API_ENDPOINTS.BILLING.RECORDS(tenantId),
      { params }
    );
    return response.data;
  },

  // Get single billing record
  async getById(tenantId: string, id: string): Promise<BillingRecord> {
    const response = await apiClient.get<BillingRecord>(
      `${API_ENDPOINTS.BILLING.RECORDS(tenantId)}/${id}`
    );
    return response.data;
  },

  // Get billing analysis
  async getAnalysis(tenantId: string): Promise<BillingAnalysis> {
    const response = await apiClient.get<BillingAnalysis>(
      API_ENDPOINTS.BILLING.ANALYSIS(tenantId)
    );
    return response.data;
  },

  // Get work hours ranking
  async getWorkHoursRanking(tenantId: string): Promise<WorkHoursRanking[]> {
    const response = await apiClient.get<WorkHoursRanking[]>(
      `${API_ENDPOINTS.BILLING.RECORDS(tenantId)}/ranking`
    );
    return response.data;
  },

  // Export billing data
  async exportToExcel(tenantId: string, params: BillingListParams = {}): Promise<Blob> {
    const response = await apiClient.get(
      `${API_ENDPOINTS.BILLING.RECORDS(tenantId)}/export`,
      {
        params,
        responseType: 'blob',
      }
    );
    return response.data;
  },

  // ============================================================================
  // Enhanced Report Service APIs
  // ============================================================================

  // Generate enhanced billing report
  async getEnhancedReport(request: EnhancedReportRequest): Promise<EnhancedBillingReport> {
    const response = await apiClient.post<EnhancedBillingReport>(
      `${BILLING_BASE}/enhanced-report`,
      request
    );
    return response.data;
  },

  // Get work hours statistics
  async getWorkHoursStatistics(
    tenantId: string,
    startDate: string,
    endDate: string
  ): Promise<{ statistics: WorkHoursStatistics[]; user_count: number }> {
    const response = await apiClient.get(
      `${BILLING_BASE}/work-hours/${tenantId}`,
      { params: { start_date: startDate, end_date: endDate } }
    );
    return response.data;
  },

  // Get project cost breakdown
  async getProjectBreakdown(
    tenantId: string,
    startDate: string,
    endDate: string
  ): Promise<{ breakdowns: ProjectCostBreakdown[]; total_cost: number }> {
    const response = await apiClient.get(
      `${BILLING_BASE}/project-breakdown/${tenantId}`,
      { params: { start_date: startDate, end_date: endDate } }
    );
    return response.data;
  },

  // Get department cost allocation
  async getDepartmentAllocation(
    tenantId: string,
    startDate: string,
    endDate: string
  ): Promise<{ allocations: DepartmentCostAllocation[]; total_cost: number }> {
    const response = await apiClient.get(
      `${BILLING_BASE}/department-allocation/${tenantId}`,
      { params: { start_date: startDate, end_date: endDate } }
    );
    return response.data;
  },

  // Create billing rule version
  async createRuleVersion(request: BillingRuleVersionRequest): Promise<{ rule: BillingRuleVersion }> {
    const response = await apiClient.post(
      `${BILLING_BASE}/rules/versions`,
      request
    );
    return response.data;
  },

  // Approve billing rule version
  async approveRuleVersion(
    tenantId: string,
    version: number,
    approvedBy: string
  ): Promise<{ rule: BillingRuleVersion }> {
    const response = await apiClient.post(
      `${BILLING_BASE}/rules/versions/${tenantId}/${version}/approve`,
      { approved_by: approvedBy }
    );
    return response.data;
  },

  // Get billing rule history
  async getRuleHistory(tenantId: string): Promise<{
    active_version: number | null;
    versions: BillingRuleVersion[];
  }> {
    const response = await apiClient.get(
      `${BILLING_BASE}/rules/versions/${tenantId}`
    );
    return response.data;
  },

  // Configure project mappings
  async configureProjectMappings(
    tenantId: string,
    mappings: Record<string, string>
  ): Promise<{ status: string }> {
    const response = await apiClient.post(
      `${BILLING_BASE}/mappings/projects`,
      { tenant_id: tenantId, mappings }
    );
    return response.data;
  },

  // Configure department mappings
  async configureDepartmentMappings(
    tenantId: string,
    projectMappings: Record<string, string>,
    userMappings: Record<string, string>
  ): Promise<{ status: string }> {
    const response = await apiClient.post(
      `${BILLING_BASE}/mappings/departments`,
      {
        tenant_id: tenantId,
        project_mappings: projectMappings,
        user_mappings: userMappings
      }
    );
    return response.data;
  },

  // Export to Excel format
  async getExcelExportData(
    tenantId: string,
    startDate: string,
    endDate: string,
    reportType: string = 'detailed'
  ): Promise<ExcelExportData> {
    const response = await apiClient.get<ExcelExportData>(
      `${BILLING_BASE}/export-excel/${tenantId}`,
      { params: { start_date: startDate, end_date: endDate, report_type: reportType } }
    );
    return response.data;
  },

  // Get cost trends
  async getCostTrends(tenantId: string, days: number = 30): Promise<Record<string, unknown>> {
    const response = await apiClient.get(
      `${BILLING_BASE}/analytics/trends/${tenantId}`,
      { params: { days } }
    );
    return response.data;
  },

  // Get user productivity
  async getUserProductivity(tenantId: string, days: number = 30): Promise<Record<string, unknown>> {
    const response = await apiClient.get(
      `${BILLING_BASE}/analytics/productivity/${tenantId}`,
      { params: { days } }
    );
    return response.data;
  },

  // Get cost forecast
  async getCostForecast(tenantId: string, targetMonth: string): Promise<Record<string, unknown>> {
    const response = await apiClient.get(
      `${BILLING_BASE}/analytics/forecast/${tenantId}/${targetMonth}`
    );
    return response.data;
  },

  // Get optimization recommendations
  async getOptimizationRecommendations(tenantId: string, days: number = 30): Promise<Record<string, unknown>> {
    const response = await apiClient.get(
      `${BILLING_BASE}/analytics/recommendations/${tenantId}`,
      { params: { days } }
    );
    return response.data;
  },
};
