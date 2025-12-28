// Billing type definitions

export type BillingStatus = 'pending' | 'paid' | 'overdue' | 'cancelled';

export interface BillingRecord {
  id: string;
  tenant_id: string;
  period_start: string;
  period_end: string;
  total_amount: number;
  status: BillingStatus;
  items: BillingItem[];
  created_at: string;
  due_date: string;
  paid_at?: string;
}

export interface BillingItem {
  id: string;
  description: string;
  quantity: number;
  unit_price: number;
  amount: number;
  category: 'annotation' | 'augmentation' | 'storage' | 'api' | 'other';
}

export interface BillingAnalysis {
  total_spending: number;
  average_monthly: number;
  trend_percentage: number;
  by_category: {
    category: string;
    amount: number;
    percentage: number;
  }[];
  monthly_trends: {
    month: string;
    amount: number;
  }[];
}

export interface WorkHoursRanking {
  user_id: string;
  user_name: string;
  total_hours: number;
  annotations_count: number;
  efficiency_score: number;
  rank: number;
}

export interface BillingListParams {
  page?: number;
  page_size?: number;
  status?: BillingStatus;
  start_date?: string;
  end_date?: string;
}

export interface BillingListResponse {
  items: BillingRecord[];
  total: number;
  page: number;
  page_size: number;
}

// Enhanced billing types for new report service

export type BillingMode = 'by_count' | 'by_time' | 'by_project' | 'hybrid';
export type ReportType = 'summary' | 'detailed' | 'user_breakdown' | 'project_breakdown' | 'department_breakdown' | 'work_hours' | 'trend_analysis';

export interface WorkHoursStatistics {
  user_id: string;
  user_name?: string;
  total_hours: number;
  billable_hours: number;
  total_annotations: number;
  annotations_per_hour: number;
  total_cost: number;
  efficiency_score: number;
}

export interface ProjectCostBreakdown {
  project_id: string;
  project_name: string;
  total_cost: number;
  total_annotations: number;
  total_time_spent: number;
  user_count: number;
  avg_cost_per_annotation: number;
  percentage_of_total: number;
}

export interface DepartmentCostAllocation {
  department_id: string;
  department_name: string;
  total_cost: number;
  projects: string[];
  user_count: number;
  percentage_of_total: number;
}

export interface BillingRuleVersion {
  id: string;
  tenant_id: string;
  version: number;
  billing_mode: BillingMode;
  rate_per_annotation: number;
  rate_per_hour: number;
  project_annual_fee: number;
  effective_from: string;
  effective_to?: string;
  created_by: string;
  approved_by?: string;
  approved_at?: string;
  is_active: boolean;
}

export interface EnhancedBillingReport {
  id: string;
  tenant_id: string;
  report_type: ReportType;
  start_date: string;
  end_date: string;
  total_cost: number;
  total_annotations: number;
  total_time_spent: number;
  user_breakdown: Record<string, {
    annotations: number;
    time_spent: number;
    cost: number;
  }>;
  project_breakdown: ProjectCostBreakdown[];
  department_breakdown: DepartmentCostAllocation[];
  work_hours_statistics: WorkHoursStatistics[];
  generated_at: string;
  generated_by: string;
  billing_rule_version?: number;
}

export interface EnhancedReportRequest {
  tenant_id: string;
  start_date: string;
  end_date: string;
  report_type?: ReportType;
  user_names?: Record<string, string>;
  project_names?: Record<string, string>;
  department_names?: Record<string, string>;
}

export interface BillingRuleVersionRequest {
  tenant_id: string;
  billing_mode?: BillingMode;
  rate_per_annotation?: number;
  rate_per_hour?: number;
  project_annual_fee?: number;
  created_by: string;
}

export interface ExcelExportData {
  tenant_id: string;
  period: string;
  sheets: Record<string, Record<string, unknown>[]>;
  generated_at: string;
}
