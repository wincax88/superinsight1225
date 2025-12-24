// Dashboard types

export interface DashboardSummary {
  business_metrics: BusinessMetrics;
  system_performance: SystemPerformance;
  generated_at: string;
}

export interface BusinessMetrics {
  annotation_efficiency?: AnnotationEfficiency;
  user_activity?: UserActivityMetrics;
  ai_models?: AIModelMetrics;
  projects?: ProjectMetrics;
}

export interface SystemPerformance {
  active_requests: number;
  avg_request_duration: Record<string, number>;
  database_performance: Record<string, unknown>;
  ai_performance: Record<string, unknown>;
}

export interface AnnotationEfficiency {
  period_hours: number;
  data_points: number;
  trends: AnnotationTrend[];
  summary: AnnotationSummary;
}

export interface AnnotationTrend {
  timestamp: number;
  datetime: string;
  annotations_per_hour: number;
  average_annotation_time: number;
  quality_score: number;
  completion_rate: number;
  revision_rate: number;
}

export interface AnnotationSummary {
  avg_annotations_per_hour: number;
  avg_quality_score: number;
  avg_completion_rate: number;
  avg_revision_rate: number;
  peak_annotations_per_hour?: number;
  best_quality_score?: number;
  trend_direction?: 'increasing' | 'decreasing' | 'stable';
}

export interface UserActivityMetrics {
  period_hours: number;
  data_points: number;
  trends: UserActivityTrend[];
  summary: UserActivitySummary;
}

export interface UserActivityTrend {
  timestamp: number;
  datetime: string;
  active_users_count: number;
  new_users_count: number;
  returning_users_count: number;
  session_duration_avg: number;
  actions_per_session: number;
  peak_concurrent_users: number;
}

export interface UserActivitySummary {
  avg_active_users: number;
  total_new_users: number;
  avg_session_duration: number;
  avg_actions_per_session: number;
  peak_concurrent_users: number;
  user_growth_trend?: 'increasing' | 'decreasing' | 'stable';
  engagement_trend?: 'increasing' | 'decreasing' | 'stable';
}

export interface AIModelMetrics {
  models: Record<string, AIModelInfo>;
  model_count: number;
  summary: AIModelSummary;
}

export interface AIModelInfo {
  inference_count: number;
  average_inference_time: number;
  success_rate: number;
  confidence_score_avg: number;
  accuracy_score: number;
  error_rate: number;
  last_updated: string;
}

export interface AIModelSummary {
  total_models: number;
  active_models: number;
  avg_success_rate: number;
}

export interface ProjectMetrics {
  projects: Record<string, ProjectInfo>;
  project_count: number;
  summary: ProjectSummary;
}

export interface ProjectInfo {
  completion_percentage: number;
  total_tasks: number;
  completed_tasks: number;
  remaining_tasks: number;
  estimated_completion?: string;
  last_updated: string;
}

export interface ProjectSummary {
  total_projects: number;
  completed_projects: number;
  avg_completion_percentage: number;
  projects_on_track: number;
}

export interface MetricCardData {
  title: string;
  value: number | string;
  trend?: number;
  suffix?: string;
  prefix?: string;
  color?: string;
}

export interface QuickAction {
  id: string;
  title: string;
  icon: string;
  path: string;
  description?: string;
}
