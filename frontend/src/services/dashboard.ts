// Dashboard service
import apiClient from './api/client';
import { API_ENDPOINTS } from '@/constants';
import type { DashboardSummary, AnnotationEfficiency, UserActivityMetrics, AIModelMetrics, ProjectMetrics } from '@/types';

export const dashboardService = {
  async getSummary(): Promise<DashboardSummary> {
    const response = await apiClient.get<DashboardSummary>(API_ENDPOINTS.METRICS.SUMMARY);
    return response.data;
  },

  async getAnnotationEfficiency(hours = 24): Promise<AnnotationEfficiency> {
    const response = await apiClient.get<AnnotationEfficiency>(API_ENDPOINTS.METRICS.ANNOTATION_EFFICIENCY, {
      params: { hours },
    });
    return response.data;
  },

  async getUserActivity(hours = 24): Promise<UserActivityMetrics> {
    const response = await apiClient.get<UserActivityMetrics>(API_ENDPOINTS.METRICS.USER_ACTIVITY, {
      params: { hours },
    });
    return response.data;
  },

  async getAIModels(modelName?: string, hours = 24): Promise<AIModelMetrics> {
    const response = await apiClient.get<AIModelMetrics>(API_ENDPOINTS.METRICS.AI_MODELS, {
      params: { model_name: modelName, hours },
    });
    return response.data;
  },

  async getProjects(projectId?: string, hours = 24): Promise<ProjectMetrics> {
    const response = await apiClient.get<ProjectMetrics>(API_ENDPOINTS.METRICS.PROJECTS, {
      params: { project_id: projectId, hours },
    });
    return response.data;
  },
};
