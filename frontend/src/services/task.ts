// Task API service
import apiClient from './api/client';
import { API_ENDPOINTS } from '@/constants';
import type {
  Task,
  TaskListParams,
  TaskListResponse,
  CreateTaskPayload,
  UpdateTaskPayload,
  TaskStats,
} from '@/types';

export const taskService = {
  // Get task list with pagination and filters
  async getList(params: TaskListParams = {}): Promise<TaskListResponse> {
    const response = await apiClient.get<TaskListResponse>(API_ENDPOINTS.TASKS.BASE, { params });
    return response.data;
  },

  // Get single task by ID
  async getById(id: string): Promise<Task> {
    const response = await apiClient.get<Task>(API_ENDPOINTS.TASKS.BY_ID(id));
    return response.data;
  },

  // Create a new task
  async create(payload: CreateTaskPayload): Promise<Task> {
    const response = await apiClient.post<Task>(API_ENDPOINTS.TASKS.BASE, payload);
    return response.data;
  },

  // Update an existing task
  async update(id: string, payload: UpdateTaskPayload): Promise<Task> {
    const response = await apiClient.patch<Task>(API_ENDPOINTS.TASKS.BY_ID(id), payload);
    return response.data;
  },

  // Delete a task
  async delete(id: string): Promise<void> {
    await apiClient.delete(API_ENDPOINTS.TASKS.BY_ID(id));
  },

  // Get task statistics
  async getStats(): Promise<TaskStats> {
    const response = await apiClient.get<TaskStats>(API_ENDPOINTS.TASKS.STATS);
    return response.data;
  },

  // Assign task to user
  async assign(id: string, userId: string): Promise<Task> {
    const response = await apiClient.post<Task>(API_ENDPOINTS.TASKS.ASSIGN(id), {
      assignee_id: userId,
    });
    return response.data;
  },

  // Batch operations
  async batchDelete(ids: string[]): Promise<void> {
    await apiClient.post(API_ENDPOINTS.TASKS.BATCH, {
      action: 'delete',
      ids,
    });
  },

  async batchUpdateStatus(ids: string[], status: string): Promise<void> {
    await apiClient.post(API_ENDPOINTS.TASKS.BATCH, {
      action: 'update_status',
      ids,
      status,
    });
  },
};
