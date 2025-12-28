// Data augmentation service
import { apiClient } from './api/client';
import { API_ENDPOINTS } from '@/constants';

export interface AugmentationJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  strategy: string;
  source_count: number;
  output_count: number;
  progress: number;
  created_at: string;
  completed_at?: string;
}

export interface SampleData {
  id: string;
  content: string;
  label: string;
  source: 'original' | 'augmented';
  quality_score: number;
  created_at: string;
}

export interface CreateJobPayload {
  name: string;
  strategy: string;
  multiplier?: number;
  description?: string;
  source_dataset_id?: string;
}

export interface JobListParams {
  page?: number;
  page_size?: number;
  status?: string;
  strategy?: string;
}

export interface JobListResponse {
  items: AugmentationJob[];
  total: number;
  page: number;
  page_size: number;
}

export interface SampleListParams {
  page?: number;
  page_size?: number;
  source?: 'original' | 'augmented';
  job_id?: string;
}

export interface SampleListResponse {
  items: SampleData[];
  total: number;
  page: number;
  page_size: number;
}

export const augmentationService = {
  // Job management
  async getJobs(params?: JobListParams): Promise<JobListResponse> {
    const response = await apiClient.get<JobListResponse>(
      API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs',
      { params }
    );
    return response.data;
  },

  async getJob(id: string): Promise<AugmentationJob> {
    const response = await apiClient.get<AugmentationJob>(
      `${API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs'}/${id}`
    );
    return response.data;
  },

  async createJob(payload: CreateJobPayload): Promise<AugmentationJob> {
    const response = await apiClient.post<AugmentationJob>(
      API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs',
      payload
    );
    return response.data;
  },

  async startJob(id: string): Promise<AugmentationJob> {
    const response = await apiClient.post<AugmentationJob>(
      `${API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs'}/${id}/start`
    );
    return response.data;
  },

  async pauseJob(id: string): Promise<AugmentationJob> {
    const response = await apiClient.post<AugmentationJob>(
      `${API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs'}/${id}/pause`
    );
    return response.data;
  },

  async deleteJob(id: string): Promise<void> {
    await apiClient.delete(
      `${API_ENDPOINTS.AUGMENTATION?.JOBS || '/api/augmentation/jobs'}/${id}`
    );
  },

  // Sample management
  async getSamples(params?: SampleListParams): Promise<SampleListResponse> {
    const response = await apiClient.get<SampleListResponse>(
      API_ENDPOINTS.AUGMENTATION?.SAMPLES || '/api/augmentation/samples',
      { params }
    );
    return response.data;
  },

  async uploadSamples(file: File): Promise<{ count: number }> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<{ count: number }>(
      API_ENDPOINTS.AUGMENTATION?.UPLOAD || '/api/augmentation/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  // Statistics
  async getStats(): Promise<{
    total_samples: number;
    augmented_samples: number;
    augmentation_ratio: number;
    jobs_completed: number;
    jobs_running: number;
  }> {
    const response = await apiClient.get(
      API_ENDPOINTS.AUGMENTATION?.STATS || '/api/augmentation/stats'
    );
    return response.data;
  },
};
