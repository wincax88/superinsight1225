// Data augmentation type definitions

export type AugmentationJobStatus = 'pending' | 'running' | 'completed' | 'failed';

export type AugmentationStrategy =
  | 'back_translation'
  | 'paraphrase'
  | 'synonym_replace'
  | 'noise_injection'
  | 'eda';

export type SampleSource = 'original' | 'augmented';

export interface AugmentationJob {
  id: string;
  name: string;
  status: AugmentationJobStatus;
  strategy: AugmentationStrategy | string;
  source_count: number;
  output_count: number;
  progress: number;
  multiplier?: number;
  description?: string;
  source_dataset_id?: string;
  created_at: string;
  updated_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface SampleData {
  id: string;
  content: string;
  label: string;
  source: SampleSource;
  quality_score: number;
  job_id?: string;
  original_id?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface AugmentationStats {
  total_samples: number;
  augmented_samples: number;
  original_samples: number;
  augmentation_ratio: number;
  jobs_completed: number;
  jobs_running: number;
  jobs_pending: number;
  jobs_failed: number;
  average_quality_score: number;
}

export interface CreateAugmentationJobPayload {
  name: string;
  strategy: AugmentationStrategy | string;
  multiplier?: number;
  description?: string;
  source_dataset_id?: string;
  config?: Record<string, unknown>;
}

export interface UpdateAugmentationJobPayload {
  name?: string;
  description?: string;
  multiplier?: number;
  config?: Record<string, unknown>;
}
