// React Query hooks for data augmentation
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { message } from 'antd';
import {
  augmentationService,
  type JobListParams,
  type SampleListParams,
  type CreateJobPayload,
} from '@/services/augmentation';

const QUERY_KEYS = {
  jobs: 'augmentation-jobs',
  job: 'augmentation-job',
  samples: 'augmentation-samples',
  stats: 'augmentation-stats',
} as const;

// Jobs
export function useAugmentationJobs(params?: JobListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.jobs, params],
    queryFn: () => augmentationService.getJobs(params),
  });
}

export function useAugmentationJob(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.job, id],
    queryFn: () => augmentationService.getJob(id),
    enabled: !!id,
  });
}

export function useCreateAugmentationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: CreateJobPayload) => augmentationService.createJob(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.jobs] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Augmentation job created successfully');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to create augmentation job');
    },
  });
}

export function useStartAugmentationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => augmentationService.startJob(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.jobs] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.job, id] });
      message.success('Augmentation job started');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to start job');
    },
  });
}

export function usePauseAugmentationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => augmentationService.pauseJob(id),
    onSuccess: (_, id) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.jobs] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.job, id] });
      message.success('Augmentation job paused');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to pause job');
    },
  });
}

export function useDeleteAugmentationJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => augmentationService.deleteJob(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.jobs] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Augmentation job deleted');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to delete job');
    },
  });
}

// Samples
export function useAugmentationSamples(params?: SampleListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.samples, params],
    queryFn: () => augmentationService.getSamples(params),
  });
}

export function useUploadSamples() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => augmentationService.uploadSamples(file),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.samples] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success(`Successfully uploaded ${data.count} samples`);
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to upload samples');
    },
  });
}

// Stats
export function useAugmentationStats() {
  return useQuery({
    queryKey: [QUERY_KEYS.stats],
    queryFn: () => augmentationService.getStats(),
  });
}
