// React Query hooks for quality management
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { message } from 'antd';
import {
  qualityService,
  type RuleListParams,
  type IssueListParams,
  type CreateRulePayload,
  type UpdateRulePayload,
} from '@/services/quality';

const QUERY_KEYS = {
  rules: 'quality-rules',
  rule: 'quality-rule',
  issues: 'quality-issues',
  issue: 'quality-issue',
  stats: 'quality-stats',
} as const;

// Rules
export function useQualityRules(params?: RuleListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.rules, params],
    queryFn: () => qualityService.getRules(params),
  });
}

export function useQualityRule(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.rule, id],
    queryFn: () => qualityService.getRule(id),
    enabled: !!id,
  });
}

export function useCreateQualityRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: CreateRulePayload) => qualityService.createRule(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Quality rule created successfully');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to create quality rule');
    },
  });
}

export function useUpdateQualityRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateRulePayload }) =>
      qualityService.updateRule(id, payload),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rule, id] });
      message.success('Quality rule updated successfully');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to update quality rule');
    },
  });
}

export function useDeleteQualityRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => qualityService.deleteRule(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Quality rule deleted');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to delete quality rule');
    },
  });
}

export function useToggleQualityRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      qualityService.toggleRule(id, enabled),
    onSuccess: (_, { id, enabled }) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rule, id] });
      message.success(`Quality rule ${enabled ? 'enabled' : 'disabled'}`);
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to toggle quality rule');
    },
  });
}

export function useRunQualityRule() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => qualityService.runRule(id),
    onSuccess: (data, id) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rule, id] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issues] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success(`Quality check completed. Found ${data.violations_found} violations.`);
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to run quality check');
    },
  });
}

export function useRunAllQualityRules() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => qualityService.runAllRules(),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.rules] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issues] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success(`All quality checks completed. Found ${data.total_violations} violations.`);
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to run quality checks');
    },
  });
}

// Issues
export function useQualityIssues(params?: IssueListParams) {
  return useQuery({
    queryKey: [QUERY_KEYS.issues, params],
    queryFn: () => qualityService.getIssues(params),
  });
}

export function useQualityIssue(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.issue, id],
    queryFn: () => qualityService.getIssue(id),
    enabled: !!id,
  });
}

export function useUpdateIssueStatus() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      id,
      status,
    }: {
      id: string;
      status: 'open' | 'fixed' | 'ignored';
    }) => qualityService.updateIssueStatus(id, status),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issues] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issue, id] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.stats] });
      message.success('Issue status updated');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to update issue status');
    },
  });
}

export function useAssignIssue() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, userId }: { id: string; userId: string }) =>
      qualityService.assignIssue(id, userId),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issues] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.issue, id] });
      message.success('Issue assigned successfully');
    },
    onError: (error: Error) => {
      message.error(error.message || 'Failed to assign issue');
    },
  });
}

// Stats
export function useQualityStats() {
  return useQuery({
    queryKey: [QUERY_KEYS.stats],
    queryFn: () => qualityService.getStats(),
  });
}
