// Task management hook
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { message } from 'antd';
import { taskService } from '@/services/task';
import type { TaskListParams, CreateTaskPayload, UpdateTaskPayload } from '@/types';

const QUERY_KEYS = {
  tasks: 'tasks',
  task: 'task',
  taskStats: 'taskStats',
};

export function useTasks(params: TaskListParams = {}) {
  return useQuery({
    queryKey: [QUERY_KEYS.tasks, params],
    queryFn: () => taskService.getList(params),
    staleTime: 30000,
  });
}

export function useTask(id: string) {
  return useQuery({
    queryKey: [QUERY_KEYS.task, id],
    queryFn: () => taskService.getById(id),
    enabled: !!id,
  });
}

export function useTaskStats() {
  return useQuery({
    queryKey: [QUERY_KEYS.taskStats],
    queryFn: () => taskService.getStats(),
    staleTime: 60000,
  });
}

export function useCreateTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: CreateTaskPayload) => taskService.create(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.tasks] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.taskStats] });
      message.success('Task created successfully');
    },
    onError: () => {
      message.error('Failed to create task');
    },
  });
}

export function useUpdateTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: UpdateTaskPayload }) =>
      taskService.update(id, payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.tasks] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.task, data.id] });
      message.success('Task updated successfully');
    },
    onError: () => {
      message.error('Failed to update task');
    },
  });
}

export function useDeleteTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => taskService.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.tasks] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.taskStats] });
      message.success('Task deleted successfully');
    },
    onError: () => {
      message.error('Failed to delete task');
    },
  });
}

export function useAssignTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, userId }: { id: string; userId: string }) =>
      taskService.assign(id, userId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.tasks] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.task, data.id] });
      message.success('Task assigned successfully');
    },
    onError: () => {
      message.error('Failed to assign task');
    },
  });
}

export function useBatchDeleteTasks() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ids: string[]) => taskService.batchDelete(ids),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.tasks] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.taskStats] });
      message.success('Tasks deleted successfully');
    },
    onError: () => {
      message.error('Failed to delete tasks');
    },
  });
}
