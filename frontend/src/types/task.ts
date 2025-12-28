// Task type definitions

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled';
export type TaskPriority = 'low' | 'medium' | 'high' | 'urgent';
export type AnnotationType = 'text_classification' | 'ner' | 'sentiment' | 'qa' | 'custom';

export interface Task {
  id: string;
  name: string;
  description?: string;
  status: TaskStatus;
  priority: TaskPriority;
  annotation_type: AnnotationType;
  assignee_id?: string;
  assignee_name?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  due_date?: string;
  progress: number;
  total_items: number;
  completed_items: number;
  tenant_id: string;
  label_studio_project_id?: string;
  tags?: string[];
}

export interface TaskListParams {
  page?: number;
  page_size?: number;
  status?: TaskStatus;
  priority?: TaskPriority;
  assignee_id?: string;
  search?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface TaskListResponse {
  items: Task[];
  total: number;
  page: number;
  page_size: number;
}

export interface CreateTaskPayload {
  name: string;
  description?: string;
  priority: TaskPriority;
  annotation_type: AnnotationType;
  assignee_id?: string;
  due_date?: string;
  tags?: string[];
  data_source?: {
    type: 'file' | 'api';
    config: Record<string, unknown>;
  };
}

export interface UpdateTaskPayload {
  name?: string;
  description?: string;
  status?: TaskStatus;
  priority?: TaskPriority;
  assignee_id?: string;
  due_date?: string;
  tags?: string[];
}

export interface TaskStats {
  total: number;
  pending: number;
  in_progress: number;
  completed: number;
  cancelled: number;
  overdue: number;
}
