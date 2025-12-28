// Task detail page
import { useParams, useNavigate } from 'react-router-dom';
import {
  Card,
  Descriptions,
  Tag,
  Progress,
  Space,
  Button,
  Skeleton,
  Alert,
  Timeline,
  Divider,
  Row,
  Col,
  Statistic,
} from 'antd';
import {
  ArrowLeftOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { useTask, useUpdateTask, useDeleteTask } from '@/hooks/useTask';
import type { TaskStatus, TaskPriority } from '@/types';

const statusColorMap: Record<TaskStatus, string> = {
  pending: 'default',
  in_progress: 'processing',
  completed: 'success',
  cancelled: 'error',
};

const priorityColorMap: Record<TaskPriority, string> = {
  low: 'green',
  medium: 'blue',
  high: 'orange',
  urgent: 'red',
};

const TaskDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: task, isLoading, error } = useTask(id || '');
  const updateTask = useUpdateTask();
  const deleteTask = useDeleteTask();

  // Mock data for development
  const mockTask = {
    id: id || '1',
    name: 'Customer Review Classification',
    description:
      'Classify customer reviews by sentiment to understand customer satisfaction levels. This task involves analyzing text reviews and categorizing them as positive, negative, or neutral.',
    status: 'in_progress' as TaskStatus,
    priority: 'high' as TaskPriority,
    annotation_type: 'sentiment' as const,
    assignee_id: 'user1',
    assignee_name: 'John Doe',
    created_by: 'admin',
    created_at: '2025-01-15T10:00:00Z',
    updated_at: '2025-01-20T14:30:00Z',
    due_date: '2025-02-01T00:00:00Z',
    progress: 65,
    total_items: 1000,
    completed_items: 650,
    tenant_id: 'tenant1',
    label_studio_project_id: 'ls-project-123',
    tags: ['urgent', 'customer', 'sentiment'],
  };

  const currentTask = task || mockTask;

  const handleStatusChange = async (newStatus: TaskStatus) => {
    if (id) {
      await updateTask.mutateAsync({ id, payload: { status: newStatus } });
    }
  };

  const handleDelete = async () => {
    if (id) {
      await deleteTask.mutateAsync(id);
      navigate('/tasks');
    }
  };

  if (isLoading) {
    return (
      <Card>
        <Skeleton active paragraph={{ rows: 10 }} />
      </Card>
    );
  }

  if (error && !mockTask) {
    return (
      <Alert
        type="error"
        message="Failed to load task"
        description="Could not fetch task details. Please try again."
        showIcon
      />
    );
  }

  return (
    <div>
      {/* Header */}
      <Card style={{ marginBottom: 16 }}>
        <Space style={{ marginBottom: 16 }}>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/tasks')}>
            Back to Tasks
          </Button>
        </Space>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
          }}
        >
          <div>
            <h2 style={{ marginBottom: 8 }}>{currentTask.name}</h2>
            <Space>
              <Tag color={statusColorMap[currentTask.status]}>
                {currentTask.status.replace('_', ' ').toUpperCase()}
              </Tag>
              <Tag color={priorityColorMap[currentTask.priority]}>
                {currentTask.priority.toUpperCase()}
              </Tag>
              {currentTask.tags?.map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </Space>
          </div>

          <Space>
            {currentTask.status === 'pending' && (
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => handleStatusChange('in_progress')}
              >
                Start Task
              </Button>
            )}
            {currentTask.status === 'in_progress' && (
              <Button
                type="primary"
                icon={<CheckCircleOutlined />}
                onClick={() => handleStatusChange('completed')}
              >
                Complete Task
              </Button>
            )}
            <Button icon={<EditOutlined />} onClick={() => navigate(`/tasks/${id}/edit`)}>
              Edit
            </Button>
            <Button danger icon={<DeleteOutlined />} onClick={handleDelete}>
              Delete
            </Button>
          </Space>
        </div>
      </Card>

      <Row gutter={16}>
        {/* Main Content */}
        <Col xs={24} lg={16}>
          {/* Progress Card */}
          <Card title="Progress" style={{ marginBottom: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Total Items"
                  value={currentTask.total_items}
                  prefix={<ClockCircleOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Completed"
                  value={currentTask.completed_items}
                  valueStyle={{ color: '#52c41a' }}
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="Remaining"
                  value={currentTask.total_items - currentTask.completed_items}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
            </Row>
            <Divider />
            <div>
              <span style={{ marginBottom: 8, display: 'block' }}>
                Overall Progress: {currentTask.progress}%
              </span>
              <Progress
                percent={currentTask.progress}
                status={currentTask.status === 'completed' ? 'success' : 'active'}
                strokeWidth={12}
              />
            </div>
          </Card>

          {/* Description */}
          <Card title="Description" style={{ marginBottom: 16 }}>
            <p>{currentTask.description || 'No description provided.'}</p>
          </Card>

          {/* Label Studio Integration */}
          {currentTask.label_studio_project_id && (
            <Card title="Label Studio" style={{ marginBottom: 16 }}>
              <Alert
                message="Label Studio Integration"
                description={
                  <div>
                    <p>
                      Project ID: <strong>{currentTask.label_studio_project_id}</strong>
                    </p>
                    <Button type="primary" style={{ marginTop: 8 }}>
                      Open in Label Studio
                    </Button>
                  </div>
                }
                type="info"
                showIcon
              />
            </Card>
          )}
        </Col>

        {/* Sidebar */}
        <Col xs={24} lg={8}>
          {/* Details */}
          <Card title="Details" style={{ marginBottom: 16 }}>
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Annotation Type">
                {currentTask.annotation_type.replace('_', ' ')}
              </Descriptions.Item>
              <Descriptions.Item label="Assignee">
                {currentTask.assignee_name || 'Unassigned'}
              </Descriptions.Item>
              <Descriptions.Item label="Created By">{currentTask.created_by}</Descriptions.Item>
              <Descriptions.Item label="Created At">
                {new Date(currentTask.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Updated At">
                {new Date(currentTask.updated_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Due Date">
                {currentTask.due_date
                  ? new Date(currentTask.due_date).toLocaleDateString()
                  : 'No due date'}
              </Descriptions.Item>
            </Descriptions>
          </Card>

          {/* Activity Timeline */}
          <Card title="Activity">
            <Timeline
              items={[
                {
                  color: 'green',
                  children: (
                    <>
                      <p style={{ marginBottom: 4 }}>Task created</p>
                      <small style={{ color: '#999' }}>
                        {new Date(currentTask.created_at).toLocaleString()}
                      </small>
                    </>
                  ),
                },
                {
                  color: 'blue',
                  children: (
                    <>
                      <p style={{ marginBottom: 4 }}>Assigned to {currentTask.assignee_name}</p>
                      <small style={{ color: '#999' }}>
                        {new Date(currentTask.updated_at).toLocaleString()}
                      </small>
                    </>
                  ),
                },
                {
                  color: 'gray',
                  children: (
                    <>
                      <p style={{ marginBottom: 4 }}>Progress updated to 65%</p>
                      <small style={{ color: '#999' }}>
                        {new Date(currentTask.updated_at).toLocaleString()}
                      </small>
                    </>
                  ),
                },
              ]}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TaskDetailPage;
