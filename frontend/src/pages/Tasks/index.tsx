// Tasks list page
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ProTable, type ProColumns, type ActionType } from '@ant-design/pro-components';
import { Button, Tag, Space, Modal, Progress, Dropdown, message } from 'antd';
import {
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  MoreOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useRef } from 'react';
import { useTasks, useDeleteTask, useBatchDeleteTasks } from '@/hooks/useTask';
import { TaskCreateModal } from './TaskCreateModal';
import type { Task, TaskStatus, TaskPriority } from '@/types';

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

const TasksPage: React.FC = () => {
  const { t } = useTranslation('common');
  const navigate = useNavigate();
  const actionRef = useRef<ActionType>();
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [currentParams, setCurrentParams] = useState({});

  const { data, isLoading, refetch } = useTasks(currentParams);
  const deleteTask = useDeleteTask();
  const batchDeleteTasks = useBatchDeleteTasks();

  const handleDelete = (id: string) => {
    Modal.confirm({
      title: 'Delete Task',
      icon: <ExclamationCircleOutlined />,
      content: 'Are you sure you want to delete this task?',
      okType: 'danger',
      onOk: async () => {
        await deleteTask.mutateAsync(id);
        refetch();
      },
    });
  };

  const handleBatchDelete = () => {
    if (selectedRowKeys.length === 0) {
      message.warning('Please select tasks to delete');
      return;
    }
    Modal.confirm({
      title: 'Delete Tasks',
      icon: <ExclamationCircleOutlined />,
      content: `Are you sure you want to delete ${selectedRowKeys.length} tasks?`,
      okType: 'danger',
      onOk: async () => {
        await batchDeleteTasks.mutateAsync(selectedRowKeys);
        setSelectedRowKeys([]);
        refetch();
      },
    });
  };

  const columns: ProColumns<Task>[] = [
    {
      title: 'Task Name',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      ellipsis: true,
      render: (_, record) => (
        <a onClick={() => navigate(`/tasks/${record.id}`)}>{record.name}</a>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      valueType: 'select',
      valueEnum: {
        pending: { text: 'Pending', status: 'Default' },
        in_progress: { text: 'In Progress', status: 'Processing' },
        completed: { text: 'Completed', status: 'Success' },
        cancelled: { text: 'Cancelled', status: 'Error' },
      },
      render: (_, record) => (
        <Tag color={statusColorMap[record.status]}>
          {record.status.replace('_', ' ').toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      width: 100,
      valueType: 'select',
      valueEnum: {
        low: { text: 'Low' },
        medium: { text: 'Medium' },
        high: { text: 'High' },
        urgent: { text: 'Urgent' },
      },
      render: (_, record) => (
        <Tag color={priorityColorMap[record.priority]}>
          {record.priority.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Annotation Type',
      dataIndex: 'annotation_type',
      key: 'annotation_type',
      width: 150,
      valueType: 'select',
      valueEnum: {
        text_classification: { text: 'Text Classification' },
        ner: { text: 'NER' },
        sentiment: { text: 'Sentiment' },
        qa: { text: 'Q&A' },
        custom: { text: 'Custom' },
      },
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      width: 180,
      search: false,
      render: (_, record) => (
        <Space direction="vertical" size={0} style={{ width: '100%' }}>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'completed' ? 'success' : 'active'}
          />
          <span style={{ fontSize: 12, color: '#999' }}>
            {record.completed_items} / {record.total_items}
          </span>
        </Space>
      ),
    },
    {
      title: 'Assignee',
      dataIndex: 'assignee_name',
      key: 'assignee_name',
      width: 120,
      ellipsis: true,
      render: (text) => text || '-',
    },
    {
      title: 'Due Date',
      dataIndex: 'due_date',
      key: 'due_date',
      width: 120,
      valueType: 'date',
      render: (_, record) => {
        if (!record.due_date) return '-';
        const dueDate = new Date(record.due_date);
        const isOverdue = dueDate < new Date() && record.status !== 'completed';
        return (
          <span style={{ color: isOverdue ? '#ff4d4f' : undefined }}>
            {dueDate.toLocaleDateString()}
          </span>
        );
      },
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      valueType: 'date',
      search: false,
      sorter: true,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      fixed: 'right',
      search: false,
      render: (_, record) => (
        <Dropdown
          menu={{
            items: [
              {
                key: 'view',
                icon: <EyeOutlined />,
                label: 'View',
                onClick: () => navigate(`/tasks/${record.id}`),
              },
              {
                key: 'edit',
                icon: <EditOutlined />,
                label: 'Edit',
                onClick: () => navigate(`/tasks/${record.id}/edit`),
              },
              { type: 'divider' },
              {
                key: 'delete',
                icon: <DeleteOutlined />,
                label: 'Delete',
                danger: true,
                onClick: () => handleDelete(record.id),
              },
            ],
          }}
        >
          <Button type="text" icon={<MoreOutlined />} />
        </Dropdown>
      ),
    },
  ];

  // Mock data for development
  const mockTasks: Task[] = [
    {
      id: '1',
      name: 'Customer Review Classification',
      description: 'Classify customer reviews by sentiment',
      status: 'in_progress',
      priority: 'high',
      annotation_type: 'sentiment',
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
      tags: ['urgent', 'customer'],
    },
    {
      id: '2',
      name: 'Product Entity Recognition',
      description: 'Identify product names and attributes',
      status: 'pending',
      priority: 'medium',
      annotation_type: 'ner',
      created_by: 'admin',
      created_at: '2025-01-18T09:00:00Z',
      updated_at: '2025-01-18T09:00:00Z',
      due_date: '2025-02-15T00:00:00Z',
      progress: 0,
      total_items: 500,
      completed_items: 0,
      tenant_id: 'tenant1',
      tags: ['product'],
    },
    {
      id: '3',
      name: 'FAQ Classification',
      description: 'Categorize FAQ questions',
      status: 'completed',
      priority: 'low',
      annotation_type: 'text_classification',
      assignee_id: 'user2',
      assignee_name: 'Jane Smith',
      created_by: 'admin',
      created_at: '2025-01-10T08:00:00Z',
      updated_at: '2025-01-19T16:00:00Z',
      progress: 100,
      total_items: 200,
      completed_items: 200,
      tenant_id: 'tenant1',
    },
  ];

  return (
    <>
      <ProTable<Task>
        headerTitle="Annotation Tasks"
        actionRef={actionRef}
        rowKey="id"
        loading={isLoading}
        columns={columns}
        dataSource={data?.items || mockTasks}
        scroll={{ x: 1200 }}
        search={{
          labelWidth: 'auto',
          defaultCollapsed: false,
        }}
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showQuickJumper: true,
          total: data?.total || mockTasks.length,
        }}
        rowSelection={{
          selectedRowKeys,
          onChange: (keys) => setSelectedRowKeys(keys as string[]),
        }}
        toolBarRender={() => [
          selectedRowKeys.length > 0 && (
            <Button
              key="batchDelete"
              danger
              icon={<DeleteOutlined />}
              onClick={handleBatchDelete}
            >
              Delete ({selectedRowKeys.length})
            </Button>
          ),
          <Button
            key="create"
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateModalOpen(true)}
          >
            Create Task
          </Button>,
        ]}
        onSubmit={(params) => {
          setCurrentParams(params);
        }}
        onReset={() => {
          setCurrentParams({});
        }}
      />

      <TaskCreateModal
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onSuccess={() => {
          setCreateModalOpen(false);
          refetch();
        }}
      />
    </>
  );
};

export default TasksPage;
