// Data augmentation management page
import { useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Progress,
  Row,
  Col,
  Statistic,
  message,
  Tabs,
  Alert,
} from 'antd';
import {
  UploadOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { UploadProps } from 'antd';

interface AugmentationJob {
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

interface SampleData {
  id: string;
  content: string;
  label: string;
  source: 'original' | 'augmented';
  quality_score: number;
  created_at: string;
}

// Mock data
const mockJobs: AugmentationJob[] = [
  {
    id: '1',
    name: 'Customer Review Augmentation',
    status: 'completed',
    strategy: 'back_translation',
    source_count: 1000,
    output_count: 3500,
    progress: 100,
    created_at: '2025-01-15T10:00:00Z',
    completed_at: '2025-01-15T12:30:00Z',
  },
  {
    id: '2',
    name: 'Product Description Enhancement',
    status: 'running',
    strategy: 'paraphrase',
    source_count: 500,
    output_count: 850,
    progress: 68,
    created_at: '2025-01-20T09:00:00Z',
  },
  {
    id: '3',
    name: 'FAQ Expansion',
    status: 'pending',
    strategy: 'synonym_replace',
    source_count: 200,
    output_count: 0,
    progress: 0,
    created_at: '2025-01-20T14:00:00Z',
  },
];

const mockSamples: SampleData[] = [
  {
    id: '1',
    content: 'This product is amazing and works great!',
    label: 'positive',
    source: 'original',
    quality_score: 95,
    created_at: '2025-01-15',
  },
  {
    id: '2',
    content: 'This product is wonderful and functions excellently!',
    label: 'positive',
    source: 'augmented',
    quality_score: 92,
    created_at: '2025-01-15',
  },
  {
    id: '3',
    content: 'The quality is poor and I am disappointed.',
    label: 'negative',
    source: 'original',
    quality_score: 94,
    created_at: '2025-01-15',
  },
];

const statusColors = {
  pending: 'default',
  running: 'processing',
  completed: 'success',
  failed: 'error',
} as const;

const strategyLabels: Record<string, string> = {
  back_translation: 'Back Translation',
  paraphrase: 'Paraphrase',
  synonym_replace: 'Synonym Replacement',
  noise_injection: 'Noise Injection',
  eda: 'Easy Data Augmentation',
};

const AugmentationPage: React.FC = () => {
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [form] = Form.useForm();

  const handleCreateJob = async (values: Record<string, unknown>) => {
    message.success('Augmentation job created successfully');
    setCreateModalOpen(false);
    form.resetFields();
  };

  const handleUpload: UploadProps['onChange'] = (info) => {
    if (info.file.status === 'done') {
      message.success(`${info.file.name} uploaded successfully`);
      setUploadModalOpen(false);
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} upload failed`);
    }
  };

  const jobColumns: ColumnsType<AugmentationJob> = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <a>{name}</a>,
    },
    {
      title: 'Strategy',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy) => (
        <Tag color="blue">{strategyLabels[strategy] || strategy}</Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Progress',
      key: 'progress',
      width: 200,
      render: (_, record) => (
        <Progress
          percent={record.progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : undefined}
        />
      ),
    },
    {
      title: 'Source / Output',
      key: 'counts',
      render: (_, record) => (
        <span>
          {record.source_count.toLocaleString()} → {record.output_count.toLocaleString()}
        </span>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          {record.status === 'pending' && (
            <Button type="link" size="small" icon={<PlayCircleOutlined />}>
              Start
            </Button>
          )}
          {record.status === 'running' && (
            <Button type="link" size="small" icon={<PauseCircleOutlined />}>
              Pause
            </Button>
          )}
          <Button type="link" size="small" icon={<EyeOutlined />}>
            View
          </Button>
          <Button type="link" danger size="small" icon={<DeleteOutlined />}>
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  const sampleColumns: ColumnsType<SampleData> = [
    {
      title: 'Content',
      dataIndex: 'content',
      key: 'content',
      ellipsis: true,
    },
    {
      title: 'Label',
      dataIndex: 'label',
      key: 'label',
      width: 100,
      render: (label) => <Tag>{label}</Tag>,
    },
    {
      title: 'Source',
      dataIndex: 'source',
      key: 'source',
      width: 120,
      render: (source) => (
        <Tag color={source === 'original' ? 'blue' : 'green'}>
          {source.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Quality',
      dataIndex: 'quality_score',
      key: 'quality_score',
      width: 100,
      render: (score) => (
        <Progress
          percent={score}
          size="small"
          status={score >= 90 ? 'success' : 'normal'}
        />
      ),
    },
  ];

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>Data Augmentation</h2>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Samples"
              value={25680}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Augmented Samples"
              value={18450}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Augmentation Ratio"
              value="3.2x"
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Main Content */}
      <Card>
        <Tabs
          defaultActiveKey="jobs"
          items={[
            {
              key: 'jobs',
              label: 'Augmentation Jobs',
              children: (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <Space>
                      <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setCreateModalOpen(true)}
                      >
                        Create Job
                      </Button>
                      <Button
                        icon={<UploadOutlined />}
                        onClick={() => setUploadModalOpen(true)}
                      >
                        Upload Samples
                      </Button>
                    </Space>
                  </div>
                  <Table
                    columns={jobColumns}
                    dataSource={mockJobs}
                    rowKey="id"
                    pagination={{ pageSize: 10 }}
                  />
                </>
              ),
            },
            {
              key: 'samples',
              label: 'Sample Data',
              children: (
                <>
                  <Alert
                    message="Sample Preview"
                    description="This shows a comparison between original and augmented samples."
                    type="info"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  <Table
                    columns={sampleColumns}
                    dataSource={mockSamples}
                    rowKey="id"
                    pagination={{ pageSize: 10 }}
                  />
                </>
              ),
            },
          ]}
        />
      </Card>

      {/* Create Job Modal */}
      <Modal
        title="Create Augmentation Job"
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => form.submit()}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateJob}>
          <Form.Item
            name="name"
            label="Job Name"
            rules={[{ required: true, message: 'Please enter job name' }]}
          >
            <Input placeholder="Enter job name" />
          </Form.Item>
          <Form.Item
            name="strategy"
            label="Augmentation Strategy"
            rules={[{ required: true }]}
          >
            <Select placeholder="Select strategy">
              <Select.Option value="back_translation">Back Translation</Select.Option>
              <Select.Option value="paraphrase">Paraphrase</Select.Option>
              <Select.Option value="synonym_replace">Synonym Replacement</Select.Option>
              <Select.Option value="eda">Easy Data Augmentation (EDA)</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="multiplier"
            label="Augmentation Multiplier"
            initialValue={3}
          >
            <InputNumber min={1} max={10} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={3} placeholder="Optional description" />
          </Form.Item>
        </Form>
      </Modal>

      {/* Upload Modal */}
      <Modal
        title="Upload Sample Data"
        open={uploadModalOpen}
        onCancel={() => setUploadModalOpen(false)}
        footer={null}
      >
        <Upload.Dragger
          name="file"
          accept=".csv,.json,.jsonl"
          onChange={handleUpload}
          action="/api/augmentation/upload"
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">Click or drag file to upload</p>
          <p className="ant-upload-hint">
            Support CSV, JSON, or JSONL format. Max file size: 100MB.
          </p>
        </Upload.Dragger>
      </Modal>
    </div>
  );
};

export default AugmentationPage;
