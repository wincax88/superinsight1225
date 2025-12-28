// Quality management page
import { useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Row,
  Col,
  Statistic,
  Progress,
  Tabs,
  List,
  Badge,
  Tooltip,
  message,
} from 'antd';
import {
  CheckCircleOutlined,
  WarningOutlined,
  CloseCircleOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  BugOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

interface QualityRule {
  id: string;
  name: string;
  type: 'format' | 'content' | 'consistency' | 'custom';
  description: string;
  enabled: boolean;
  severity: 'warning' | 'error';
  violations_count: number;
  last_run?: string;
}

interface QualityIssue {
  id: string;
  rule_id: string;
  rule_name: string;
  task_id: string;
  task_name: string;
  severity: 'warning' | 'error';
  description: string;
  status: 'open' | 'fixed' | 'ignored';
  created_at: string;
  assigned_to?: string;
}

// Mock data
const mockRules: QualityRule[] = [
  {
    id: '1',
    name: 'Label Consistency Check',
    type: 'consistency',
    description: 'Ensure labels are consistent across similar samples',
    enabled: true,
    severity: 'error',
    violations_count: 12,
    last_run: '2025-01-20T10:00:00Z',
  },
  {
    id: '2',
    name: 'Text Length Validation',
    type: 'format',
    description: 'Check if annotation text meets minimum length requirements',
    enabled: true,
    severity: 'warning',
    violations_count: 45,
    last_run: '2025-01-20T10:00:00Z',
  },
  {
    id: '3',
    name: 'Empty Label Detection',
    type: 'content',
    description: 'Detect annotations with empty or missing labels',
    enabled: true,
    severity: 'error',
    violations_count: 3,
    last_run: '2025-01-20T10:00:00Z',
  },
  {
    id: '4',
    name: 'Duplicate Detection',
    type: 'consistency',
    description: 'Find duplicate or near-duplicate annotations',
    enabled: false,
    severity: 'warning',
    violations_count: 0,
  },
];

const mockIssues: QualityIssue[] = [
  {
    id: '1',
    rule_id: '1',
    rule_name: 'Label Consistency Check',
    task_id: 'task1',
    task_name: 'Customer Review Classification',
    severity: 'error',
    description: 'Inconsistent labeling for similar review content',
    status: 'open',
    created_at: '2025-01-20T08:30:00Z',
    assigned_to: 'John Doe',
  },
  {
    id: '2',
    rule_id: '2',
    rule_name: 'Text Length Validation',
    task_id: 'task1',
    task_name: 'Customer Review Classification',
    severity: 'warning',
    description: 'Annotation text is below minimum length (5 chars)',
    status: 'fixed',
    created_at: '2025-01-19T14:20:00Z',
  },
  {
    id: '3',
    rule_id: '3',
    rule_name: 'Empty Label Detection',
    task_id: 'task2',
    task_name: 'Product Entity Recognition',
    severity: 'error',
    description: 'Empty label found in annotation #1234',
    status: 'open',
    created_at: '2025-01-20T09:15:00Z',
  },
];

const typeColors: Record<string, string> = {
  format: 'blue',
  content: 'green',
  consistency: 'orange',
  custom: 'purple',
};

const severityColors = {
  warning: 'warning',
  error: 'error',
} as const;

const statusColors = {
  open: 'error',
  fixed: 'success',
  ignored: 'default',
} as const;

const QualityPage: React.FC = () => {
  const [ruleModalOpen, setRuleModalOpen] = useState(false);
  const [ruleForm] = Form.useForm();

  const handleCreateRule = async (values: Record<string, unknown>) => {
    message.success('Quality rule created successfully');
    setRuleModalOpen(false);
    ruleForm.resetFields();
  };

  const handleToggleRule = (id: string, enabled: boolean) => {
    message.success(`Rule ${enabled ? 'enabled' : 'disabled'}`);
  };

  const handleRunAllRules = () => {
    message.info('Running all quality checks...');
  };

  const ruleColumns: ColumnsType<QualityRule> = [
    {
      title: 'Rule Name',
      dataIndex: 'name',
      key: 'name',
      render: (name, record) => (
        <Space>
          {record.enabled ? (
            <CheckCircleOutlined style={{ color: '#52c41a' }} />
          ) : (
            <CloseCircleOutlined style={{ color: '#999' }} />
          )}
          <span style={{ fontWeight: 500 }}>{name}</span>
        </Space>
      ),
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      width: 120,
      render: (type) => <Tag color={typeColors[type]}>{type.toUpperCase()}</Tag>,
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: keyof typeof severityColors) => (
        <Badge status={severityColors[severity]} text={severity.toUpperCase()} />
      ),
    },
    {
      title: 'Violations',
      dataIndex: 'violations_count',
      key: 'violations_count',
      width: 100,
      render: (count) => (
        <Tag color={count > 0 ? 'red' : 'green'}>{count}</Tag>
      ),
    },
    {
      title: 'Enabled',
      dataIndex: 'enabled',
      key: 'enabled',
      width: 100,
      render: (enabled, record) => (
        <Switch
          checked={enabled}
          size="small"
          onChange={(checked) => handleToggleRule(record.id, checked)}
        />
      ),
    },
    {
      title: 'Last Run',
      dataIndex: 'last_run',
      key: 'last_run',
      width: 150,
      render: (date) => (date ? new Date(date).toLocaleString() : '-'),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Tooltip title="Run Rule">
            <Button type="link" size="small" icon={<PlayCircleOutlined />} />
          </Tooltip>
          <Tooltip title="Edit">
            <Button type="link" size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title="Delete">
            <Button type="link" danger size="small" icon={<DeleteOutlined />} />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const issueColumns: ColumnsType<QualityIssue> = [
    {
      title: 'Issue',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Rule',
      dataIndex: 'rule_name',
      key: 'rule_name',
      width: 200,
    },
    {
      title: 'Task',
      dataIndex: 'task_name',
      key: 'task_name',
      width: 200,
      render: (name) => <a>{name}</a>,
    },
    {
      title: 'Severity',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: keyof typeof severityColors) => (
        <Badge status={severityColors[severity]} text={severity.toUpperCase()} />
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date) => new Date(date).toLocaleString(),
    },
  ];

  // Calculate stats
  const totalViolations = mockRules.reduce((sum, r) => sum + r.violations_count, 0);
  const openIssues = mockIssues.filter((i) => i.status === 'open').length;
  const errorCount = mockIssues.filter((i) => i.severity === 'error' && i.status === 'open').length;

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>Quality Management</h2>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Active Rules"
              value={mockRules.filter((r) => r.enabled).length}
              suffix={`/ ${mockRules.length}`}
              prefix={<SafetyCertificateOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Total Violations"
              value={totalViolations}
              prefix={<WarningOutlined />}
              valueStyle={{ color: totalViolations > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Open Issues"
              value={openIssues}
              prefix={<BugOutlined />}
              valueStyle={{ color: openIssues > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Quality Score"
              value={92}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Main Content */}
      <Card>
        <Tabs
          defaultActiveKey="rules"
          items={[
            {
              key: 'rules',
              label: (
                <span>
                  <SafetyCertificateOutlined />
                  Quality Rules
                </span>
              ),
              children: (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <Space>
                      <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() => setRuleModalOpen(true)}
                      >
                        Create Rule
                      </Button>
                      <Button
                        icon={<PlayCircleOutlined />}
                        onClick={handleRunAllRules}
                      >
                        Run All Rules
                      </Button>
                    </Space>
                  </div>
                  <Table
                    columns={ruleColumns}
                    dataSource={mockRules}
                    rowKey="id"
                    pagination={false}
                    expandable={{
                      expandedRowRender: (record) => (
                        <p style={{ margin: 0, color: '#666' }}>
                          {record.description}
                        </p>
                      ),
                    }}
                  />
                </>
              ),
            },
            {
              key: 'issues',
              label: (
                <span>
                  <BugOutlined />
                  Issues
                  {openIssues > 0 && (
                    <Badge
                      count={openIssues}
                      size="small"
                      style={{ marginLeft: 8 }}
                    />
                  )}
                </span>
              ),
              children: (
                <Table
                  columns={issueColumns}
                  dataSource={mockIssues}
                  rowKey="id"
                  pagination={{ pageSize: 10 }}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Create Rule Modal */}
      <Modal
        title="Create Quality Rule"
        open={ruleModalOpen}
        onCancel={() => setRuleModalOpen(false)}
        onOk={() => ruleForm.submit()}
        width={600}
      >
        <Form form={ruleForm} layout="vertical" onFinish={handleCreateRule}>
          <Form.Item
            name="name"
            label="Rule Name"
            rules={[{ required: true, message: 'Please enter rule name' }]}
          >
            <Input placeholder="Enter rule name" />
          </Form.Item>
          <Form.Item
            name="type"
            label="Rule Type"
            rules={[{ required: true }]}
          >
            <Select placeholder="Select type">
              <Select.Option value="format">Format</Select.Option>
              <Select.Option value="content">Content</Select.Option>
              <Select.Option value="consistency">Consistency</Select.Option>
              <Select.Option value="custom">Custom</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="severity"
            label="Severity"
            rules={[{ required: true }]}
            initialValue="warning"
          >
            <Select>
              <Select.Option value="warning">Warning</Select.Option>
              <Select.Option value="error">Error</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input.TextArea rows={3} placeholder="Describe the rule" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default QualityPage;
