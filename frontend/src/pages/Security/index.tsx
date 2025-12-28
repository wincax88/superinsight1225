// Security audit page
import { useState } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  DatePicker,
  Input,
  Select,
  Row,
  Col,
  Statistic,
  Timeline,
  Tabs,
  Alert,
  Modal,
  Form,
  Descriptions,
  Typography,
} from 'antd';
import {
  SecurityScanOutlined,
  UserOutlined,
  LockOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExportOutlined,
  EyeOutlined,
  FilterOutlined,
  GlobalOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { RangePicker } = DatePicker;
const { Text } = Typography;

interface AuditLog {
  id: string;
  user_id: string;
  user_name: string;
  action: string;
  resource: string;
  ip_address: string;
  user_agent: string;
  status: 'success' | 'failed';
  details?: string;
  created_at: string;
}

interface SecurityEvent {
  id: string;
  type: 'login_attempt' | 'permission_change' | 'data_access' | 'config_change';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  user_name?: string;
  ip_address?: string;
  created_at: string;
  resolved: boolean;
}

// Mock data
const mockAuditLogs: AuditLog[] = [
  {
    id: '1',
    user_id: 'user1',
    user_name: 'admin',
    action: 'LOGIN',
    resource: 'auth/login',
    ip_address: '192.168.1.100',
    user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    status: 'success',
    created_at: '2025-01-20T10:30:00Z',
  },
  {
    id: '2',
    user_id: 'user2',
    user_name: 'john.doe',
    action: 'CREATE',
    resource: 'tasks/123',
    ip_address: '192.168.1.101',
    user_agent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    status: 'success',
    details: 'Created task "Customer Review Classification"',
    created_at: '2025-01-20T09:45:00Z',
  },
  {
    id: '3',
    user_id: 'user3',
    user_name: 'jane.smith',
    action: 'DELETE',
    resource: 'annotations/456',
    ip_address: '192.168.1.102',
    user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    status: 'success',
    created_at: '2025-01-20T09:30:00Z',
  },
  {
    id: '4',
    user_id: 'unknown',
    user_name: 'unknown',
    action: 'LOGIN',
    resource: 'auth/login',
    ip_address: '45.33.32.156',
    user_agent: 'curl/7.64.1',
    status: 'failed',
    details: 'Invalid credentials - 3rd attempt',
    created_at: '2025-01-20T08:15:00Z',
  },
];

const mockSecurityEvents: SecurityEvent[] = [
  {
    id: '1',
    type: 'login_attempt',
    severity: 'high',
    description: 'Multiple failed login attempts detected from IP 45.33.32.156',
    ip_address: '45.33.32.156',
    created_at: '2025-01-20T08:20:00Z',
    resolved: false,
  },
  {
    id: '2',
    type: 'permission_change',
    severity: 'medium',
    description: 'User role changed from annotator to manager',
    user_name: 'jane.smith',
    created_at: '2025-01-19T14:30:00Z',
    resolved: true,
  },
  {
    id: '3',
    type: 'data_access',
    severity: 'low',
    description: 'Large data export initiated',
    user_name: 'admin',
    created_at: '2025-01-19T11:00:00Z',
    resolved: true,
  },
];

const actionColors: Record<string, string> = {
  LOGIN: 'blue',
  LOGOUT: 'default',
  CREATE: 'green',
  UPDATE: 'orange',
  DELETE: 'red',
  VIEW: 'cyan',
  EXPORT: 'purple',
};

const severityColors = {
  low: 'default',
  medium: 'warning',
  high: 'orange',
  critical: 'error',
} as const;

const SecurityPage: React.FC = () => {
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);

  const handleViewDetail = (log: AuditLog) => {
    setSelectedLog(log);
    setDetailModalOpen(true);
  };

  const handleExport = () => {
    // Export logic
  };

  const logColumns: ColumnsType<AuditLog> = [
    {
      title: 'Time',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date) => new Date(date).toLocaleString(),
      sorter: (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    },
    {
      title: 'User',
      dataIndex: 'user_name',
      key: 'user_name',
      width: 120,
      render: (name) => (
        <Space>
          <UserOutlined />
          {name}
        </Space>
      ),
    },
    {
      title: 'Action',
      dataIndex: 'action',
      key: 'action',
      width: 100,
      render: (action) => (
        <Tag color={actionColors[action] || 'default'}>{action}</Tag>
      ),
    },
    {
      title: 'Resource',
      dataIndex: 'resource',
      key: 'resource',
      ellipsis: true,
    },
    {
      title: 'IP Address',
      dataIndex: 'ip_address',
      key: 'ip_address',
      width: 140,
      render: (ip) => (
        <Space>
          <GlobalOutlined />
          {ip}
        </Space>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status) =>
        status === 'success' ? (
          <Tag icon={<CheckCircleOutlined />} color="success">
            Success
          </Tag>
        ) : (
          <Tag icon={<WarningOutlined />} color="error">
            Failed
          </Tag>
        ),
      filters: [
        { text: 'Success', value: 'success' },
        { text: 'Failed', value: 'failed' },
      ],
      onFilter: (value, record) => record.status === value,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 80,
      render: (_, record) => (
        <Button
          type="link"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetail(record)}
        >
          View
        </Button>
      ),
    },
  ];

  // Calculate stats
  const todayLogs = mockAuditLogs.filter(
    (log) => new Date(log.created_at).toDateString() === new Date().toDateString()
  );
  const failedLogs = mockAuditLogs.filter((log) => log.status === 'failed');
  const unresolvedEvents = mockSecurityEvents.filter((e) => !e.resolved);

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>Security Audit</h2>

      {/* Stats */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Total Logs (Today)"
              value={todayLogs.length}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Failed Attempts"
              value={failedLogs.length}
              prefix={<WarningOutlined />}
              valueStyle={{ color: failedLogs.length > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Security Events"
              value={unresolvedEvents.length}
              prefix={<SecurityScanOutlined />}
              valueStyle={{ color: unresolvedEvents.length > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="Active Users"
              value={3}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Alerts */}
      {unresolvedEvents.length > 0 && (
        <Alert
          message="Security Alert"
          description={`There are ${unresolvedEvents.length} unresolved security events that require attention.`}
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
          action={
            <Button size="small" type="primary">
              View Events
            </Button>
          }
        />
      )}

      {/* Main Content */}
      <Card>
        <Tabs
          defaultActiveKey="logs"
          items={[
            {
              key: 'logs',
              label: (
                <span>
                  <LockOutlined />
                  Audit Logs
                </span>
              ),
              children: (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <Space wrap>
                      <RangePicker />
                      <Select
                        placeholder="Action"
                        style={{ width: 120 }}
                        allowClear
                        options={[
                          { value: 'LOGIN', label: 'Login' },
                          { value: 'CREATE', label: 'Create' },
                          { value: 'UPDATE', label: 'Update' },
                          { value: 'DELETE', label: 'Delete' },
                        ]}
                      />
                      <Input.Search
                        placeholder="Search user or resource"
                        style={{ width: 250 }}
                      />
                      <Button icon={<FilterOutlined />}>More Filters</Button>
                      <Button icon={<ExportOutlined />} onClick={handleExport}>
                        Export
                      </Button>
                    </Space>
                  </div>
                  <Table
                    columns={logColumns}
                    dataSource={mockAuditLogs}
                    rowKey="id"
                    pagination={{
                      pageSize: 10,
                      showSizeChanger: true,
                      showTotal: (total) => `Total ${total} logs`,
                    }}
                  />
                </>
              ),
            },
            {
              key: 'events',
              label: (
                <span>
                  <SecurityScanOutlined />
                  Security Events
                  {unresolvedEvents.length > 0 && (
                    <Tag color="red" style={{ marginLeft: 8 }}>
                      {unresolvedEvents.length}
                    </Tag>
                  )}
                </span>
              ),
              children: (
                <Timeline
                  items={mockSecurityEvents.map((event) => ({
                    color:
                      event.severity === 'critical' || event.severity === 'high'
                        ? 'red'
                        : event.severity === 'medium'
                        ? 'orange'
                        : 'blue',
                    children: (
                      <div>
                        <Space>
                          <Tag color={severityColors[event.severity]}>
                            {event.severity.toUpperCase()}
                          </Tag>
                          <Text strong>{event.type.replace('_', ' ').toUpperCase()}</Text>
                          {event.resolved && (
                            <Tag color="success" icon={<CheckCircleOutlined />}>
                              Resolved
                            </Tag>
                          )}
                        </Space>
                        <p style={{ margin: '8px 0' }}>{event.description}</p>
                        <Text type="secondary">
                          {new Date(event.created_at).toLocaleString()}
                          {event.user_name && ` • User: ${event.user_name}`}
                          {event.ip_address && ` • IP: ${event.ip_address}`}
                        </Text>
                      </div>
                    ),
                  }))}
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Detail Modal */}
      <Modal
        title="Audit Log Details"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalOpen(false)}>
            Close
          </Button>,
        ]}
        width={600}
      >
        {selectedLog && (
          <Descriptions column={1} bordered>
            <Descriptions.Item label="Time">
              {new Date(selectedLog.created_at).toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="User">{selectedLog.user_name}</Descriptions.Item>
            <Descriptions.Item label="Action">
              <Tag color={actionColors[selectedLog.action]}>{selectedLog.action}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="Resource">{selectedLog.resource}</Descriptions.Item>
            <Descriptions.Item label="IP Address">{selectedLog.ip_address}</Descriptions.Item>
            <Descriptions.Item label="User Agent">{selectedLog.user_agent}</Descriptions.Item>
            <Descriptions.Item label="Status">
              {selectedLog.status === 'success' ? (
                <Tag color="success">Success</Tag>
              ) : (
                <Tag color="error">Failed</Tag>
              )}
            </Descriptions.Item>
            {selectedLog.details && (
              <Descriptions.Item label="Details">{selectedLog.details}</Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default SecurityPage;
