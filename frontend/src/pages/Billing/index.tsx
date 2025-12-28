// Billing page
import { useState } from 'react';
import {
  Card,
  Table,
  Tag,
  Space,
  Button,
  Row,
  Col,
  Statistic,
  DatePicker,
  Modal,
  Descriptions,
  message,
} from 'antd';
import {
  DollarOutlined,
  DownloadOutlined,
  EyeOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useAuthStore } from '@/stores/authStore';
import type { BillingRecord, BillingStatus } from '@/types/billing';

const { RangePicker } = DatePicker;

const statusColorMap: Record<BillingStatus, string> = {
  pending: 'processing',
  paid: 'success',
  overdue: 'error',
  cancelled: 'default',
};

const statusTextMap: Record<BillingStatus, string> = {
  pending: 'Pending',
  paid: 'Paid',
  overdue: 'Overdue',
  cancelled: 'Cancelled',
};

// Mock data
const mockBillingRecords: BillingRecord[] = [
  {
    id: '1',
    tenant_id: 'tenant1',
    period_start: '2025-01-01',
    period_end: '2025-01-31',
    total_amount: 12500,
    status: 'paid',
    items: [
      { id: '1', description: 'Annotation Services', quantity: 5000, unit_price: 2, amount: 10000, category: 'annotation' },
      { id: '2', description: 'Data Storage', quantity: 100, unit_price: 25, amount: 2500, category: 'storage' },
    ],
    created_at: '2025-02-01T00:00:00Z',
    due_date: '2025-02-15T00:00:00Z',
    paid_at: '2025-02-10T10:30:00Z',
  },
  {
    id: '2',
    tenant_id: 'tenant1',
    period_start: '2024-12-01',
    period_end: '2024-12-31',
    total_amount: 9800,
    status: 'paid',
    items: [
      { id: '3', description: 'Annotation Services', quantity: 4000, unit_price: 2, amount: 8000, category: 'annotation' },
      { id: '4', description: 'API Calls', quantity: 1800, unit_price: 1, amount: 1800, category: 'api' },
    ],
    created_at: '2025-01-01T00:00:00Z',
    due_date: '2025-01-15T00:00:00Z',
    paid_at: '2025-01-12T14:20:00Z',
  },
  {
    id: '3',
    tenant_id: 'tenant1',
    period_start: '2025-02-01',
    period_end: '2025-02-28',
    total_amount: 15200,
    status: 'pending',
    items: [
      { id: '5', description: 'Annotation Services', quantity: 6000, unit_price: 2, amount: 12000, category: 'annotation' },
      { id: '6', description: 'Data Augmentation', quantity: 800, unit_price: 4, amount: 3200, category: 'augmentation' },
    ],
    created_at: '2025-03-01T00:00:00Z',
    due_date: '2025-03-15T00:00:00Z',
  },
];

const BillingPage: React.FC = () => {
  const { currentTenant } = useAuthStore();
  const [selectedRecord, setSelectedRecord] = useState<BillingRecord | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [loading] = useState(false);

  const handleExport = () => {
    message.success('Export started. The file will be downloaded shortly.');
  };

  const handleViewDetail = (record: BillingRecord) => {
    setSelectedRecord(record);
    setDetailModalOpen(true);
  };

  const columns: ColumnsType<BillingRecord> = [
    {
      title: 'Period',
      key: 'period',
      width: 200,
      render: (_, record) => (
        <span>
          {new Date(record.period_start).toLocaleDateString()} -{' '}
          {new Date(record.period_end).toLocaleDateString()}
        </span>
      ),
    },
    {
      title: 'Amount',
      dataIndex: 'total_amount',
      key: 'total_amount',
      width: 150,
      render: (amount: number) => (
        <span style={{ fontWeight: 600, color: '#1890ff' }}>
          ¥{amount.toLocaleString()}
        </span>
      ),
      sorter: (a, b) => a.total_amount - b.total_amount,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: BillingStatus) => (
        <Tag color={statusColorMap[status]}>{statusTextMap[status]}</Tag>
      ),
      filters: [
        { text: 'Pending', value: 'pending' },
        { text: 'Paid', value: 'paid' },
        { text: 'Overdue', value: 'overdue' },
        { text: 'Cancelled', value: 'cancelled' },
      ],
      onFilter: (value, record) => record.status === value,
    },
    {
      title: 'Due Date',
      dataIndex: 'due_date',
      key: 'due_date',
      width: 120,
      render: (date: string) => new Date(date).toLocaleDateString(),
      sorter: (a, b) => new Date(a.due_date).getTime() - new Date(b.due_date).getTime(),
    },
    {
      title: 'Paid At',
      dataIndex: 'paid_at',
      key: 'paid_at',
      width: 150,
      render: (date: string | undefined) =>
        date ? new Date(date).toLocaleString() : '-',
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      fixed: 'right',
      render: (_, record) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetail(record)}
        >
          View
        </Button>
      ),
    },
  ];

  // Calculate stats
  const totalSpending = mockBillingRecords.reduce((sum, r) => sum + r.total_amount, 0);
  const averageMonthly = totalSpending / mockBillingRecords.length;
  const pendingAmount = mockBillingRecords
    .filter((r) => r.status === 'pending')
    .reduce((sum, r) => sum + r.total_amount, 0);

  return (
    <div>
      {/* Stats Cards */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Spending"
              value={totalSpending}
              prefix={<DollarOutlined />}
              suffix="¥"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Average Monthly"
              value={averageMonthly}
              prefix={<RiseOutlined />}
              suffix="¥"
              precision={0}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Pending Payment"
              value={pendingAmount}
              prefix={<FallOutlined />}
              suffix="¥"
              valueStyle={{ color: pendingAmount > 0 ? '#faad14' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Billing Table */}
      <Card
        title="Billing Records"
        extra={
          <Space>
            <RangePicker />
            <Button icon={<DownloadOutlined />} onClick={handleExport}>
              Export
            </Button>
          </Space>
        }
      >
        <Table<BillingRecord>
          columns={columns}
          dataSource={mockBillingRecords}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `Total ${total} records`,
          }}
          scroll={{ x: 900 }}
        />
      </Card>

      {/* Detail Modal */}
      <Modal
        title="Billing Details"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalOpen(false)}>
            Close
          </Button>,
          <Button key="download" type="primary" icon={<DownloadOutlined />}>
            Download Invoice
          </Button>,
        ]}
        width={700}
      >
        {selectedRecord && (
          <>
            <Descriptions column={2} style={{ marginBottom: 24 }}>
              <Descriptions.Item label="Period">
                {new Date(selectedRecord.period_start).toLocaleDateString()} -{' '}
                {new Date(selectedRecord.period_end).toLocaleDateString()}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={statusColorMap[selectedRecord.status]}>
                  {statusTextMap[selectedRecord.status]}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Total Amount">
                <span style={{ fontWeight: 600, fontSize: 18, color: '#1890ff' }}>
                  ¥{selectedRecord.total_amount.toLocaleString()}
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="Due Date">
                {new Date(selectedRecord.due_date).toLocaleDateString()}
              </Descriptions.Item>
            </Descriptions>

            <Table
              size="small"
              dataSource={selectedRecord.items}
              rowKey="id"
              pagination={false}
              columns={[
                { title: 'Description', dataIndex: 'description', key: 'description' },
                { title: 'Category', dataIndex: 'category', key: 'category' },
                { title: 'Quantity', dataIndex: 'quantity', key: 'quantity' },
                {
                  title: 'Unit Price',
                  dataIndex: 'unit_price',
                  key: 'unit_price',
                  render: (v) => `¥${v}`,
                },
                {
                  title: 'Amount',
                  dataIndex: 'amount',
                  key: 'amount',
                  render: (v) => `¥${v.toLocaleString()}`,
                },
              ]}
            />
          </>
        )}
      </Modal>
    </div>
  );
};

export default BillingPage;
