// Admin console page
import { useState } from 'react';
import {
  Card,
  Tabs,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  message,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
} from 'antd';
import {
  TeamOutlined,
  UserOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { useAuthStore } from '@/stores/authStore';

interface Tenant {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'suspended';
  plan: 'free' | 'pro' | 'enterprise';
  users_count: number;
  storage_used: number;
  storage_limit: number;
  created_at: string;
}

interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'manager' | 'annotator';
  status: 'active' | 'inactive';
  tenant_id: string;
  last_login?: string;
  created_at: string;
}

// Mock data
const mockTenants: Tenant[] = [
  {
    id: '1',
    name: 'Acme Corporation',
    status: 'active',
    plan: 'enterprise',
    users_count: 25,
    storage_used: 45,
    storage_limit: 100,
    created_at: '2024-01-15',
  },
  {
    id: '2',
    name: 'Tech Startup',
    status: 'active',
    plan: 'pro',
    users_count: 8,
    storage_used: 12,
    storage_limit: 50,
    created_at: '2024-06-20',
  },
  {
    id: '3',
    name: 'Research Lab',
    status: 'inactive',
    plan: 'free',
    users_count: 3,
    storage_used: 2,
    storage_limit: 10,
    created_at: '2025-01-05',
  },
];

const mockUsers: User[] = [
  {
    id: '1',
    username: 'admin',
    email: 'admin@example.com',
    role: 'admin',
    status: 'active',
    tenant_id: '1',
    last_login: '2025-01-20T10:30:00Z',
    created_at: '2024-01-15',
  },
  {
    id: '2',
    username: 'john.doe',
    email: 'john@example.com',
    role: 'manager',
    status: 'active',
    tenant_id: '1',
    last_login: '2025-01-19T14:20:00Z',
    created_at: '2024-03-10',
  },
  {
    id: '3',
    username: 'jane.smith',
    email: 'jane@example.com',
    role: 'annotator',
    status: 'active',
    tenant_id: '1',
    last_login: '2025-01-20T09:15:00Z',
    created_at: '2024-05-22',
  },
  {
    id: '4',
    username: 'bob.wilson',
    email: 'bob@example.com',
    role: 'annotator',
    status: 'inactive',
    tenant_id: '2',
    created_at: '2024-08-14',
  },
];

const statusColors = {
  active: 'success',
  inactive: 'default',
  suspended: 'error',
} as const;

const planColors = {
  free: 'default',
  pro: 'blue',
  enterprise: 'gold',
} as const;

const roleColors = {
  admin: 'red',
  manager: 'orange',
  annotator: 'blue',
} as const;

const AdminPage: React.FC = () => {
  const { user } = useAuthStore();
  const [tenantModalOpen, setTenantModalOpen] = useState(false);
  const [userModalOpen, setUserModalOpen] = useState(false);
  const [tenantForm] = Form.useForm();
  const [userForm] = Form.useForm();

  // Check admin access
  if (user?.role !== 'admin') {
    return (
      <Alert
        type="error"
        message="Access Denied"
        description="You don't have permission to access the admin console."
        showIcon
      />
    );
  }

  const handleCreateTenant = async (values: Partial<Tenant>) => {
    message.success(`Tenant "${values.name}" created successfully`);
    setTenantModalOpen(false);
    tenantForm.resetFields();
  };

  const handleCreateUser = async (values: Partial<User>) => {
    message.success(`User "${values.username}" created successfully`);
    setUserModalOpen(false);
    userForm.resetFields();
  };

  const handleDeleteTenant = (id: string) => {
    Modal.confirm({
      title: 'Delete Tenant',
      icon: <ExclamationCircleOutlined />,
      content: 'Are you sure you want to delete this tenant? This action cannot be undone.',
      okType: 'danger',
      onOk: () => {
        message.success('Tenant deleted successfully');
      },
    });
  };

  const handleDeleteUser = (id: string) => {
    Modal.confirm({
      title: 'Delete User',
      icon: <ExclamationCircleOutlined />,
      content: 'Are you sure you want to delete this user?',
      okType: 'danger',
      onOk: () => {
        message.success('User deleted successfully');
      },
    });
  };

  const tenantColumns: ColumnsType<Tenant> = [
    { title: 'Name', dataIndex: 'name', key: 'name' },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Plan',
      dataIndex: 'plan',
      key: 'plan',
      render: (plan: keyof typeof planColors) => (
        <Tag color={planColors[plan]}>{plan.toUpperCase()}</Tag>
      ),
    },
    { title: 'Users', dataIndex: 'users_count', key: 'users_count' },
    {
      title: 'Storage',
      key: 'storage',
      render: (_, record) => (
        <Progress
          percent={Math.round((record.storage_used / record.storage_limit) * 100)}
          size="small"
          format={() => `${record.storage_used}/${record.storage_limit} GB`}
        />
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button type="link" icon={<EditOutlined />} size="small">
            Edit
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            size="small"
            onClick={() => handleDeleteTenant(record.id)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  const userColumns: ColumnsType<User> = [
    { title: 'Username', dataIndex: 'username', key: 'username' },
    { title: 'Email', dataIndex: 'email', key: 'email' },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      render: (role: keyof typeof roleColors) => (
        <Tag color={roleColors[role]}>{role.toUpperCase()}</Tag>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status) =>
        status === 'active' ? (
          <Tag icon={<CheckCircleOutlined />} color="success">
            Active
          </Tag>
        ) : (
          <Tag icon={<CloseCircleOutlined />} color="default">
            Inactive
          </Tag>
        ),
    },
    {
      title: 'Last Login',
      dataIndex: 'last_login',
      key: 'last_login',
      render: (date) => (date ? new Date(date).toLocaleString() : '-'),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button type="link" icon={<EditOutlined />} size="small">
            Edit
          </Button>
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
            size="small"
            onClick={() => handleDeleteUser(record.id)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>Admin Console</h2>

      {/* Stats Overview */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Tenants"
              value={mockTenants.length}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Users"
              value={mockUsers.length}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Active Users"
              value={mockUsers.filter((u) => u.status === 'active').length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Main Tabs */}
      <Card>
        <Tabs
          defaultActiveKey="tenants"
          items={[
            {
              key: 'tenants',
              label: (
                <span>
                  <TeamOutlined />
                  Tenants
                </span>
              ),
              children: (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => setTenantModalOpen(true)}
                    >
                      Create Tenant
                    </Button>
                  </div>
                  <Table
                    columns={tenantColumns}
                    dataSource={mockTenants}
                    rowKey="id"
                    pagination={{ pageSize: 10 }}
                  />
                </>
              ),
            },
            {
              key: 'users',
              label: (
                <span>
                  <UserOutlined />
                  Users
                </span>
              ),
              children: (
                <>
                  <div style={{ marginBottom: 16 }}>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => setUserModalOpen(true)}
                    >
                      Create User
                    </Button>
                  </div>
                  <Table
                    columns={userColumns}
                    dataSource={mockUsers}
                    rowKey="id"
                    pagination={{ pageSize: 10 }}
                  />
                </>
              ),
            },
            {
              key: 'settings',
              label: (
                <span>
                  <SettingOutlined />
                  System Settings
                </span>
              ),
              children: (
                <Alert
                  type="info"
                  message="System Settings"
                  description="System configuration options will be available here."
                  showIcon
                />
              ),
            },
          ]}
        />
      </Card>

      {/* Create Tenant Modal */}
      <Modal
        title="Create Tenant"
        open={tenantModalOpen}
        onCancel={() => setTenantModalOpen(false)}
        onOk={() => tenantForm.submit()}
      >
        <Form form={tenantForm} layout="vertical" onFinish={handleCreateTenant}>
          <Form.Item
            name="name"
            label="Tenant Name"
            rules={[{ required: true, message: 'Please enter tenant name' }]}
          >
            <Input placeholder="Enter tenant name" />
          </Form.Item>
          <Form.Item
            name="plan"
            label="Plan"
            rules={[{ required: true }]}
            initialValue="free"
          >
            <Select>
              <Select.Option value="free">Free</Select.Option>
              <Select.Option value="pro">Pro</Select.Option>
              <Select.Option value="enterprise">Enterprise</Select.Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* Create User Modal */}
      <Modal
        title="Create User"
        open={userModalOpen}
        onCancel={() => setUserModalOpen(false)}
        onOk={() => userForm.submit()}
      >
        <Form form={userForm} layout="vertical" onFinish={handleCreateUser}>
          <Form.Item
            name="username"
            label="Username"
            rules={[{ required: true, message: 'Please enter username' }]}
          >
            <Input placeholder="Enter username" />
          </Form.Item>
          <Form.Item
            name="email"
            label="Email"
            rules={[
              { required: true, message: 'Please enter email' },
              { type: 'email', message: 'Please enter a valid email' },
            ]}
          >
            <Input placeholder="Enter email" />
          </Form.Item>
          <Form.Item
            name="role"
            label="Role"
            rules={[{ required: true }]}
            initialValue="annotator"
          >
            <Select>
              <Select.Option value="admin">Admin</Select.Option>
              <Select.Option value="manager">Manager</Select.Option>
              <Select.Option value="annotator">Annotator</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="tenant_id"
            label="Tenant"
            rules={[{ required: true, message: 'Please select tenant' }]}
          >
            <Select placeholder="Select tenant">
              {mockTenants.map((t) => (
                <Select.Option key={t.id} value={t.id}>
                  {t.name}
                </Select.Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AdminPage;
