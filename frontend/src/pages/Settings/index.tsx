// Settings page
import { useState } from 'react';
import {
  Card,
  Tabs,
  Form,
  Input,
  Button,
  Switch,
  Select,
  message,
  Divider,
  Row,
  Col,
  Avatar,
  Upload,
  Space,
} from 'antd';
import {
  UserOutlined,
  LockOutlined,
  BellOutlined,
  GlobalOutlined,
  UploadOutlined,
  SaveOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '@/stores/authStore';
import { useUIStore } from '@/stores/uiStore';

const { TabPane } = Tabs;
const { Option } = Select;

const SettingsPage: React.FC = () => {
  const { t, i18n } = useTranslation('common');
  const { user } = useAuthStore();
  const { theme, setTheme } = useUIStore();
  const [profileForm] = Form.useForm();
  const [passwordForm] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const handleProfileSubmit = async (values: Record<string, string>) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      message.success('Profile updated successfully');
      console.log('Profile values:', values);
    } catch {
      message.error('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordSubmit = async (values: Record<string, string>) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      message.success('Password changed successfully');
      passwordForm.resetFields();
      console.log('Password values:', values);
    } catch {
      message.error('Failed to change password');
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (lang: string) => {
    i18n.changeLanguage(lang);
    message.success(`Language changed to ${lang === 'zh' ? 'Chinese' : 'English'}`);
  };

  const handleThemeChange = (isDark: boolean) => {
    setTheme(isDark ? 'dark' : 'light');
    message.success(`Theme changed to ${isDark ? 'dark' : 'light'} mode`);
  };

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>Settings</h2>

      <Card>
        <Tabs defaultActiveKey="profile" tabPosition="left">
          {/* Profile Tab */}
          <TabPane
            tab={
              <span>
                <UserOutlined />
                Profile
              </span>
            }
            key="profile"
          >
            <h3>Profile Settings</h3>
            <Divider />

            <Row gutter={24}>
              <Col xs={24} md={8} style={{ textAlign: 'center', marginBottom: 24 }}>
                <Avatar size={120} icon={<UserOutlined />} />
                <div style={{ marginTop: 16 }}>
                  <Upload showUploadList={false}>
                    <Button icon={<UploadOutlined />}>Change Avatar</Button>
                  </Upload>
                </div>
              </Col>

              <Col xs={24} md={16}>
                <Form
                  form={profileForm}
                  layout="vertical"
                  initialValues={{
                    username: user?.username || '',
                    email: user?.email || '',
                    display_name: user?.username || '',
                  }}
                  onFinish={handleProfileSubmit}
                >
                  <Form.Item
                    name="username"
                    label="Username"
                    rules={[{ required: true, message: 'Please enter username' }]}
                  >
                    <Input disabled placeholder="Username" />
                  </Form.Item>

                  <Form.Item
                    name="email"
                    label="Email"
                    rules={[
                      { required: true, message: 'Please enter email' },
                      { type: 'email', message: 'Please enter a valid email' },
                    ]}
                  >
                    <Input placeholder="Email address" />
                  </Form.Item>

                  <Form.Item name="display_name" label="Display Name">
                    <Input placeholder="Display name" />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      icon={<SaveOutlined />}
                    >
                      Save Changes
                    </Button>
                  </Form.Item>
                </Form>
              </Col>
            </Row>
          </TabPane>

          {/* Security Tab */}
          <TabPane
            tab={
              <span>
                <LockOutlined />
                Security
              </span>
            }
            key="security"
          >
            <h3>Change Password</h3>
            <Divider />

            <Form
              form={passwordForm}
              layout="vertical"
              onFinish={handlePasswordSubmit}
              style={{ maxWidth: 400 }}
            >
              <Form.Item
                name="current_password"
                label="Current Password"
                rules={[{ required: true, message: 'Please enter current password' }]}
              >
                <Input.Password placeholder="Current password" />
              </Form.Item>

              <Form.Item
                name="new_password"
                label="New Password"
                rules={[
                  { required: true, message: 'Please enter new password' },
                  { min: 8, message: 'Password must be at least 8 characters' },
                ]}
              >
                <Input.Password placeholder="New password" />
              </Form.Item>

              <Form.Item
                name="confirm_password"
                label="Confirm Password"
                dependencies={['new_password']}
                rules={[
                  { required: true, message: 'Please confirm password' },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue('new_password') === value) {
                        return Promise.resolve();
                      }
                      return Promise.reject(new Error('Passwords do not match'));
                    },
                  }),
                ]}
              >
                <Input.Password placeholder="Confirm new password" />
              </Form.Item>

              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading}>
                  Change Password
                </Button>
              </Form.Item>
            </Form>
          </TabPane>

          {/* Notifications Tab */}
          <TabPane
            tab={
              <span>
                <BellOutlined />
                Notifications
              </span>
            }
            key="notifications"
          >
            <h3>Notification Preferences</h3>
            <Divider />

            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Email Notifications</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Receive email notifications for important updates
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch defaultChecked />
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Task Assignments</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Get notified when tasks are assigned to you
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch defaultChecked />
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Task Completions</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Get notified when your tasks are reviewed
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch defaultChecked />
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Billing Alerts</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Receive alerts about billing and payments
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch defaultChecked />
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>System Announcements</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Receive system maintenance and update notices
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch defaultChecked />
                </Col>
              </Row>
            </Space>
          </TabPane>

          {/* Appearance Tab */}
          <TabPane
            tab={
              <span>
                <GlobalOutlined />
                Appearance
              </span>
            }
            key="appearance"
          >
            <h3>Appearance Settings</h3>
            <Divider />

            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Language</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Select your preferred language
                    </p>
                  </div>
                </Col>
                <Col>
                  <Select
                    value={i18n.language}
                    onChange={handleLanguageChange}
                    style={{ width: 150 }}
                  >
                    <Option value="en">English</Option>
                    <Option value="zh">中文</Option>
                  </Select>
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Dark Mode</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Switch between light and dark themes
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch
                    checked={theme === 'dark'}
                    onChange={handleThemeChange}
                    checkedChildren="Dark"
                    unCheckedChildren="Light"
                  />
                </Col>
              </Row>

              <Row justify="space-between" align="middle">
                <Col>
                  <div>
                    <strong>Compact Mode</strong>
                    <p style={{ margin: 0, color: '#999' }}>
                      Use a more compact UI layout
                    </p>
                  </div>
                </Col>
                <Col>
                  <Switch />
                </Col>
              </Row>
            </Space>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default SettingsPage;
