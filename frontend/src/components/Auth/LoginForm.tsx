// Login form component
import { useState } from 'react';
import { Form, Input, Button, Checkbox } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/hooks/useAuth';
import type { LoginCredentials } from '@/types';

interface LoginFormProps {
  onSuccess?: () => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess }) => {
  const { t } = useTranslation('auth');
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (values: LoginCredentials) => {
    setLoading(true);
    try {
      await login(values);
      onSuccess?.();
    } catch {
      // Error is handled in useAuth hook
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form
      name="login"
      initialValues={{ remember: true }}
      onFinish={handleSubmit}
      size="large"
      layout="vertical"
    >
      <Form.Item
        name="username"
        rules={[{ required: true, message: t('login.usernamePlaceholder') }]}
      >
        <Input prefix={<UserOutlined />} placeholder={t('login.usernamePlaceholder')} />
      </Form.Item>

      <Form.Item name="password" rules={[{ required: true, message: t('login.passwordPlaceholder') }]}>
        <Input.Password prefix={<LockOutlined />} placeholder={t('login.passwordPlaceholder')} />
      </Form.Item>

      <Form.Item>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Form.Item name="remember" valuePropName="checked" noStyle>
            <Checkbox>{t('login.rememberMe')}</Checkbox>
          </Form.Item>
          <a href="/forgot-password">{t('login.forgotPassword')}</a>
        </div>
      </Form.Item>

      <Form.Item>
        <Button type="primary" htmlType="submit" loading={loading} block>
          {t('login.submit')}
        </Button>
      </Form.Item>
    </Form>
  );
};
