// Register form component
import { useState } from 'react';
import { Form, Input, Button, Select, Checkbox, message } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, TeamOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { authService } from '@/services/auth';
import { ROUTES } from '@/constants';

interface RegisterFormProps {
  onSuccess?: () => void;
}

interface RegisterFormValues {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  tenant_name?: string;
  tenant_type: 'new' | 'join';
  invite_code?: string;
  agreement: boolean;
}

export const RegisterForm: React.FC<RegisterFormProps> = ({ onSuccess }) => {
  const { t } = useTranslation('auth');
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [tenantType, setTenantType] = useState<'new' | 'join'>('new');

  const handleSubmit = async (values: RegisterFormValues) => {
    setLoading(true);
    try {
      await authService.register({
        username: values.username,
        email: values.email,
        password: values.password,
        tenant_name: values.tenant_type === 'new' ? values.tenant_name : undefined,
        invite_code: values.tenant_type === 'join' ? values.invite_code : undefined,
      });
      message.success(t('register.success'));
      onSuccess?.();
      navigate(ROUTES.LOGIN);
    } catch (error) {
      message.error(t('register.failed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form
      name="register"
      onFinish={handleSubmit}
      size="large"
      layout="vertical"
      initialValues={{ tenant_type: 'new', agreement: false }}
    >
      <Form.Item
        name="username"
        rules={[
          { required: true, message: t('register.usernameRequired') },
          { min: 3, max: 20, message: t('register.usernameLength') },
          { pattern: /^[a-zA-Z0-9_]+$/, message: t('register.usernamePattern') },
        ]}
      >
        <Input prefix={<UserOutlined />} placeholder={t('register.usernamePlaceholder')} />
      </Form.Item>

      <Form.Item
        name="email"
        rules={[
          { required: true, message: t('register.emailRequired') },
          { type: 'email', message: t('register.emailInvalid') },
        ]}
      >
        <Input prefix={<MailOutlined />} placeholder={t('register.emailPlaceholder')} />
      </Form.Item>

      <Form.Item
        name="password"
        rules={[
          { required: true, message: t('register.passwordRequired') },
          { min: 8, message: t('register.passwordLength') },
        ]}
      >
        <Input.Password prefix={<LockOutlined />} placeholder={t('register.passwordPlaceholder')} />
      </Form.Item>

      <Form.Item
        name="confirmPassword"
        dependencies={['password']}
        rules={[
          { required: true, message: t('register.confirmPasswordRequired') },
          ({ getFieldValue }) => ({
            validator(_, value) {
              if (!value || getFieldValue('password') === value) {
                return Promise.resolve();
              }
              return Promise.reject(new Error(t('register.passwordMismatch')));
            },
          }),
        ]}
      >
        <Input.Password
          prefix={<LockOutlined />}
          placeholder={t('register.confirmPasswordPlaceholder')}
        />
      </Form.Item>

      <Form.Item
        name="tenant_type"
        label={t('register.tenantType')}
        rules={[{ required: true }]}
      >
        <Select onChange={(value) => setTenantType(value)}>
          <Select.Option value="new">{t('register.createNewTenant')}</Select.Option>
          <Select.Option value="join">{t('register.joinExistingTenant')}</Select.Option>
        </Select>
      </Form.Item>

      {tenantType === 'new' ? (
        <Form.Item
          name="tenant_name"
          rules={[
            { required: true, message: t('register.tenantNameRequired') },
            { min: 2, max: 50, message: t('register.tenantNameLength') },
          ]}
        >
          <Input prefix={<TeamOutlined />} placeholder={t('register.tenantNamePlaceholder')} />
        </Form.Item>
      ) : (
        <Form.Item
          name="invite_code"
          rules={[{ required: true, message: t('register.inviteCodeRequired') }]}
        >
          <Input placeholder={t('register.inviteCodePlaceholder')} />
        </Form.Item>
      )}

      <Form.Item
        name="agreement"
        valuePropName="checked"
        rules={[
          {
            validator: (_, value) =>
              value
                ? Promise.resolve()
                : Promise.reject(new Error(t('register.agreementRequired'))),
          },
        ]}
      >
        <Checkbox>
          {t('register.agreementText')}{' '}
          <a href="/terms" target="_blank">
            {t('register.termsOfService')}
          </a>{' '}
          {t('register.and')}{' '}
          <a href="/privacy" target="_blank">
            {t('register.privacyPolicy')}
          </a>
        </Checkbox>
      </Form.Item>

      <Form.Item>
        <Button type="primary" htmlType="submit" loading={loading} block>
          {t('register.submit')}
        </Button>
      </Form.Item>
    </Form>
  );
};
