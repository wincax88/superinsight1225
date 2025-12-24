// Login page
import { Card, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import { Navigate } from 'react-router-dom';
import { LoginForm } from '@/components/Auth/LoginForm';
import { useAuthStore } from '@/stores/authStore';
import { ROUTES } from '@/constants';
import styles from './style.module.scss';

const { Title, Text } = Typography;

const LoginPage: React.FC = () => {
  const { t } = useTranslation('auth');
  const { isAuthenticated } = useAuthStore();

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to={ROUTES.DASHBOARD} replace />;
  }

  return (
    <div className={styles.container}>
      <Card className={styles.card}>
        <div className={styles.header}>
          <img src="/logo.svg" alt="SuperInsight" className={styles.logo} />
          <Title level={2} className={styles.title}>
            SuperInsight
          </Title>
          <Text type="secondary">{t('login.subtitle')}</Text>
        </div>
        <LoginForm />
        <div className={styles.footer}>
          <a href={ROUTES.REGISTER}>{t('login.registerLink')}</a>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;
