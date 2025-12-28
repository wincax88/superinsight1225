// Register page
import { Card, Typography } from 'antd';
import { useTranslation } from 'react-i18next';
import { Navigate, Link } from 'react-router-dom';
import { RegisterForm } from '@/components/Auth/RegisterForm';
import { useAuthStore } from '@/stores/authStore';
import { ROUTES } from '@/constants';
import styles from '../Login/style.module.scss';

const { Title, Text } = Typography;

const RegisterPage: React.FC = () => {
  const { t } = useTranslation('auth');
  const { isAuthenticated } = useAuthStore();

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to={ROUTES.DASHBOARD} replace />;
  }

  return (
    <div className={styles.container}>
      <Card className={styles.card} style={{ maxWidth: 480 }}>
        <div className={styles.header}>
          <img src="/logo.svg" alt="SuperInsight" className={styles.logo} />
          <Title level={2} className={styles.title}>
            {t('register.title')}
          </Title>
          <Text type="secondary">{t('register.subtitle')}</Text>
        </div>
        <RegisterForm />
        <div className={styles.footer}>
          <Text type="secondary">{t('register.haveAccount')} </Text>
          <Link to={ROUTES.LOGIN}>{t('register.loginLink')}</Link>
        </div>
      </Card>
    </div>
  );
};

export default RegisterPage;
