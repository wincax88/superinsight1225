// Dashboard page
import { Row, Col, Typography, Alert } from 'antd';
import {
  FileTextOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
  DollarOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { MetricCard, TrendChart, QuickActions } from '@/components/Dashboard';
import { useDashboard } from '@/hooks/useDashboard';
import { useAuthStore } from '@/stores/authStore';

const { Title } = Typography;

const DashboardPage: React.FC = () => {
  const { t } = useTranslation('dashboard');
  const { user } = useAuthStore();
  const { annotationEfficiency, isLoading, error } = useDashboard();

  // Prepare chart data from annotation efficiency
  const chartData = annotationEfficiency?.trends?.map((trend) => ({
    timestamp: trend.timestamp,
    datetime: trend.datetime,
    value: trend.annotations_per_hour,
  })) || [];

  // Calculate mock metrics when no real data
  const metrics = {
    activeTasks: 12,
    todayAnnotations: 156,
    totalCorpus: 25000,
    totalBilling: 8500,
  };

  if (error) {
    return (
      <Alert
        type="warning"
        message="数据加载失败"
        description="无法连接到后端服务器。请确保后端服务正在运行。"
        showIcon
      />
    );
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>
        {t('welcome', { name: user?.username || 'User' })}
      </Title>

      {/* Metric Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            title={t('metrics.activeTasks')}
            value={metrics.activeTasks}
            icon={<FileTextOutlined />}
            color="#1890ff"
            loading={isLoading}
            trend={5.2}
            trendLabel="较昨日"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            title={t('metrics.todayAnnotations')}
            value={metrics.todayAnnotations}
            icon={<CheckCircleOutlined />}
            color="#52c41a"
            loading={isLoading}
            trend={12.5}
            trendLabel="较昨日"
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            title={t('metrics.totalCorpus')}
            value={metrics.totalCorpus.toLocaleString()}
            icon={<DatabaseOutlined />}
            color="#faad14"
            loading={isLoading}
          />
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <MetricCard
            title={t('metrics.totalBilling')}
            value={metrics.totalBilling.toLocaleString()}
            suffix="¥"
            icon={<DollarOutlined />}
            color="#722ed1"
            loading={isLoading}
            trend={-3.1}
            trendLabel="较上月"
          />
        </Col>
      </Row>

      {/* Charts */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <TrendChart
            title={t('charts.annotationTrend')}
            data={chartData.length > 0 ? chartData : generateMockChartData()}
            loading={isLoading}
            color="#1890ff"
            height={300}
          />
        </Col>
        <Col xs={24} lg={8}>
          <QuickActions />
        </Col>
      </Row>
    </div>
  );
};

// Generate mock chart data for demo
function generateMockChartData() {
  const now = Date.now();
  return Array.from({ length: 24 }, (_, i) => ({
    timestamp: now - (23 - i) * 3600000,
    datetime: new Date(now - (23 - i) * 3600000).toISOString(),
    value: Math.floor(Math.random() * 50) + 20,
  }));
}

export default DashboardPage;
