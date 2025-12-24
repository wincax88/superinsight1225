// Quick actions component
import { Card, Button, Space } from 'antd';
import { PlusOutlined, DollarOutlined, DatabaseOutlined, SettingOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ROUTES } from '@/constants';

interface QuickAction {
  key: string;
  icon: React.ReactNode;
  label: string;
  path: string;
  color?: string;
}

const defaultActions: QuickAction[] = [
  {
    key: 'createTask',
    icon: <PlusOutlined />,
    label: 'quickActions.createTask',
    path: ROUTES.TASKS,
    color: '#1890ff',
  },
  {
    key: 'viewBilling',
    icon: <DollarOutlined />,
    label: 'quickActions.viewBilling',
    path: ROUTES.BILLING,
    color: '#52c41a',
  },
  {
    key: 'manageData',
    icon: <DatabaseOutlined />,
    label: 'quickActions.manageData',
    path: ROUTES.TASKS,
    color: '#faad14',
  },
  {
    key: 'settings',
    icon: <SettingOutlined />,
    label: 'quickActions.settings',
    path: ROUTES.SETTINGS,
    color: '#722ed1',
  },
];

interface QuickActionsProps {
  actions?: QuickAction[];
}

export const QuickActions: React.FC<QuickActionsProps> = ({ actions = defaultActions }) => {
  const navigate = useNavigate();
  const { t } = useTranslation('dashboard');

  return (
    <Card title={t('quickActions.title')}>
      <Space wrap>
        {actions.map((action) => (
          <Button
            key={action.key}
            icon={action.icon}
            onClick={() => navigate(action.path)}
            style={{ borderColor: action.color, color: action.color }}
          >
            {t(action.label)}
          </Button>
        ))}
      </Space>
    </Card>
  );
};
