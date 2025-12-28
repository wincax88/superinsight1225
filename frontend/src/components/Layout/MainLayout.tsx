// Main layout component
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { ProLayout } from '@ant-design/pro-components';
import {
  DashboardOutlined,
  OrderedListOutlined,
  DollarOutlined,
  SettingOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  SafetyCertificateOutlined,
  AuditOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuthStore } from '@/stores/authStore';
import { useUIStore } from '@/stores/uiStore';
import { HeaderContent } from './HeaderContent';
import { ROUTES } from '@/constants';

const menuItems = [
  {
    path: ROUTES.DASHBOARD,
    name: 'dashboard',
    icon: <DashboardOutlined />,
  },
  {
    path: ROUTES.TASKS,
    name: 'tasks',
    icon: <OrderedListOutlined />,
  },
  {
    path: ROUTES.AUGMENTATION,
    name: 'augmentation',
    icon: <ThunderboltOutlined />,
  },
  {
    path: ROUTES.QUALITY,
    name: 'quality',
    icon: <SafetyCertificateOutlined />,
  },
  {
    path: ROUTES.BILLING,
    name: 'billing',
    icon: <DollarOutlined />,
  },
  {
    path: ROUTES.SECURITY,
    name: 'security',
    icon: <AuditOutlined />,
    access: 'admin',
  },
  {
    path: ROUTES.SETTINGS,
    name: 'settings',
    icon: <SettingOutlined />,
  },
  {
    path: ROUTES.ADMIN,
    name: 'admin',
    icon: <SafetyOutlined />,
    access: 'admin',
  },
];

export const MainLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation('common');
  const { user } = useAuthStore();
  const { theme, sidebarCollapsed, toggleSidebar } = useUIStore();

  const filteredMenuItems = menuItems.filter((item) => {
    if (item.access === 'admin') {
      return user?.role === 'admin';
    }
    return true;
  });

  return (
    <ProLayout
      title="SuperInsight"
      logo="/logo.svg"
      navTheme={theme === 'dark' ? 'realDark' : 'light'}
      layout="mix"
      splitMenus={false}
      fixedHeader
      fixSiderbar
      collapsed={sidebarCollapsed}
      onCollapse={toggleSidebar}
      location={{ pathname: location.pathname }}
      route={{
        path: '/',
        routes: filteredMenuItems.map((item) => ({
          ...item,
          name: t(`menu.${item.name}`),
        })),
      }}
      menuItemRender={(item, dom) => (
        <div onClick={() => item.path && navigate(item.path)}>{dom}</div>
      )}
      headerContentRender={() => <HeaderContent />}
      contentStyle={{
        padding: 24,
        minHeight: 'calc(100vh - 56px)',
      }}
    >
      <Outlet />
    </ProLayout>
  );
};
