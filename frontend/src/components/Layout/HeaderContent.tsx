// Header content component
import { Dropdown, Space, Button, Avatar, Switch } from 'antd';
import type { MenuProps } from 'antd';
import {
  UserOutlined,
  LogoutOutlined,
  SettingOutlined,
  GlobalOutlined,
  BulbOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/hooks/useAuth';
import { useUIStore } from '@/stores/uiStore';
import { THEMES } from '@/constants';

export const HeaderContent: React.FC = () => {
  const { t, i18n } = useTranslation('common');
  const { user, currentTenant, logout } = useAuth();
  const { theme, toggleTheme, setLanguage } = useUIStore();

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: t('menu.settings'),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: t('menu.settings'),
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      danger: true,
    },
  ];

  const languageMenuItems: MenuProps['items'] = [
    {
      key: 'zh',
      label: '中文',
    },
    {
      key: 'en',
      label: 'English',
    },
  ];

  const handleUserMenuClick: MenuProps['onClick'] = ({ key }) => {
    if (key === 'logout') {
      logout();
    }
  };

  const handleLanguageChange: MenuProps['onClick'] = ({ key }) => {
    i18n.changeLanguage(key);
    setLanguage(key as 'zh' | 'en');
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: 16,
        paddingRight: 24,
      }}
    >
      {/* Theme switch */}
      <Switch
        checkedChildren={<BulbOutlined />}
        unCheckedChildren={<BulbOutlined />}
        checked={theme === THEMES.DARK}
        onChange={toggleTheme}
      />

      {/* Language switch */}
      <Dropdown menu={{ items: languageMenuItems, onClick: handleLanguageChange }}>
        <Button type="text" icon={<GlobalOutlined />}>
          {i18n.language === 'zh' ? '中文' : 'EN'}
        </Button>
      </Dropdown>

      {/* Tenant info */}
      {currentTenant && (
        <span style={{ color: 'rgba(0, 0, 0, 0.45)' }}>{currentTenant.name}</span>
      )}

      {/* User dropdown */}
      <Dropdown menu={{ items: userMenuItems, onClick: handleUserMenuClick }}>
        <Space style={{ cursor: 'pointer' }}>
          <Avatar size="small" icon={<UserOutlined />} />
          <span>{user?.username || 'User'}</span>
        </Space>
      </Dropdown>
    </div>
  );
};
