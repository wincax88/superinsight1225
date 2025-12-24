// Main App component
import { ConfigProvider, App as AntApp, theme } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import zhCN from 'antd/locale/zh_CN';
import enUS from 'antd/locale/en_US';
import { useUIStore } from '@/stores/uiStore';
import { AppRouter } from '@/router';
import { ErrorBoundary } from '@/components/Common/ErrorBoundary';
import { THEMES } from '@/constants';

// Import i18n config
import '@/locales/config';

// Import global styles
import '@/styles/global.scss';

// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  const { theme: currentTheme, language } = useUIStore();

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        locale={language === 'zh' ? zhCN : enUS}
        theme={{
          algorithm: currentTheme === THEMES.DARK ? theme.darkAlgorithm : theme.defaultAlgorithm,
          token: {
            colorPrimary: '#1890ff',
            borderRadius: 6,
          },
        }}
      >
        <AntApp>
          <ErrorBoundary>
            <AppRouter />
          </ErrorBoundary>
        </AntApp>
      </ConfigProvider>
    </QueryClientProvider>
  );
}

export default App;
