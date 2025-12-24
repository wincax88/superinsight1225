// Loading component
import { Spin } from 'antd';
import { useTranslation } from 'react-i18next';

interface LoadingProps {
  tip?: string;
  fullScreen?: boolean;
}

export const Loading: React.FC<LoadingProps> = ({ tip, fullScreen = false }) => {
  const { t } = useTranslation('common');

  const spinElement = <Spin size="large" tip={tip || t('status.loading')} />;

  if (fullScreen) {
    return (
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          zIndex: 9999,
        }}
      >
        {spinElement}
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 0',
      }}
    >
      {spinElement}
    </div>
  );
};
