// 500 Server Error page
import { Button, Result } from 'antd';
import { useNavigate } from 'react-router-dom';
import { ROUTES } from '@/constants';

const ServerErrorPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Result
        status="500"
        title="500"
        subTitle="抱歉，服务器发生错误。"
        extra={[
          <Button type="primary" key="home" onClick={() => navigate(ROUTES.HOME)}>
            返回首页
          </Button>,
          <Button key="retry" onClick={() => window.location.reload()}>
            刷新页面
          </Button>,
        ]}
      />
    </div>
  );
};

export default ServerErrorPage;
