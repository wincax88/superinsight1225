// Metric card component
import { Card, Statistic, Tooltip } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons';
import type { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: number | string;
  suffix?: string;
  prefix?: ReactNode;
  trend?: number;
  trendLabel?: string;
  loading?: boolean;
  color?: string;
  icon?: ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  suffix,
  prefix,
  trend,
  trendLabel,
  loading = false,
  color,
  icon,
}) => {
  const getTrendIcon = () => {
    if (trend === undefined || trend === 0) {
      return <MinusOutlined style={{ color: '#999' }} />;
    }
    if (trend > 0) {
      return <ArrowUpOutlined style={{ color: '#52c41a' }} />;
    }
    return <ArrowDownOutlined style={{ color: '#ff4d4f' }} />;
  };

  const getTrendColor = () => {
    if (trend === undefined || trend === 0) return '#999';
    return trend > 0 ? '#52c41a' : '#ff4d4f';
  };

  return (
    <Card loading={loading}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <Statistic
          title={title}
          value={value}
          suffix={suffix}
          prefix={prefix}
          valueStyle={{ color }}
        />
        {icon && (
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: '50%',
              backgroundColor: color ? `${color}20` : '#f0f0f0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 24,
              color: color || '#1890ff',
            }}
          >
            {icon}
          </div>
        )}
      </div>
      {trend !== undefined && (
        <Tooltip title={trendLabel}>
          <div style={{ marginTop: 8, color: getTrendColor() }}>
            {getTrendIcon()}
            <span style={{ marginLeft: 4 }}>
              {Math.abs(trend).toFixed(1)}%
            </span>
          </div>
        </Tooltip>
      )}
    </Card>
  );
};
