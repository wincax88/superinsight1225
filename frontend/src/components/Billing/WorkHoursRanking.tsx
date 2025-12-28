// Work hours ranking component
import { useState } from 'react';
import { Card, Table, Avatar, Tag, Space, Select, Button, Tooltip, Progress } from 'antd';
import {
  TrophyOutlined,
  UserOutlined,
  RiseOutlined,
  FallOutlined,
  GiftOutlined,
  ExportOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

interface RankingUser {
  rank: number;
  user_id: string;
  user_name: string;
  avatar?: string;
  total_hours: number;
  annotations_count: number;
  efficiency_score: number;
  quality_score: number;
  trend: 'up' | 'down' | 'stable';
  trend_value: number;
  bonus_eligible: boolean;
}

interface WorkHoursRankingProps {
  tenantId?: string;
  onExport?: () => void;
  onReward?: (userId: string, amount: number) => void;
}

// Mock data
const mockRankingData: RankingUser[] = [
  {
    rank: 1,
    user_id: '1',
    user_name: 'Alice Chen',
    total_hours: 168.5,
    annotations_count: 2450,
    efficiency_score: 98,
    quality_score: 96,
    trend: 'up',
    trend_value: 12,
    bonus_eligible: true,
  },
  {
    rank: 2,
    user_id: '2',
    user_name: 'Bob Wang',
    total_hours: 156.2,
    annotations_count: 2180,
    efficiency_score: 95,
    quality_score: 94,
    trend: 'up',
    trend_value: 8,
    bonus_eligible: true,
  },
  {
    rank: 3,
    user_id: '3',
    user_name: 'Carol Li',
    total_hours: 142.8,
    annotations_count: 1920,
    efficiency_score: 92,
    quality_score: 97,
    trend: 'stable',
    trend_value: 0,
    bonus_eligible: true,
  },
  {
    rank: 4,
    user_id: '4',
    user_name: 'David Zhang',
    total_hours: 128.5,
    annotations_count: 1680,
    efficiency_score: 88,
    quality_score: 91,
    trend: 'down',
    trend_value: -5,
    bonus_eligible: false,
  },
  {
    rank: 5,
    user_id: '5',
    user_name: 'Eva Liu',
    total_hours: 115.2,
    annotations_count: 1450,
    efficiency_score: 85,
    quality_score: 93,
    trend: 'up',
    trend_value: 3,
    bonus_eligible: false,
  },
];

const getRankIcon = (rank: number) => {
  const colors: Record<number, string> = {
    1: '#FFD700',
    2: '#C0C0C0',
    3: '#CD7F32',
  };
  if (rank <= 3) {
    return (
      <TrophyOutlined
        style={{ color: colors[rank], fontSize: 18, marginRight: 8 }}
      />
    );
  }
  return <span style={{ marginRight: 8, fontWeight: 600 }}>{rank}</span>;
};

const getTrendIcon = (trend: 'up' | 'down' | 'stable', value: number) => {
  if (trend === 'up') {
    return (
      <Tag color="success" icon={<RiseOutlined />}>
        +{value}%
      </Tag>
    );
  }
  if (trend === 'down') {
    return (
      <Tag color="error" icon={<FallOutlined />}>
        {value}%
      </Tag>
    );
  }
  return <Tag color="default">-</Tag>;
};

export const WorkHoursRanking: React.FC<WorkHoursRankingProps> = ({
  onExport,
  onReward,
}) => {
  const [period, setPeriod] = useState<'week' | 'month' | 'quarter'>('month');
  const [loading] = useState(false);

  const columns: ColumnsType<RankingUser> = [
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank',
      width: 80,
      render: (rank) => (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {getRankIcon(rank)}
        </div>
      ),
    },
    {
      title: 'User',
      dataIndex: 'user_name',
      key: 'user_name',
      render: (name, record) => (
        <Space>
          <Avatar src={record.avatar} icon={<UserOutlined />} size="small" />
          <span style={{ fontWeight: record.rank <= 3 ? 600 : 400 }}>{name}</span>
          {record.bonus_eligible && (
            <Tooltip title="Eligible for bonus">
              <GiftOutlined style={{ color: '#faad14' }} />
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: 'Hours',
      dataIndex: 'total_hours',
      key: 'total_hours',
      width: 100,
      render: (hours) => (
        <span style={{ fontWeight: 600 }}>{hours.toFixed(1)}h</span>
      ),
      sorter: (a, b) => a.total_hours - b.total_hours,
    },
    {
      title: 'Annotations',
      dataIndex: 'annotations_count',
      key: 'annotations_count',
      width: 120,
      render: (count) => count.toLocaleString(),
      sorter: (a, b) => a.annotations_count - b.annotations_count,
    },
    {
      title: 'Efficiency',
      dataIndex: 'efficiency_score',
      key: 'efficiency_score',
      width: 120,
      render: (score) => (
        <Progress
          percent={score}
          size="small"
          status={score >= 90 ? 'success' : score >= 70 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.efficiency_score - b.efficiency_score,
    },
    {
      title: 'Quality',
      dataIndex: 'quality_score',
      key: 'quality_score',
      width: 120,
      render: (score) => (
        <Progress
          percent={score}
          size="small"
          status={score >= 90 ? 'success' : score >= 70 ? 'normal' : 'exception'}
        />
      ),
      sorter: (a, b) => a.quality_score - b.quality_score,
    },
    {
      title: 'Trend',
      dataIndex: 'trend',
      key: 'trend',
      width: 100,
      render: (trend, record) => getTrendIcon(trend, record.trend_value),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_, record) =>
        record.bonus_eligible && (
          <Button
            type="link"
            size="small"
            icon={<GiftOutlined />}
            onClick={() => onReward?.(record.user_id, 100)}
          >
            Reward
          </Button>
        ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <TrophyOutlined style={{ color: '#faad14' }} />
          Work Hours Ranking
        </Space>
      }
      extra={
        <Space>
          <Select
            value={period}
            onChange={setPeriod}
            style={{ width: 120 }}
            options={[
              { value: 'week', label: 'This Week' },
              { value: 'month', label: 'This Month' },
              { value: 'quarter', label: 'This Quarter' },
            ]}
          />
          <Button icon={<ExportOutlined />} onClick={onExport}>
            Export
          </Button>
        </Space>
      }
    >
      <Table<RankingUser>
        columns={columns}
        dataSource={mockRankingData}
        rowKey="user_id"
        loading={loading}
        pagination={false}
        size="middle"
        rowClassName={(record) => (record.rank <= 3 ? 'highlight-row' : '')}
      />
    </Card>
  );
};
