// Trend chart component
import { Card } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import dayjs from 'dayjs';

interface ChartDataPoint {
  timestamp: number;
  datetime: string;
  value: number;
  label?: string;
}

interface TrendChartProps {
  title: string;
  data: ChartDataPoint[];
  loading?: boolean;
  color?: string;
  height?: number;
  showLegend?: boolean;
  valueFormatter?: (value: number) => string;
}

export const TrendChart: React.FC<TrendChartProps> = ({
  title,
  data,
  loading = false,
  color = '#1890ff',
  height = 300,
  showLegend = false,
  valueFormatter,
}) => {
  const formattedData = data.map((item) => ({
    ...item,
    time: dayjs(item.datetime).format('HH:mm'),
    formattedValue: valueFormatter ? valueFormatter(item.value) : item.value,
  }));

  return (
    <Card title={title} loading={loading}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip
            formatter={(value) =>
              valueFormatter && typeof value === 'number' ? valueFormatter(value) : value
            }
          />
          {showLegend && <Legend />}
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};
