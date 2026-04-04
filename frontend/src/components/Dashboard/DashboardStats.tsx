import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { 
  VideoIcon, 
  ActivityIcon, 
  ClockIcon, 
  CheckCircleIcon,
  TrendingUpIcon, // Added for visual indicators
  TrendingDownIcon
} from 'lucide-react';
import { cn } from '@/utils/cn';

const StatCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: string;
  trendPositive?: boolean;
}> = ({ title, value, icon, trend, trendPositive }) => (
  <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-200 dark:bg-gray-800 dark:border-gray-700 hover:shadow-md transition-shadow duration-300">
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider dark:text-gray-400">
          {title}
        </p>
        <div className="flex items-baseline space-x-2 mt-2">
          <p className="text-3xl font-extrabold text-gray-900 dark:text-white">
            {value}
          </p>
          {trend && (
            <div className={cn(
              "flex items-center text-xs font-bold px-2 py-0.5 rounded-full",
              trendPositive 
                ? 'text-green-700 bg-green-100 dark:bg-green-900/30 dark:text-green-400' 
                : 'text-gray-500 bg-gray-100 dark:bg-gray-700 dark:text-gray-400'
            )}>
              {trendPositive ? <TrendingUpIcon className="h-3 w-3 mr-1" /> : null}
              {trend}
            </div>
          )}
        </div>
      </div>
      <div className="p-4 bg-primary-50 rounded-2xl dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">
        {icon}
      </div>
    </div>
  </div>
);

export const DashboardStats: React.FC = () => {
  const { data: health } = useQuery({
    queryKey: ['system-health'], // Fixed key to match useHealth hook
    queryFn: () => apiClient.health(),
    refetchInterval: 30000,
  });

  // Mock data - integrated with partial real data from health
  const stats = {
    totalVideos: 42,
    processingJobs: 3,
    completedJobs: 156,
    uptime: health?.uptime ? Math.floor(health.uptime / 3600) : 0,
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        title="Total Videos"
        value={stats.totalVideos}
        icon={<VideoIcon className="h-7 w-7" />}
        trend="+12%"
        trendPositive={true}
      />
      <StatCard
        title="Active Jobs"
        value={stats.processingJobs}
        icon={<ActivityIcon className="h-7 w-7" />}
        trend="3 Active"
      />
      <StatCard
        title="Completed"
        value={stats.completedJobs}
        icon={<CheckCircleIcon className="h-7 w-7" />}
        trend="98% Success"
        trendPositive={true}
      />
      <StatCard
        title="Uptime"
        value={`${stats.uptime}h`}
        icon={<ClockIcon className="h-7 w-7" />}
        trend="99.9%"
        trendPositive={true}
      />
    </div>
  );
};
