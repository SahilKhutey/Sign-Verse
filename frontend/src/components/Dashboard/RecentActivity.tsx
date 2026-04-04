import React from 'react';
import { 
  UploadIcon, 
  ActivityIcon, 
  CheckCircleIcon, 
  XIcon,
  ClockIcon,
  ChevronRightIcon
} from 'lucide-react';
import { cn } from '@/utils/cn';

const activityItems = [
  {
    id: 1,
    type: 'upload',
    title: 'Video uploaded',
    description: 'sign_language_video.mp4',
    time: '2 minutes ago',
    status: 'completed',
    icon: UploadIcon,
    color: 'text-green-600',
    bgColor: 'bg-green-100 dark:bg-green-900/30'
  },
  {
    id: 2,
    type: 'processing',
    title: 'Pose estimation started',
    description: 'Job #1234',
    time: '5 minutes ago',
    status: 'processing',
    icon: ActivityIcon,
    color: 'text-blue-600',
    bgColor: 'bg-blue-100 dark:bg-blue-900/30'
  },
  {
    id: 3,
    type: 'completed',
    title: 'Simulation completed',
    description: '3D animation ready',
    time: '1 hour ago',
    status: 'completed',
    icon: CheckCircleIcon,
    color: 'text-green-600',
    bgColor: 'bg-green-100 dark:bg-green-900/30'
  },
  {
    id: 4,
    type: 'error',
    title: 'Processing failed',
    description: 'Invalid video format',
    time: '2 hours ago',
    status: 'failed',
    icon: XIcon,
    color: 'text-red-600',
    bgColor: 'bg-red-100 dark:bg-red-900/30'
  }
];

export const RecentActivity: React.FC = () => {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200 dark:bg-gray-800 dark:border-gray-700 overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center">
          <ClockIcon className="h-5 w-5 mr-2 text-primary-500" />
          Recent Activity
        </h3>
        <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">
          Real-time Log
        </span>
      </div>

      {/* Activity List */}
      <div className="flex-1 overflow-y-auto px-1">
        <div className="divide-y divide-gray-100 dark:divide-gray-700/50">
          {activityItems.map((item) => (
            <div 
              key={item.id} 
              className="group flex items-center space-x-4 p-4 hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-all duration-200 cursor-pointer"
            >
              <div className={cn(
                "p-2.5 rounded-xl transition-transform group-hover:scale-110",
                item.bgColor,
                item.color
              )}>
                <item.icon className="h-5 w-5" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                  {item.title}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">
                  {item.description}
                </p>
              </div>
              <div className="flex flex-col items-end space-y-1">
                <span className="text-[10px] font-medium text-gray-400 dark:text-gray-500 whitespace-nowrap">
                  {item.time}
                </span>
                <ChevronRightIcon className="h-4 w-4 text-gray-300 group-hover:text-primary-500 transition-colors" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer / CTA */}
      <div className="p-4 border-t border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50">
        <button className="w-full text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300 text-sm font-bold flex items-center justify-center group transition-colors">
          View Audit Log
          <ChevronRightIcon className="h-4 w-4 ml-1 transform group-hover:translate-x-1 transition-transform" />
        </button>
      </div>
    </div>
  );
};
