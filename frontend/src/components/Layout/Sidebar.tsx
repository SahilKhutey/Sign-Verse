import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router'; // Reverted to next/router
import { useAppStore } from '@/store/appStore';
import {
  HomeIcon,
  UploadIcon,
  ActivityIcon,
  SettingsIcon,
  VideoIcon,
  BarChart3Icon
} from 'lucide-react';
import { cn } from '@/utils/cn';

const navigation = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Upload', href: '/upload', icon: UploadIcon },
  { name: 'Pose Estimation', href: '/pose', icon: ActivityIcon },
  { name: 'Simulation', href: '/simulation', icon: VideoIcon },
  { name: 'Analytics', href: '/analytics', icon: BarChart3Icon },
  { name: 'Settings', href: '/settings', icon: SettingsIcon },
];

export const Sidebar: React.FC = () => {
  const router = useRouter(); // Using useRouter instead of usePathname()
  const { currentPage } = useAppStore();

  return (
    <div className="flex h-full flex-col bg-white shadow-lg dark:bg-gray-900 overflow-y-auto">
      {/* Logo */}
      <div className="flex h-16 items-center justify-center border-b border-gray-200 px-4 dark:border-gray-800">
        <div className="flex items-center space-x-2">
          <ActivityIcon className="h-8 w-8 text-indigo-600" />
          <span className="text-xl font-bold text-gray-900 dark:text-white">
            SignVerse
          </span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => {
          const isActive = router.pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-100'
                  : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
              )}
            >
              <item.icon className={cn(
                'mr-3 h-5 w-5',
                isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-gray-400 group-hover:text-gray-500'
              )} />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="border-t border-gray-200 p-4 dark:border-gray-800">
        <div className="text-center text-xs text-gray-500 dark:text-gray-400">
          v1.0.0
        </div>
      </div>
    </div>
  );
};
