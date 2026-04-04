'use client'; // This is a client component

import React from 'react';
import { useAppStore } from '@/store/appStore';
import { 
  MenuIcon, 
  BellIcon, 
  SunIcon, 
  MoonIcon,
  UserIcon,
  SearchIcon
} from 'lucide-react';
import { cn } from '@/utils/cn';

/**
 * Header Component
 * 
 * Provides the top navigation bar with sidebar toggle, search, 
 * theme switching, notifications, and user profile summary.
 */
export const Header: React.FC = () => {
  const { sidebarOpen, toggleSidebar, theme, toggleTheme } = useAppStore();

  return (
    <header className="sticky top-0 z-40 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 transition-colors duration-200">
      <div className="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        
        {/* Left side: Navigation Toggle & Search */}
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleSidebar}
            className="p-2 -ml-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:hover:text-gray-300 dark:hover:bg-gray-800 lg:hidden focus:outline-none focus:ring-2 focus:ring-primary-500"
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
          >
            <MenuIcon className="h-6 w-6" />
          </button>

          <div className="relative hidden md:flex items-center max-w-sm w-full group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <SearchIcon className="h-4 w-4 text-gray-400 group-focus-within:text-primary-500 transition-colors" />
            </div>
            <input
              type="text"
              placeholder="Search experiments, videos, or jobs..."
              className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-xl leading-5 bg-gray-50 dark:bg-gray-800 dark:border-gray-700 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm transition-all"
            />
          </div>
        </div>

        {/* Right side: Actions & Profile */}
        <div className="flex items-center space-x-2 sm:space-x-4">
          
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl text-gray-500 hover:text-gray-700 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800 transition-all border border-transparent hover:border-gray-200 dark:hover:border-gray-700"
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? (
              <MoonIcon className="h-5 w-5" />
            ) : (
              <SunIcon className="h-5 w-5" />
            )}
          </button>

          {/* Notifications */}
          <button 
            className="p-2 rounded-xl text-gray-500 hover:text-gray-700 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-800 transition-all border border-transparent hover:border-gray-200 dark:hover:border-gray-700 relative"
            aria-label="View notifications"
          >
            <BellIcon className="h-5 w-5" />
            <span className="absolute top-2 right-2 h-2.5 w-2.5 bg-red-500 border-2 border-white dark:border-gray-900 rounded-full animate-pulse"></span>
          </button>

          {/* User profile */}
          <div className="flex items-center space-x-3 pl-2 sm:pl-4 border-l border-gray-200 dark:border-gray-800">
            <div className="hidden sm:block text-right">
              <p className="text-sm font-semibold text-gray-900 dark:text-white leading-none">
                System Administrator
              </p>
              <p className="text-[10px] uppercase tracking-wider font-bold text-primary-600 dark:text-primary-400 mt-1">
                Superuser Active
              </p>
            </div>
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-lg shadow-primary-500/20">
              <UserIcon className="h-5 w-5 text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
