import React from 'react';
import { useAppStore } from '@/store/appStore';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { sidebarOpen, theme } = useAppStore();

  return (
    <div className={`min-h-screen bg-gray-50 transition-colors duration-200 ${theme === 'dark' ? 'dark' : ''}`}>
      <div className="flex">
        {/* Sidebar Container */}
        <div
          className={`fixed inset-y-0 left-0 z-50 w-64 transform transition-transform duration-200 lg:static lg:translate-x-0 ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <Sidebar />
        </div>

        {/* Main content wrapper */}
        <div className="flex-1 lg:pl-0 flex flex-col min-w-0">
          <Header />
          <main className="p-6">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
};
