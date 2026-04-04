import React from 'react';
import { Layout } from '@/components/Layout/Layout';
// Using relative imports for now if the aliases aren't fully resolved, or @/ if defined
import { DashboardStats } from '@/components/Dashboard/DashboardStats';
import { RecentActivity } from '@/components/Dashboard/RecentActivity';
import { SystemHealth } from '@/components/Dashboard/SystemHealth';

/**
 * Dashboard Landing Page
 * 
 * Uses the Pages Router convention (src/pages/index.tsx).
 * Renders the main statistics and activity overview.
 */
export default function Dashboard() {
  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col space-y-1">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
            Dashboard
          </h1>
          <p className="text-base text-gray-500 dark:text-gray-400">
            Welcome to SignVerse &mdash; AI-Driven Motion Intelligence & Pose Analysis
          </p>
        </div>

        {/* Stats Grid */}
        <DashboardStats />

        {/* Main Dashboard Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* System Health Monitor */}
          <section aria-labelledby="health-heading">
            <SystemHealth />
          </section>

          {/* Activity Log */}
          <section aria-labelledby="activity-heading">
            <RecentActivity />
          </section>
        </div>
      </div>
    </Layout>
  );
}
