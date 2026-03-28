import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Users, Shield, Zap, TrendingUp, Clock } from 'lucide-react';
import { getBackendHealth, getDashboardJson, getInferenceHealth } from '../api/signverse';

const Dashboard = () => {
  const [backendHealth, setBackendHealth] = useState(null);
  const [inferenceHealth, setInferenceHealth] = useState(null);
  const [dashboardJson, setDashboardJson] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [dashboardError, setDashboardError] = useState(null);

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([getBackendHealth(), getInferenceHealth(), getDashboardJson()])
      .then(([b, i, d]) => {
        if (!mounted) return;
        if (b.status === 'fulfilled') setBackendHealth(b.value);
        if (i.status === 'fulfilled') setInferenceHealth(i.value);
        if (d.status === 'fulfilled') setDashboardJson(d.value);
        if (b.status === 'rejected' || i.status === 'rejected') {
          setHealthError('Health check failed');
        }
        if (d.status === 'rejected') {
          setDashboardError('Dashboard feed unavailable');
        }
      });
    return () => { mounted = false; };
  }, []);

  const statusColor = healthError ? 'text-amber-400' : 'text-emerald-400';
  const env = import.meta.env.MODE || 'development';
  const registrySnapshot = dashboardJson?.model_registry;
  const nlpEval = dashboardJson?.nlp_eval;

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-8"
    >
      <header className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold">Platform Overview</h1>
        <p className="text-slate-400">Welcome back! Here's your translation activity at a glance.</p>
        <div className="text-xs text-slate-500">Environment: {env}</div>
        {healthError && (
          <div className="text-sm text-amber-400">
            Health checks failed. Verify backend and inference services are running.
          </div>
        )}
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { label: 'Backend Status', value: backendHealth?.status || 'Unknown', icon: Activity, color: statusColor },
          { label: 'Inference Models', value: (inferenceHealth?.models?.length ?? 0).toString(), icon: Zap, color: 'text-amber-400' },
          { label: 'Active Users', value: '--', icon: Users, color: 'text-emerald-400' },
          { label: 'Secure Sessions', value: healthError ? 'Check' : 'OK', icon: Shield, color: 'text-indigo-400' },
        ].map((stat, i) => (
          <div key={i} className="glass-card p-6 flex flex-col gap-4">
            <div className="flex justify-between items-start">
              <span className="text-slate-400 text-sm font-medium">{stat.label}</span>
              <stat.icon className={stat.color} size={20} />
            </div>
            <p className="text-3xl font-bold">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="dashboard-grid">
        <div className="glass-card p-8 min-h-[400px]">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="text-indigo-400" size={24} /> 
            Usage Statistics
          </h3>
          <div className="w-full h-64 bg-slate-900/50 rounded-2xl border border-dashed border-slate-700 flex items-center justify-center">
            <p className="text-slate-500 italic">Historical data visualization placeholder</p>
          </div>
        </div>

        <div className="glass-card p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Clock className="text-indigo-400" size={24} />
            Recent Activity
          </h3>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex flex-col gap-1 p-3 bg-slate-900/30 rounded-xl border border-slate-800">
                <p className="text-sm font-medium">Session #{i*124}</p>
                <p className="text-xs text-slate-500">12 min ago - ASL to English</p>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-card p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Zap className="text-indigo-400" size={24} />
            Model Registry Snapshot
          </h3>
          {dashboardError ? (
            <p className="text-sm text-amber-400">{dashboardError}</p>
          ) : (
            <pre className="text-xs text-slate-200 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-auto max-h-64">
              {registrySnapshot ? JSON.stringify(registrySnapshot, null, 2) : 'No registry data yet.'}
            </pre>
          )}
        </div>

        <div className="glass-card p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="text-indigo-400" size={24} />
            Latest NLP Evaluation
          </h3>
          {dashboardError ? (
            <p className="text-sm text-amber-400">{dashboardError}</p>
          ) : (
            <pre className="text-xs text-slate-200 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-auto max-h-64">
              {nlpEval ? JSON.stringify(nlpEval, null, 2) : 'No evaluation report yet.'}
            </pre>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;
