import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Users, Shield, Zap, TrendingUp, Clock } from 'lucide-react';
import {
  getBackendHealth,
  getDashboardJson,
  getInferenceHealth,
  getTextGlossPipelineStatus,
  triggerTextGlossPipeline,
} from '../api/signverse';

const Dashboard = () => {
  const [backendHealth, setBackendHealth] = useState(null);
  const [inferenceHealth, setInferenceHealth] = useState(null);
  const [dashboardJson, setDashboardJson] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [dashboardError, setDashboardError] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [pipelineError, setPipelineError] = useState(null);
  const [triggeringPipeline, setTriggeringPipeline] = useState(false);

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

  useEffect(() => {
    let alive = true;
    const hasAdminToken = Boolean(import.meta.env.VITE_ADMIN_TOKEN);
    if (!hasAdminToken) {
      return () => { alive = false; };
    }

    const fetchStatus = () => {
      getTextGlossPipelineStatus()
        .then((data) => {
          if (!alive) return;
          setPipelineStatus(data);
          setPipelineError(null);
        })
        .catch((err) => {
          if (!alive) return;
          setPipelineError(err?.message || 'Failed to load pipeline status');
        });
    };

    fetchStatus();
    const timer = setInterval(fetchStatus, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const statusColor = healthError ? 'text-amber-400' : 'text-emerald-400';
  const env = import.meta.env.MODE || 'development';
  const hasAdminToken = Boolean(import.meta.env.VITE_ADMIN_TOKEN);
  const registrySnapshot = dashboardJson?.model_registry;
  const nlpEval = dashboardJson?.nlp_eval;
  const datasetReport = dashboardJson?.text_gloss_dataset;
  const datasetSummary = datasetReport
    ? {
        total: datasetReport.total_pairs ?? 0,
        train: datasetReport.train_pairs ?? 0,
        val: datasetReport.val_pairs ?? 0,
        test: datasetReport.test_pairs ?? 0,
        dedupeRemoved: datasetReport.dedupe_removed ?? 0,
        dedupeRate: datasetReport.dedupe_rate ?? 0,
        textLen: datasetReport.length_stats?.text,
        glossLen: datasetReport.length_stats?.gloss,
        warnings: datasetReport.warnings || [],
      }
    : null;

  const runPipeline = async () => {
    if (!hasAdminToken || triggeringPipeline) return;
    try {
      setTriggeringPipeline(true);
      setPipelineError(null);
      await triggerTextGlossPipeline({
        curriculum: true,
        augment: true,
        fail_on_warnings: true,
        register: true,
      });
      const status = await getTextGlossPipelineStatus();
      setPipelineStatus(status);
    } catch (err) {
      setPipelineError(err?.message || 'Failed to trigger pipeline');
    } finally {
      setTriggeringPipeline(false);
    }
  };

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

        <div className="glass-card p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="text-indigo-400" size={24} />
            Text-Gloss Dataset Report
          </h3>
          {dashboardError ? (
            <p className="text-sm text-amber-400">{dashboardError}</p>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Total Pairs</div>
                  <div className="text-xl font-semibold">{datasetSummary?.total ?? '--'}</div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Dedupe Removed</div>
                  <div className="text-xl font-semibold">{datasetSummary?.dedupeRemoved ?? '--'}</div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Dedupe Rate</div>
                  <div className="text-xl font-semibold">
                    {datasetSummary ? `${Math.round(datasetSummary.dedupeRate * 100)}%` : '--'}
                  </div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Train / Val / Test</div>
                  <div className="text-sm text-slate-200">
                    {datasetSummary ? `${datasetSummary.train} / ${datasetSummary.val} / ${datasetSummary.test}` : '--'}
                  </div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Avg Lengths</div>
                  <div className="text-sm text-slate-200">
                    {datasetSummary && datasetSummary.textLen && datasetSummary.glossLen
                      ? `Text ${datasetSummary.textLen.avg} | Gloss ${datasetSummary.glossLen.avg}`
                      : '--'}
                  </div>
                </div>
              </div>
              {datasetSummary && datasetSummary.warnings.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 text-xs text-amber-200">
                  Warnings: {datasetSummary.warnings.join(' | ')}
                </div>
              )}
              <pre className="text-xs text-slate-200 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-auto max-h-64">
                {datasetReport ? JSON.stringify(datasetReport, null, 2) : 'No dataset report yet.'}
              </pre>
            </div>
          )}
        </div>

        <div className="glass-card p-8">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Shield className="text-indigo-400" size={24} />
            Admin Pipeline Control
          </h3>
          {!hasAdminToken ? (
            <p className="text-sm text-amber-400">
              Set <code>VITE_ADMIN_TOKEN</code> to enable trigger and status polling.
            </p>
          ) : (
            <div className="space-y-4">
              <p className="text-xs text-slate-400">
                Internal admin feature. Use only in trusted environments.
              </p>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">State</div>
                  <div className="text-lg font-semibold">{pipelineStatus?.state || 'unknown'}</div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Run ID</div>
                  <div className="text-sm text-slate-200">{pipelineStatus?.run_id || '--'}</div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Started</div>
                  <div className="text-sm text-slate-200">{pipelineStatus?.started_at || '--'}</div>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3">
                  <div className="text-xs text-slate-400">Finished</div>
                  <div className="text-sm text-slate-200">{pipelineStatus?.finished_at || '--'}</div>
                </div>
              </div>
              {pipelineError && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 text-xs text-amber-200">
                  {pipelineError}
                </div>
              )}
              <div className="flex gap-3">
                <button
                  className="btn-primary"
                  disabled={triggeringPipeline || pipelineStatus?.state === 'running' || pipelineStatus?.state === 'queued'}
                  onClick={runPipeline}
                >
                  {triggeringPipeline ? 'Queueing...' : 'Run Text-Gloss Pipeline'}
                </button>
              </div>
              {pipelineStatus?.log_tail && (
                <pre className="text-xs text-slate-200 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-auto max-h-64">
                  {pipelineStatus.log_tail}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;
