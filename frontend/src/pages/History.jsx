import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, Search, Filter, Download } from 'lucide-react';
import { getHistory } from '../api/signverse';

const HistoryPage = () => {
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('signverse_token');
    const userId = localStorage.getItem('signverse_user_id');
    if (!token || !userId) {
      setError('Missing token or user id in localStorage (signverse_token, signverse_user_id).');
      return;
    }

    getHistory(userId, token)
      .then(setHistory)
      .catch(() => setError('Failed to load history'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-card p-8"
    >
      <div className="flex justify-between items-center mb-10">
        <h2 className="text-3xl font-bold">Translation History</h2>
        <div className="flex gap-4">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-700 rounded-xl px-4 py-2">
            <Search size={18} className="text-slate-500" />
            <input type="text" placeholder="Search logs..." className="bg-transparent outline-none w-48 text-sm" />
          </div>
          <button className="flex items-center gap-2 bg-slate-900 border border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors">
            <Filter size={18} /> Filter
          </button>
        </div>
      </div>

      {loading && (
        <div className="text-sm text-slate-400 mb-6">Loading history...</div>
      )}
      {error && (
        <div className="text-sm text-slate-400 mb-6">{error}</div>
      )}

      <div className="space-y-6">
        {(history.length ? history : [1, 2, 3, 4, 5, 6]).map((item, i) => (
          <div key={i} className="group flex items-center justify-between p-6 bg-slate-900/40 rounded-3xl border border-slate-800 hover:border-indigo-500/50 transition-all hover:bg-slate-900/60">
            <div className="flex gap-6 items-center">
              <div className="w-14 h-14 bg-indigo-500/10 rounded-2xl flex items-center justify-center group-hover:bg-indigo-500/20 transition-colors">
                <Camera className="text-indigo-400" size={24} />
              </div>
              <div>
                <p className="text-lg font-semibold">ASL Translation Record #{item?.id ?? i+1024}</p>
                <p className="text-sm text-slate-500 flex items-center gap-2">
                  Today • <span className="text-indigo-400/80">English (US)</span>
                </p>
              </div>
            </div>
            <div className="text-right flex items-center gap-8">
              <div>
                <p className="text-lg text-indigo-200 font-medium">"{item?.input_text ?? 'How can I help you today?'}"</p>
                <p className="text-xs text-slate-500">Type: {item?.translation_type ?? 'sign_to_text'}</p>
              </div>
              <button className="p-3 bg-slate-800 rounded-xl text-slate-400 hover:text-white transition-colors">
                <Download size={20} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
};

export default HistoryPage;
