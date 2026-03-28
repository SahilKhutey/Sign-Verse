import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, NavLink } from 'react-router-dom';
import { Camera, History, Users, LayoutDashboard, Languages, User, LogOut } from 'lucide-react';
import { AnimatePresence } from 'framer-motion';

// Pages (to be created)
import LiveTranslator from './pages/LiveTranslator';
import HistoryPage from './pages/History';
import Collaboration from './pages/Collaboration';
import Dashboard from './pages/Dashboard';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <nav className="nav">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
              <Languages className="text-white" size={24} />
            </div>
            <h1 className="text-2xl font-bold gradient-text">SignVerse</h1>
          </div>

          <div className="flex items-center gap-8">
            <NavLink to="/" className={({isActive}) => `flex items-center gap-2 transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-400'}`}>
              <LayoutDashboard size={20} /> Dashboard
            </NavLink>
            <NavLink to="/translate" className={({isActive}) => `flex items-center gap-2 transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-400'}`}>
              <Camera size={20} /> Translate
            </NavLink>
            <NavLink to="/collaboration" className={({isActive}) => `flex items-center gap-2 transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-400'}`}>
              <Users size={20} /> Rooms
            </NavLink>
            <NavLink to="/history" className={({isActive}) => `flex items-center gap-2 transition-colors ${isActive ? 'text-indigo-400' : 'text-slate-400'}`}>
              <History size={20} /> History
            </NavLink>
          </div>

          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-slate-800 rounded-full flex items-center justify-center border border-slate-700">
              <User size={20} />
            </div>
            <button className="text-slate-400 hover:text-red-400">
              <LogOut size={20} />
            </button>
          </div>
        </nav>

        <main className="container flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/translate" element={<LiveTranslator />} />
            <Route path="/collaboration" element={<Collaboration />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
