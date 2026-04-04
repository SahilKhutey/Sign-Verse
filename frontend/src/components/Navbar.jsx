import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Home, Camera, Settings, History, Users, Layers } from 'lucide-react';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Dashboard' },
    { path: '/translate', icon: Camera, label: 'Translator' },
    { path: '/ar-dashboard', icon: Layers, label: 'AR Mode' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <nav className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex items-center justify-between relative z-50">
      <div className="flex items-center gap-8">
        <div 
          className="text-xl font-bold text-primary cursor-pointer tracking-tighter"
          onClick={() => navigate('/')}
        >
          SIGNVERSE <span className="text-white/50 text-xs font-light">AI</span>
        </div>
        
        <div className="flex items-center gap-1">
          {navItems.map((item) => (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all ${
                location.pathname === item.path
                  ? 'bg-primary/10 text-primary shadow-[0_0_15px_rgba(99,102,241,0.1)]'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <item.icon size={18} />
              <span className="text-sm font-bold uppercase tracking-wide">{item.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex flex-col items-end mr-2">
            <span className="text-[10px] text-slate-500 font-mono">CONNECTION: SECURE</span>
            <span className="text-[10px] text-green-500 font-mono">LATENCY: 12MS</span>
        </div>
        <div className="h-10 w-10 rounded-full border border-primary/30 p-1 flex items-center justify-center">
             <div className="h-full w-full rounded-full bg-gradient-to-tr from-primary to-indigo-400" />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
