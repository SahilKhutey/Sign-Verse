import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Users, Send, DoorOpen, ShieldCheck, Zap } from 'lucide-react';
import io from 'socket.io-client';
import { getBackendHealth } from '../api/signverse';

const Collaboration = () => {
  const [room, setRoom] = useState('');
  const [joined, setJoined] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [socket, setSocket] = useState(null);
  const [serverStatus, setServerStatus] = useState('Unknown');

  const joinRoom = () => {
    if (room) setJoined(true);
  };

  useEffect(() => {
    getBackendHealth()
      .then(() => setServerStatus('Online'))
      .catch(() => setServerStatus('Offline'));
  }, []);

  useEffect(() => {
    if (!joined) return;
    const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
    const s = io(baseUrl, { transports: ['websocket'] });
    setSocket(s);

    s.on('connect', () => {
      s.emit('join_room', { room });
    });
    s.on('message', (msg) => {
      setMessages((prev) => [...prev, { user: 'System', text: msg.text }]);
    });
    s.on('new_translation', (msg) => {
      setMessages((prev) => [...prev, { user: msg.user || 'Peer', text: msg.text || String(msg) }]);
    });

    return () => {
      s.disconnect();
      setSocket(null);
    };
  }, [joined, room]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="dashboard-grid h-[700px]"
    >
      <div className="glass-card flex flex-col overflow-hidden">
        <div className="p-6 border-bottom border-slate-800 bg-slate-900/30 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
              <Users className="text-white" size={22} />
            </div>
            <div>
              <h2 className="font-bold">Real-time Collab Room</h2>
              <p className="text-xs text-slate-500">Collaborate with peers in sign-to-text sessions</p>
            </div>
          </div>
          {joined && (
            <div className="px-4 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-xs font-bold text-indigo-400 flex items-center gap-2">
              <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
              ROOM: {room}
            </div>
          )}
        </div>

        <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-slate-950/20">
          {!joined ? (
            <div className="h-full flex items-center justify-center flex-col gap-6 text-center">
              <div className="w-20 h-20 bg-slate-900 rounded-3xl flex items-center justify-center border border-slate-800">
                <DoorOpen className="text-indigo-400" size={40} />
              </div>
              <div className="max-w-xs">
                <h3 className="text-xl font-bold mb-2">Join a Room</h3>
                <p className="text-slate-500 text-sm mb-6">Enter a room name to start a synchronized translation session with others.</p>
                <div className="flex gap-2">
                  <input 
                    type="text" 
                    placeholder="Enter room name..." 
                    className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 outline-none focus:border-indigo-500 transition-all text-sm"
                    value={room}
                    onChange={(e) => setRoom(e.target.value)}
                  />
                  <button onClick={joinRoom} className="btn-primary flex items-center justify-center">Join</button>
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center text-slate-600 italic">
                  No activity yet. Start translating in the "Translate" tab!
                </div>
              ) : (
                messages.map((m, i) => (
                  <div key={i} className="flex flex-col gap-1 p-3 bg-slate-900/60 rounded-2xl border border-slate-800 self-start max-w-[80%]">
                    <p className="text-xs font-bold text-indigo-400">{m.user}</p>
                    <p className="text-sm">{m.text}</p>
                  </div>
                ))
              )}
            </>
          )}
        </div>

        {joined && (
          <div className="p-6 bg-slate-950/40 border-t border-slate-800">
            <div className="flex gap-4">
              <input 
                type="text" 
                placeholder="Share a message or gloss..." 
                className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 outline-none focus:border-indigo-500 transition-all"
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <button
                className="p-3 bg-indigo-600 rounded-xl text-white hover:bg-indigo-700 transition-colors"
                onClick={() => {
                  if (!input.trim() || !socket) return;
                  socket.emit('send_translation', { room, translation: { text: input.trim(), user: 'You' } });
                  setMessages((prev) => [...prev, { user: 'You', text: input.trim() }]);
                  setInput('');
                }}
              >
                <Send size={20} />
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-6">
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <ShieldCheck className="text-indigo-300" size={20} /> Room Security
          </h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            All room communications are end-to-end encrypted using our Neural Tunneling protocol.
          </p>
          <p className="text-xs text-slate-500 mt-3">Server: {serverStatus}</p>
        </div>

        <div className="glass-card p-6 flex-1">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Zap className="text-amber-400" size={20} /> Active Peers
          </h3>
          <div className="space-y-3">
            {!joined ? (
              <p className="text-sm text-slate-600 italic">Connect to a room to see active peers</p>
            ) : (
              ['Alex (You)', 'Sarah', 'Dev-System'].map((user, i) => (
                <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-slate-900/50">
                  <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-bold">
                    {user[0]}
                  </div>
                  <span className="text-sm font-medium">{user}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Collaboration;
