import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Camera, 
  Settings, 
  Layers, 
  Cpu, 
  CheckCircle, 
  XCircle, 
  Edit3, 
  Send,
  Zap,
  Activity,
  Maximize2
} from 'lucide-react';
import { getInferenceHealth, submitCorrection } from '../api/signverse';

const ARDashboard = () => {
  const [prediction, setPrediction] = useState('Waiting for gesture...');
  const [confidence, setConfidence] = useState(0);
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [correctionText, setCorrectionText] = useState('');
  const [history, setHistory] = useState([]);
  const [streamStatus, setStreamStatus] = useState('Disconnected');
  const [latency, setLatency] = useState(0);
  const [submitting, setSubmitting] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  // Simulation of AR Floating Panels
  const panels = [
    { id: 'stats', title: 'Neural Metrics', icon: Cpu, x: 20, y: 150 },
    { id: 'output', title: 'Live Translation', icon: Zap, x: 800, y: 150 },
    { id: 'debug', title: 'System Logs', icon: Activity, x: 20, y: 500 },
  ];

  useEffect(() => {
    // Start Webcam
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStreamStatus('Connected');
      } catch (err) {
        console.error("Camera error:", err);
        setStreamStatus('Camera Error');
      }
    };
    startWebcam();

    // Setup WebSocket (Mocking logic similar to LiveTranslator for now)
    const setupWS = () => {
        const url = import.meta.env.VITE_INFERENCE_WS || 'ws://localhost:8000/ws/stream';
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onmessage = (evt) => {
            const data = JSON.parse(evt.data);
            if (data.gesture_label) {
                setPrediction(data.gesture_label);
                setConfidence(Math.round((data.confidence || 0.95) * 100));
                setLatency(data.latency_ms || 24);
                
                // Add to history if high confidence
                if (data.confidence > 0.8) {
                    setHistory(prev => [{
                        label: data.gesture_label,
                        ts: new Date().toLocaleTimeString(),
                        conf: Math.round(data.confidence * 100)
                    }, ...prev].slice(0, 5));
                }
            }
        };

        ws.onclose = () => setStreamStatus('Disconnected');
    };

    // Only start WS if in real environment
    if (import.meta.env.MODE !== 'test') {
        // setupWS(); 
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
         videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const handleCorrection = async () => {
    if (!correctionText.trim()) return;
    setSubmitting(true);
    try {
      await submitCorrection({
        original_text: prediction,
        corrected_text: correctionText,
        language: "ASL",
        timestamp: new Date().toISOString()
      });
      setPrediction(correctionText);
      setIsCorrecting(false);
      setCorrectionText('');
      // Notification would go here
    } catch (err) {
      console.error("Feedback failed:", err);
    } finally {
        setSubmitting(false);
    }
  };

  return (
    <div className="relative h-[calc(100vh-80px)] w-full bg-black overflow-hidden font-mono">
      {/* Background Camera Feed */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover opacity-60 grayscale-[0.3]"
      />
      
      {/* AR HUD Overlay - Scanning Lines */}
      <div className="absolute inset-0 pointer-events-none opacity-20 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%]" />

      {/* Main UI Header */}
      <motion.div 
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-10 flex justify-between items-center p-6 bg-gradient-to-b from-black/80 to-transparent"
      >
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-primary flex items-center justify-center animate-pulse">
            <Layers className="text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tighter text-white">SIGNVERSE <span className="text-primary">XR-ALPHA</span></h1>
            <div className="flex items-center gap-2 text-[10px] text-slate-400">
               <span className="w-2 h-2 bg-green-500 rounded-full animate-ping" />
               SYSTEM STATUS: {streamStatus} | LATENCY: {latency}ms
            </div>
          </div>
        </div>
        
        <div className="flex gap-4">
            <button className="p-2 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
                <Settings className="text-white/60" size={20} />
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-primary rounded-lg font-bold text-sm shadow-lg shadow-primary/20">
                <Maximize2 size={16} /> CALIBRATE
            </button>
        </div>
      </motion.div>

      {/* Floating Panels Container */}
      <div className="relative z-10 h-full w-full">
        
        {/* CENTER FOCUS: TRANSLATION OUTPUT */}
        <motion.div 
          layoutId="center-portal"
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px]"
        >
          <div className="glass-card p-1 border-primary/30 bg-black/40 backdrop-blur-xl">
            <div className="p-8 text-center">
              <div className="text-primary text-[10px] mb-2 uppercase tracking-[0.3em] font-black">Predicted Gesture</div>
              
              <AnimatePresence mode="wait">
                <motion.div
                  key={prediction}
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 1.1, opacity: 0 }}
                  className="text-5xl font-black text-white mb-6 tracking-tight"
                >
                  {prediction}
                </motion.div>
              </AnimatePresence>

              <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden mb-8">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${confidence}%` }}
                  className="h-full bg-primary shadow-[0_0_15px_rgba(99,102,241,0.8)]" 
                />
              </div>

              <div className="flex justify-center gap-6">
                 {!isCorrecting ? (
                   <>
                      <button 
                        onClick={() => setIsCorrecting(true)}
                        className="flex flex-col items-center gap-2 group"
                      >
                        <div className="w-12 h-12 rounded-full border border-white/20 flex items-center justify-center group-hover:border-primary/50 group-hover:bg-primary/10 transition-all">
                            <Edit3 className="text-white/60 group-hover:text-primary" size={18} />
                        </div>
                        <span className="text-[10px] text-white/40 group-hover:text-primary uppercase font-bold">Correct</span>
                      </button>
                      <button className="flex flex-col items-center gap-2 group">
                        <div className="w-12 h-12 rounded-full border border-white/20 flex items-center justify-center group-hover:border-green-500/50 group-hover:bg-green-500/10 transition-all">
                            <CheckCircle className="text-white/60 group-hover:text-green-500" size={18} />
                        </div>
                        <span className="text-[10px] text-white/40 group-hover:text-green-500 uppercase font-bold">Validate</span>
                      </button>
                   </>
                 ) : (
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="flex items-center gap-3 bg-white/5 p-2 rounded-2xl border border-white/10 w-full"
                    >
                        <input 
                            autoFocus
                            className="bg-transparent border-none outline-none text-white px-4 flex-1 font-bold"
                            placeholder="Enter correction..."
                            value={correctionText}
                            onChange={(e) => setCorrectionText(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleCorrection()}
                        />
                        <button 
                            disabled={submitting}
                            onClick={handleCorrection}
                            className="p-3 bg-primary rounded-xl hover:scale-105 transition-transform disabled:opacity-50"
                        >
                            <Send size={18} />
                        </button>
                        <button 
                            onClick={() => setIsCorrecting(false)}
                            className="p-3 bg-white/10 rounded-xl hover:bg-white/20"
                        >
                            <XCircle size={18} />
                        </button>
                    </motion.div>
                 )}
              </div>
            </div>
          </div>
        </motion.div>

        {/* SIDE PANEL: RECENT FEEDBACK LOOP */}
        <motion.div 
           initial={{ x: -100, opacity: 0 }}
           animate={{ x: 0, opacity: 1 }}
           className="absolute left-6 top-32 w-72 space-y-4"
        >
            <div className="glass-card p-4 border-white/5">
                <div className="flex items-center gap-2 mb-4 text-primary">
                    <History size={16} />
                    <span className="text-xs font-bold uppercase tracking-widest">Recent Activity</span>
                </div>
                <div className="space-y-3">
                    {history.length > 0 ? history.map((h, i) => (
                        <div key={i} className="flex justify-between items-center text-[10px] pb-2 border-b border-white/5 last:border-0">
                            <span className="text-white font-bold">{h.label}</span>
                            <span className="text-slate-500">{h.ts}</span>
                            <span className="text-primary">{h.conf}%</span>
                        </div>
                    )) : (
                        <div className="text-[10px] text-slate-500 italic">No session data yet...</div>
                    )}
                </div>
            </div>

            <div className="glass-card p-4 border-primary/20 bg-primary/5">
                <div className="flex items-center gap-2 mb-2 text-primary">
                    <Activity size={16} />
                    <span className="text-xs font-bold uppercase tracking-widest">Continuous Learning</span>
                </div>
                <p className="text-[10px] text-slate-400 mb-3">Model hot-reloads every 50 validations. EWC penalty active.</p>
                <div className="flex justify-between text-[10px] mb-1">
                    <span>Batch Readiness</span>
                    <span className="text-primary">42 / 50</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-primary w-[84%]" />
                </div>
            </div>
        </motion.div>

        {/* RIGHT PANEL: NEURAL AVATAR VIEW */}
        <motion.div 
           initial={{ x: 100, opacity: 0 }}
           animate={{ x: 0, opacity: 1 }}
           className="absolute right-6 top-32 w-80"
        >
            <div className="glass-card p-1 border-white/5 aspect-[3/4] overflow-hidden relative group">
                <div className="absolute inset-0 bg-gradient-to-t from-black to-transparent z-10" />
                
                {/* Avatar Placeholder */}
                <div className="w-full h-full bg-slate-900 flex items-center justify-center">
                    <div className="relative">
                        <div className="w-64 h-64 border border-primary/30 rounded-full animate-[spin_10s_linear_infinite]" />
                        <div className="absolute inset-0 flex items-center justify-center">
                            <Cpu className="text-primary/50" size={48} />
                        </div>
                    </div>
                </div>

                <div className="absolute bottom-6 left-6 right-6 z-20">
                    <div className="text-white font-bold mb-1">NEURAL AVATAR V2</div>
                    <div className="text-[10px] text-slate-400">READY FOR TSL INFERENCE</div>
                </div>
            </div>
        </motion.div>

      </div>

      {/* Grid Pattern Background */}
      <div className="absolute inset-0 pointer-events-none opacity-5" 
           style={{ backgroundImage: 'radial-gradient(circle, #6366f1 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
    </div>
  );
};

export default ARDashboard;
