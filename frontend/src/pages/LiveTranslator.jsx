import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Camera, Mic, Info } from 'lucide-react';
import { textToSign, getInferenceHealth } from '../api/signverse';

const LiveTranslator = () => {
  const [sentiment, setSentiment] = useState('Neutral');
  const [confidence, setConfidence] = useState(98.4);
  const [text, setText] = useState('');
  const [tokens, setTokens] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inferenceStatus, setInferenceStatus] = useState('Unknown');
  const [streamStatus, setStreamStatus] = useState('Offline');
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const [fps, setFps] = useState(0);
  const [lastLabel, setLastLabel] = useState('Detecting...');
  const [labelHistory, setLabelHistory] = useState([]);
  const [smoothLabel, setSmoothLabel] = useState('Detecting...');
  const [latency, setLatency] = useState(null);
  const [confidencePct, setConfidencePct] = useState(null);
  const [droppedFrames, setDroppedFrames] = useState(0);
  const [serverTs, setServerTs] = useState(null);
  const [streamFps, setStreamFps] = useState(10);
  const [streamWindow, setStreamWindow] = useState(8);
  const [minConfidence, setMinConfidence] = useState(0.4);
  const [maxFrameBytes, setMaxFrameBytes] = useState(1000000);
  const [jpegQuality, setJpegQuality] = useState(0.6);
  const [streamError, setStreamError] = useState(null);
  const [useSequence, setUseSequence] = useState(false);
  const [sequenceWindow, setSequenceWindow] = useState(30);
  const [sequenceReady, setSequenceReady] = useState(false);
  const [sequenceLength, setSequenceLength] = useState(0);

  useEffect(() => {
    getInferenceHealth()
      .then(() => setInferenceStatus('Online'))
      .catch(() => setInferenceStatus('Offline'));
  }, []);

  useEffect(() => {
    // Start webcam
    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (e) {
        setStreamStatus('Camera denied');
      }
    };
    start();
  }, []);

  const startStream = () => {
    const baseUrl = import.meta.env.VITE_INFERENCE_WS || 'ws://localhost:8000/ws/stream';
    const token = import.meta.env.VITE_STREAM_TOKEN;
    const url = token ? `${baseUrl}?token=${encodeURIComponent(token)}` : baseUrl;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => {
      setStreamStatus('Online');
      setStreamError(null);
      ws.send(JSON.stringify({
        type: 'config',
        fps: streamFps,
        window: streamWindow,
        min_confidence: minConfidence,
        max_frame_bytes: maxFrameBytes,
        use_sequence: useSequence,
        sequence_window: sequenceWindow,
      }));
    };
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.error) {
          setStreamError(`${data.error}: ${data.detail || 'unknown'}`);
          return;
        }
        if (data.type === 'config_ack') {
          return;
        }
        if (data.gesture_label) {
          setTokens([data.gesture_label]);
          setLastLabel(data.gesture_label);
          setLabelHistory((prev) => {
            const next = [data.gesture_label, ...prev].slice(0, 10);
            // client-side smoothing by majority
            const counts = next.reduce((acc, v) => {
              acc[v] = (acc[v] || 0) + 1;
              return acc;
            }, {});
            let best = next[0];
            let bestCount = 0;
            Object.entries(counts).forEach(([k, c]) => {
              if (c > bestCount) {
                best = k;
                bestCount = c;
              }
            });
            setSmoothLabel(best);
            return next;
          });
        }
        if (data.latency_ms !== undefined) setLatency(data.latency_ms);
        if (data.confidence !== undefined && data.confidence !== null) {
          setConfidencePct(Math.round(data.confidence * 100));
        }
        if (data.dropped_frames !== undefined) setDroppedFrames(data.dropped_frames);
        if (data.server_ts !== undefined) setServerTs(data.server_ts);
        if (data.sequence_ready !== undefined) setSequenceReady(data.sequence_ready);
        if (data.sequence_length !== undefined) setSequenceLength(data.sequence_length);
      } catch {}
    };
    ws.onclose = () => setStreamStatus('Offline');

    // frame capture loop
    let lastTime = performance.now();
    const captureInterval = Math.round(1000 / Math.max(1, streamFps));
    const loop = () => {
      if (!videoRef.current || !canvasRef.current || ws.readyState !== 1) {
        return;
      }
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) ws.send(blob);
      }, 'image/jpeg', Math.min(0.95, Math.max(0.1, jpegQuality)));

      const now = performance.now();
      setFps(Math.round(1000 / Math.max(1, now - lastTime)));
      lastTime = now;
      setTimeout(loop, captureInterval);
    };
    setTimeout(loop, 200);
  };

  const stopStream = () => {
    if (wsRef.current) wsRef.current.close();
    setStreamStatus('Offline');
    setStreamError(null);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="dashboard-grid"
    >
      <div className="glass-card p-8 flex flex-col gap-6 min-h-[600px]">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold">Live AI Translator</h2>
          <div className="flex gap-2 bg-slate-900 p-1 rounded-lg">
            <button className="px-3 py-1 bg-indigo-600 rounded-md text-sm">Sign-to-Text</button>
            <button className="px-3 py-1 text-slate-400 text-sm">Speech-to-Sign</button>
          </div>
        </div>

        <div className="flex-1 bg-slate-900 rounded-3xl border border-slate-800 flex items-center justify-center relative overflow-hidden group">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover opacity-80"
          />
          <canvas ref={canvasRef} className="hidden" />
          <Camera size={48} className="text-slate-700 group-hover:scale-110 transition-transform" />
          
          {/* Emotion Overlay */}
          <div className="absolute top-6 right-6 glass-card bg-indigo-500/10 border-indigo-500/20 px-4 py-2 flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-indigo-300">Sentiment: {sentiment}</span>
          </div>

          <p className="absolute bottom-6 left-6 text-indigo-400 font-medium flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
            Camera Active
          </p>
          <p className="absolute bottom-6 right-6 text-xs text-slate-300">
            FPS: {fps} | {smoothLabel} | {latency !== null ? `${latency}ms` : '...'}
          </p>
        </div>

        <div className="glass-card bg-indigo-500/10 border-indigo-500/20 p-6">
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm text-indigo-300 font-medium uppercase tracking-wider">AI Confidence: {confidence}%</p>
            <Info size={16} className="text-slate-500" />
          </div>
          <p className="text-2xl font-semibold">"HELLO WORLD, I AM LEARNING SIGN LANGUAGE"</p>
          <div className="text-xs text-slate-500 mt-2">Inference: {inferenceStatus}</div>
          <div className="text-xs text-slate-500 mt-1">Stream: {streamStatus}</div>
          <div className="text-xs text-slate-500 mt-1">
            Sequence: {useSequence ? (sequenceReady ? `Ready (${sequenceLength})` : `Buffering (${sequenceLength})`) : 'Off'}
          </div>
          {confidencePct !== null && (
            <div className="text-xs text-slate-500 mt-1">Confidence: {confidencePct}%</div>
          )}
          {streamError && (
            <div className="text-xs text-amber-400 mt-1">Stream error: {streamError}</div>
          )}
          {droppedFrames > 0 && (
            <div className="text-xs text-slate-500 mt-1">Dropped frames: {droppedFrames}</div>
          )}
          {serverTs && (
            <div className="text-xs text-slate-500 mt-1">Server time: {serverTs}</div>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-6">
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4">Text to Sign (API)</h3>
          <textarea
            className="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 outline-none focus:border-indigo-500 transition-all text-sm min-h-[90px]"
            placeholder="Type a sentence to convert..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            className="w-full btn-primary mt-4"
            disabled={loading || !text.trim()}
            onClick={async () => {
              setLoading(true);
              setError(null);
              setTokens([]);
              try {
                const res = await textToSign(text.trim());
                setTokens(res.tokens || []);
              } catch (e) {
                setError('API request failed');
              } finally {
                setLoading(false);
              }
            }}
          >
            {loading ? 'Converting...' : 'Convert'}
          </button>
          {error && <p className="text-sm text-red-400 mt-3">{error}</p>}
          {tokens.length > 0 && (
            <div className="mt-4 text-sm text-slate-300">
              <div className="text-xs text-slate-500 mb-1">Tokens</div>
              <div className="flex flex-wrap gap-2">
                {tokens.map((t, i) => (
                  <span key={i} className="px-2 py-1 bg-slate-900 border border-slate-700 rounded-lg">{t}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4">Target Language</h3>
          <select className="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 outline-none focus:border-indigo-500 transition-all">
            <option>English (US)</option>
            <option>Hindi (IN)</option>
            <option>Spanish (ES)</option>
            <option>French (FR)</option>
          </select>
        </div>

        <div className="glass-card p-6 flex-1">
          <h3 className="text-lg font-semibold mb-4">Audio Output</h3>
          <div className="space-y-4">
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Synthesis Engine</span>
              <span className="text-indigo-400">SignVerse Neural</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Voice Gender</span>
              <span className="text-indigo-400">Female (Standard)</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-slate-400">Emotional Pitch</span>
              <span className="text-indigo-400">Synced</span>
            </div>
            <button className="w-full btn-primary mt-4 flex items-center justify-center gap-2">
              <Mic size={18} /> Test Speaker
            </button>
          </div>
        </div>

        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-4">Live Stream</h3>
          <div className="grid grid-cols-2 gap-3 text-xs mb-4">
            <label className="flex flex-col gap-1 text-slate-400">
              FPS
              <input
                type="number"
                min="1"
                max="60"
                value={streamFps}
                onChange={(e) => setStreamFps(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              Smoothing Window
              <input
                type="number"
                min="1"
                max="60"
                value={streamWindow}
                onChange={(e) => setStreamWindow(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              Min Confidence
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              JPEG Quality
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="0.95"
                value={jpegQuality}
                onChange={(e) => setJpegQuality(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              Max Frame Bytes
              <input
                type="number"
                min="10000"
                max="5000000"
                value={maxFrameBytes}
                onChange={(e) => setMaxFrameBytes(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              Sequence Window
              <input
                type="number"
                min="5"
                max="120"
                value={sequenceWindow}
                onChange={(e) => setSequenceWindow(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-200"
              />
            </label>
            <label className="flex items-center gap-2 text-slate-400 mt-6">
              <input
                type="checkbox"
                checked={useSequence}
                onChange={(e) => setUseSequence(e.target.checked)}
              />
              Enable Sequence Mode
            </label>
          </div>
          <div className="flex gap-2">
            <button className="btn-primary" onClick={startStream}>Start Stream</button>
            <button className="bg-slate-800 text-slate-200 px-4 py-2 rounded-xl" onClick={stopStream}>Stop</button>
          </div>
          {labelHistory.length > 0 && (
            <div className="text-xs text-slate-400 mt-3">
              Recent: {labelHistory.join(' | ')}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default LiveTranslator;
