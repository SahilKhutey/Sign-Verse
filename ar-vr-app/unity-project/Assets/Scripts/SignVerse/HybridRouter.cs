using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using NativeWebSocket;

namespace SignVerse
{
    /// <summary>
    /// HybridRouter — Latency-aware edge/cloud routing client for AR Glasses.
    ///
    /// Every inference frame it:
    ///   1. Collects edge telemetry (confidence, latency, entropy, RTT estimate)
    ///   2. Sends telemetry + 848-dim keypoints to /ws/hybrid
    ///   3. Receives a routing decision: "edge" or "cloud"
    ///   4.a "edge"  → display EdgeInferenceEngine.currentTranslation directly
    ///   4.b "cloud" → overwrite with cloud gesture label from server response
    ///
    /// Rate limiting, JWT auth, and session telemetry are all managed server-side.
    /// </summary>
    public class HybridRouter : MonoBehaviour
    {
        // ── Inspector ──────────────────────────────────────────────────────────
        [Header("Connection")]
        public string hybridServerUrl = "wss://localhost:8000/ws/hybrid";
        public string authToken = "";

        [Header("Components")]
        public EdgeInferenceEngine edgeEngine;
        public SignOverlayHandler  overlayHandler;

        [Header("Routing Thresholds (Mirror Server)")]
        [Range(0f, 1f)]  public float confidenceThreshold   = 0.50f;
        [Range(10f, 100f)] public float latencyBudgetMs     = 45f;

        [Header("Telemetry")]
        public bool showDebugRoute = true;

        // ── State ──────────────────────────────────────────────────────────────
        public string finalTranslation  = "";
        public string lastRoute         = "edge";
        public float  lastRouteScore    = 0f;

        private WebSocket   _ws;
        private bool        _connected      = false;
        private float       _lastSendTime   = 0f;
        private const float SendIntervalSec = 0.033f; // ~30 fps

        // RTT rolling average
        private float _rttMs     = 0f;
        private float _sendTs    = 0f;

        // ── Lifecycle ──────────────────────────────────────────────────────────
        async void Start()
        {
            await Connect();
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            if (_ws != null) _ws.DispatchMessageQueue();
#endif
            if (_connected && Time.time - _lastSendTime > SendIntervalSec)
            {
                SendTelemetryFrame();
                _lastSendTime = Time.time;
            }

            // Sync overlay with final translation
            if (overlayHandler != null)
                overlayHandler.hybridManager = null; // Detach old manager; driven here
        }

        private async System.Threading.Tasks.Task Connect()
        {
            string url = $"{hybridServerUrl}?token={authToken}";
            _ws = new WebSocket(url);

            _ws.OnOpen    += () => { _connected = true; Debug.Log("[HybridRouter] Connected."); };
            _ws.OnClose   += (_) => { _connected = false; Debug.Log("[HybridRouter] Disconnected."); };
            _ws.OnError   += (e) => Debug.LogError($"[HybridRouter] WS Error: {e}");
            _ws.OnMessage += bytes =>
            {
                // Measure round-trip time
                float nowMs = Time.realtimeSinceStartup * 1000f;
                _rttMs = 0.8f * _rttMs + 0.2f * (nowMs - _sendTs);

                string json = Encoding.UTF8.GetString(bytes);
                HandleServerMessage(json);
            };

            await _ws.Connect();
        }

        // ── Send ───────────────────────────────────────────────────────────────
        private void SendTelemetryFrame()
        {
            if (edgeEngine == null || _ws.State != WebSocketState.Open) return;

            // Collect edge metrics
            float confidence       = edgeEngine.confidence;
            float edgeLatencyMs    = edgeEngine.lastInferenceLatencyMs;
            float entropy          = ComputeTokenEntropy(edgeEngine.lastTokenWindow);
            int   gestureComplexity = edgeEngine.distinctTokenCount;

            // Build JSON payload (no raw video — 848-dim vectors only)
            var payload = new TelemetryPayload
            {
                type              = "infer",
                device_id         = SystemInfo.deviceUniqueIdentifier,
                keypoints         = edgeEngine.lastMotionVector,   // float[848]
                confidence        = confidence,
                latency_ms        = edgeLatencyMs,
                entropy           = entropy,
                gesture_complexity = gestureComplexity,
                network_rtt       = _rttMs,
                frame_id          = (long)(Time.realtimeSinceStartup * 1000),
            };

            string json = JsonUtility.ToJson(payload);
            _sendTs = Time.realtimeSinceStartup * 1000f;
            _ws.SendText(json);
        }

        // ── Receive ────────────────────────────────────────────────────────────
        private void HandleServerMessage(string json)
        {
            var msg = JsonUtility.FromJson<RouterResponse>(json);
            if (msg == null) return;

            lastRoute = msg.route;

            if (msg.type == "error")
            {
                Debug.LogWarning($"[HybridRouter] Server error: {msg.code}");
                return;
            }

            if (msg.route == "edge")
            {
                // Use local result — already rendered by EdgeInferenceEngine
                finalTranslation = edgeEngine != null ? edgeEngine.currentTranslation : finalTranslation;
            }
            else if (msg.route == "cloud" && !string.IsNullOrEmpty(msg.gesture_label))
            {
                finalTranslation = $"[Cloud] {msg.gesture_label}";
            }

            // Push to overlay
            if (overlayHandler != null && !string.IsNullOrEmpty(finalTranslation))
                overlayHandler.DisplayText(finalTranslation);

            if (showDebugRoute)
                Debug.Log($"[HybridRouter] Route={msg.route} | Reason={msg.reason} | Conf={msg.confidence:F2} | Lat={msg.latency_ms}ms");
        }

        // ── Helpers ────────────────────────────────────────────────────────────
        private float ComputeTokenEntropy(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0) return 0f;
            var counts = new Dictionary<int, int>();
            foreach (int t in tokens) counts[t] = counts.TryGetValue(t, out int c) ? c + 1 : 1;
            float entropy = 0f;
            float n = tokens.Length;
            foreach (var kv in counts)
            {
                float p = kv.Value / n;
                if (p > 0) entropy -= p * Mathf.Log(p, 2);
            }
            return Mathf.Clamp01(entropy / 9f); // Normalise: log2(512) ≈ 9
        }

        private async void OnApplicationQuit()
        {
            if (_ws != null) await _ws.Close();
        }

        // ── DTOs ───────────────────────────────────────────────────────────────
        [Serializable]
        private class TelemetryPayload
        {
            public string type;
            public string device_id;
            public float[] keypoints;
            public float confidence;
            public float latency_ms;
            public float entropy;
            public int   gesture_complexity;
            public float network_rtt;
            public long  frame_id;
        }

        [Serializable]
        private class RouterResponse
        {
            public string type;
            public string route;
            public string reason;
            public string code;
            public string gesture_label;
            public int    gesture_id;
            public float  confidence;
            public int    latency_ms;
            public long   frame_id;
        }
    }
}
