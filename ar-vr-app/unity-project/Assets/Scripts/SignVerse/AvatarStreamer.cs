using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

namespace SignVerse
{
    /// <summary>
    /// AvatarStreamer — Handles real-time WebSocket communication with the SignVerse AI Server.
    /// Streams camera frames (as bytes) and receives 848-dim motion intelligence results.
    /// </summary>
    public class AvatarStreamer : MonoBehaviour
    {
        [Header("Connection")]
        public string serverUrl = "ws://localhost:8000/ws/stream";
        public string authToken = "";

        [Header("Streaming Settings")]
        [Range(1, 60)] public int targetFps = 30;
        
        private WebSocket _websocket;
        private bool _isConnecting = false;

        public event Action<SignVerseResult> OnResultReceived;

        [Serializable]
        public class SignVerseResult
        {
            public int gesture_id;
            public string gesture_label;
            public string intent;
            public float[] intelligence_vector; // The 848-dim vector
            public long server_ts;
            public int latency_ms;
        }

        async void Start()
        {
            await Connect();
        }

        async System.Threading.Tasks.Task Connect()
        {
            _isConnecting = true;
            string url = $"{serverUrl}?token={authToken}";
            _websocket = new WebSocket(url);

            _websocket.OnOpen += () => Debug.Log("[SignVerse] Connected to AI Server.");
            _websocket.OnError += (e) => Debug.LogError("[SignVerse] WebSocket Error: " + e);
            _websocket.OnClose += (c) => Debug.Log("[SignVerse] Connection closed.");

            _websocket.OnMessage += (bytes) =>
            {
                var json = System.Text.Encoding.UTF8.GetString(bytes);
                var result = JsonUtility.FromJson<SignVerseResult>(json);
                OnResultReceived?.Invoke(result);
            };

            await _websocket.Connect();
            _isConnecting = false;
        }

        void Update()
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
                _websocket.DispatchMessageQueue();
            #endif
        }

        public async void SendFrame(byte[] imageBytes)
        {
            if (_websocket.State == WebSocketState.Open)
            {
                await _websocket.Send(imageBytes);
            }
        }

        private async void OnApplicationQuit()
        {
            await _websocket.Close();
        }
    }
}
