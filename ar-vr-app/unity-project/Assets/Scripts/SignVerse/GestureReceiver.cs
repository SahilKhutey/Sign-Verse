using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

namespace SignVerse
{
    /// <summary>
    /// GestureReceiver — Handles back-end connectivity for gesture translation.
    /// Manages the WebSocket connection and invokes events for translated text/tokens.
    /// </summary>
    public class GestureReceiver : MonoBehaviour
    {
        [Header("Connection")]
        public string serverUrl = "ws://localhost:8000/ws/stream";
        public string authToken = "";

        private WebSocket _websocket;
        private bool _isConnecting = false;

        public event Action<TranslationResult> OnTranslationReceived;

        [Serializable]
        public class TranslationResult
        {
            public string text;
            public string[] tokens;
            public float confidence;
            public string intent; // e.g. "UI_ACTIVATE", "SIGNING"
        }

        async void Start()
        {
            await Connect();
        }

        async System.Threading.Tasks.Task Connect()
        {
            if (_isConnecting) return;
            _isConnecting = true;

            string url = $"{serverUrl}?token={authToken}";
            _websocket = new WebSocket(url);

            _websocket.OnOpen += () => Debug.Log("[SignVerse] Connected to Gesture Backend.");
            _websocket.OnError += (e) => Debug.LogError("[SignVerse] WebSocket Error: " + e);
            _websocket.OnClose += (c) => Debug.Log("[SignVerse] Connection closed.");

            _websocket.OnMessage += (bytes) =>
            {
                var json = System.Text.Encoding.UTF8.GetString(bytes);
                var result = JsonUtility.FromJson<TranslationResult>(json);
                OnTranslationReceived?.Invoke(result);
            };

            await _websocket.Connect();
            _isConnecting = false;
        }

        public async void SendRaw(string json)
        {
            if (_websocket != null && _websocket.State == WebSocketState.Open)
            {
                await _websocket.SendText(json);
            }
        }

        void Update()
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
                if (_websocket != null) _websocket.DispatchMessageQueue();
            #endif
        }

        private async void OnApplicationQuit()
        {
            if (_websocket != null) await _websocket.Close();
        }
    }
}
