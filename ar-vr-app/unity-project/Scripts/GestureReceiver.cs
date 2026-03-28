using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

/// <summary>
/// GestureReceiver — Connects to SignVerse AI Server via WebSocket.
/// Receives real-time gesture classifications and sign tokens
/// and passes them to downstream components.
/// </summary>
public class GestureReceiver : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverUrl = "ws://localhost:8888/translate?target_lang=ASL";
    public float reconnectDelay = 3f;

    [Header("References")]
    public AvatarController avatarController;
    public SignRenderer signRenderer;

    // Events
    public event Action<int> OnGestureDetected;
    public event Action<List<string>> OnSignTokensReceived;

    private WebSocket webSocket;
    private bool isConnecting = false;
    private int frameId = 0;

    private async void Start()
    {
        await ConnectToServer();
    }

    private async System.Threading.Tasks.Task ConnectToServer()
    {
        isConnecting = true;
        webSocket = new WebSocket(serverUrl);

        webSocket.OnOpen += () => {
            Debug.Log("[GestureReceiver] Connected to SignVerse AI Server");
            isConnecting = false;
        };

        webSocket.OnMessage += (bytes) => {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            ProcessMessage(json);
        };

        webSocket.OnError += (e) => {
            Debug.LogWarning($"[GestureReceiver] WebSocket error: {e}");
        };

        webSocket.OnClose += (code) => {
            Debug.Log($"[GestureReceiver] Disconnected (code: {code})");
            StartCoroutine(ReconnectAfterDelay());
        };

        await webSocket.Connect();
    }

    private void ProcessMessage(string json)
    {
        try {
            GestureMessage msg = JsonUtility.FromJson<GestureMessage>(json);

            if (msg.gesture_id >= 0)
                OnGestureDetected?.Invoke(msg.gesture_id);

            if (msg.sign_tokens != null && msg.sign_tokens.Count > 0) {
                OnSignTokensReceived?.Invoke(msg.sign_tokens);
                avatarController?.PlaySignSequence(msg.sign_tokens);
                signRenderer?.RenderTokens(msg.sign_tokens);
            }
        }
        catch (Exception e) {
            Debug.LogWarning($"[GestureReceiver] Parse error: {e.Message}");
        }
    }

    public async void SendFrame(float[] keypoints, string text = "")
    {
        if (webSocket?.State != WebSocketState.Open) return;

        FramePayload payload = new FramePayload {
            keypoints = keypoints,
            text = text,
            frame_id = frameId++
        };

        string json = JsonUtility.ToJson(payload);
        await webSocket.SendText(json);
    }

    private IEnumerator ReconnectAfterDelay()
    {
        yield return new WaitForSeconds(reconnectDelay);
        if (!isConnecting)
            _ = ConnectToServer();
    }

    private void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        webSocket?.DispatchMessageQueue();
#endif
    }

    private async void OnApplicationQuit()
    {
        if (webSocket != null)
            await webSocket.Close();
    }

    [Serializable]
    private class GestureMessage
    {
        public int gesture_id;
        public List<string> sign_tokens;
        public int frame_id;
    }

    [Serializable]
    private class FramePayload
    {
        public float[] keypoints;
        public string text;
        public int frame_id;
    }
}
