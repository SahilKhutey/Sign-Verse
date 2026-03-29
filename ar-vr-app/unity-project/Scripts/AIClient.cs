using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// AIClient — Production Unity Client for REST API interaction.
/// Connects to the FastAPI backend for gesture classification.
/// </summary>
public class AIClient : MonoBehaviour
{
    [Header("Server API")]
    public string apiUrl = "http://localhost:8000/predict";
    public float pollRate = 0.5f;

    [Header("Dependencies")]
    public AvatarController avatarController;

    private void Start()
    {
        // Example: Periodic heartbeats or health checks
        StartCoroutine(HealthCheck());
    }

    private IEnumerator HealthCheck()
    {
        using (UnityWebRequest req = UnityWebRequest.Get("http://localhost:8000/health"))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
                Debug.Log($"[AIClient] Server: {req.downloadHandler.text}");
            else
                Debug.LogWarning("[AIClient] Backend offline.");
        }
    }

    public void RequestInference(List<float[]> keypointSequence)
    {
        string json = JsonUtility.ToJson(new InferencePayload { sequence = keypointSequence });
        StartCoroutine(PostPrediction(json));
    }

    private IEnumerator PostPrediction(string json)
    {
        using (UnityWebRequest req = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] body = System.Text.Encoding.UTF8.GetBytes(json);
            req.uploadHandler = new UploadHandlerRaw(body);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");

            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                ProcessResponse(req.downloadHandler.text);
            }
            else
            {
                Debug.LogWarning($"[AIClient] Prediction failed: {req.error}");
            }
        }
    }

    private void ProcessResponse(string json)
    {
        try {
            PredictionResponse res = JsonUtility.FromJson<PredictionResponse>(json);
            Debug.Log($"[AIClient] Inference Result: {res.label} (ID: {res.gesture_id})");
            
            // Trigger animation
            avatarController?.PlayGesture(res.label);
        }
        catch (Exception e) {
            Debug.LogWarning($"[AIClient] Parse error: {e.Message}");
        }
    }

    [Serializable]
    private class InferencePayload
    {
        public List<float[]> sequence;
    }

    [Serializable]
    private class PredictionResponse
    {
        public int gesture_id;
        public string label;
    }
}
