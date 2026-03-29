using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class GestureReceiver : MonoBehaviour
{
    public string serverUrl = "http://localhost:8000/inference/gesture";
    public string feedbackUrl = "http://localhost:8000/feedback/correction";
    public float pollInterval = 0.5f;

    [Serializable]
    public class GestureResponse
    {
        public string gesture;
        public float confidence;
        public string language;
        public string translation;
    }

    [Serializable]
    public class CorrectionData
    {
        public string original_text;
        public string corrected_text;
        public string language;
        public string video_session_id;
        public float[][] sequence_data;
    }

    void Start()
    {
        StartCoroutine(GetGestureData());
    }

    IEnumerator GetGestureData()
    {
        while (true)
        {
            using (UnityWebRequest webRequest = UnityWebRequest.Get(serverUrl))
            {
                yield return webRequest.SendWebRequest();

                if (webRequest.result == UnityWebRequest.Result.Success)
                {
                    string jsonResponse = webRequest.downloadHandler.text;
                    GestureResponse data = JsonUtility.FromJson<GestureResponse>(jsonResponse);
                    
                    Debug.Log($"[{data.language}] Sign: {data.gesture} -> {data.translation}");
                    
                    // Notify other components (e.g., Avatar, UI)
                    BroadcastMessage("OnGestureReceived", data, SendMessageOptions.DontRequireReceiver);
                }
            }
            yield return new WaitForSeconds(pollInterval);
        }
    }

    public void ReportCorrection(string original, string corrected, string lang, float[][] sequence)
    {
        CorrectionData correction = new CorrectionData
        {
            original_text = original,
            corrected_text = corrected,
            language = lang,
            video_session_id = "unity_xr_" + DateTime.Now.Ticks,
            sequence_data = sequence
        };

        string json = JsonUtility.ToJson(correction);
        StartCoroutine(PostCorrection(json));
    }

    IEnumerator PostCorrection(string json)
    {
        using (UnityWebRequest request = new UnityWebRequest(feedbackUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Correction reported successfully.");
            }
            else
            {
                Debug.LogError("Failed to report correction: " + request.error);
            }
        }
    }
}
