using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace SignVerse
{
    /// <summary>
    /// HybridCommunicationManager — Manages the state switching between Edge AI and Cloud Fallback.
    /// Orchestrates GestureReceiver and EdgeInferenceEngine.
    /// </summary>
    public class HybridCommunicationManager : MonoBehaviour
    {
        [Header("Components")]
        public EdgeInferenceEngine edgeEngine;
        public GestureReceiver gestureReceiver;

        [Header("Hybrid Settings")]
        public float localConfidenceThreshold = 0.5f;
        public float cloudFallbackDelay = 1.0f; // Seconds of low confidence before switching to cloud

        private bool _isUsingCloud = false;
        private float _lowConfidenceTimer = 0.0f;

        public string finalTranslationText = "";

        void Start()
        {
            if (gestureReceiver != null)
            {
                gestureReceiver.OnTranslationReceived += HandleCloudResult;
            }
        }

        void Update()
        {
            if (edgeEngine == null || gestureReceiver == null) return;

            // 1. Check local confidence
            if (edgeEngine.confidence < localConfidenceThreshold)
            {
                _lowConfidenceTimer += Time.deltaTime;
            }
            else
            {
                _lowConfidenceTimer = 0;
                _isUsingCloud = false;
            }

            // 2. Trigger cloud fallback if local fails for too long
            if (_lowConfidenceTimer > cloudFallbackDelay)
            {
                _isUsingCloud = true;
            }

            // 3. Coordinate results
            if (!_isUsingCloud)
            {
                finalTranslationText = edgeEngine.currentTranslation;
            }
            // If using cloud, finalTranslationText is updated via HandleCloudResult
        }

        public void SendLandmarks(float[] intelVector)
        {
            if (_isUsingCloud && gestureReceiver != null)
            {
                // Send the 848-dim vector to the cloud for complex inference
                string json = "{\"type\":\"frame\", \"keypoints\": [" + string.Join(",", intelVector) + "], \"mode\":\"3d\"}";
                gestureReceiver.SendRaw(json); 
            }
        }

        private void HandleCloudResult(GestureReceiver.TranslationResult result)
        {
            if (_isUsingCloud)
            {
                finalTranslationText = "Cloud: " + result.text;
                Debug.Log($"[SignVerse] Hybrid Cloud Result: {result.text}");
            }
        }
    }
}
