using UnityEngine;
using SignVerse;

namespace SignVerse
{
    /// <summary>
    /// AvatarController — Unified Entry Point for SignVerse Unity XR.
    /// Manages real-time WebCam capture, streaming, and animation mapping.
    /// </summary>
    public class AvatarController : MonoBehaviour
    {
        [Header("Components")]
        public AvatarStreamer streamer;
        public SignRigMapper mapper;

        [Header("WebCam Settings")]
        public int webcamWidth = 640;
        public int webcamHeight = 480;
        public int fps = 30;

        private WebCamTexture _webcamTexture;
        private Texture2D _sendTexture;
        private float _nextFrameTime = 0f;

        [Header("XR Feedback")]
        public GestureReceiver gestureReceiver;
        public GameObject translationHUD;

        void Start()
        {
            // Initialize WebCam
            _webcamTexture = new WebCamTexture(webcamWidth, webcamHeight, fps);
            _webcamTexture.Play();
            _sendTexture = new Texture2D(webcamWidth, webcamHeight, TextureFormat.RGB24, false);

            // Subscribe to Streamer Results
            if (streamer != null)
            {
                streamer.OnResultReceived += HandleStreamResult;
            }

            // Subscribe to Translation Results
            if (gestureReceiver != null)
            {
                gestureReceiver.OnTranslationReceived += HandleTranslationResult;
            }
        }

        private void HandleTranslationResult(GestureReceiver.TranslationResult result)
        {
            if (translationHUD != null)
            {
                translationHUD.SetActive(!string.IsNullOrEmpty(result.text));
            }
        }

        private void HandleStreamResult(AvatarStreamer.SignVerseResult result)
        {
            if (mapper != null)
            {
                mapper.MapVector(result.intelligence_vector);
            }
            
            // Set Avatar "Expressiveness" based on confidence
            animator.SetFloat("SignConfidence", result.gesture_id > 0 ? 1.0f : 0.0f);
        }

        void OnDestroy()
        {
            if (_webcamTexture != null)
                _webcamTexture.Stop();
        }
    }
}
