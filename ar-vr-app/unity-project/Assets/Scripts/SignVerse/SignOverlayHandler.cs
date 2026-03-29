using UnityEngine;
using TMPro;

namespace SignVerse
{
    /// <summary>
    /// SignOverlayHandler — Manages real-time text overlays for sign translation.
    /// Can be driven by HybridCommunicationManager (legacy) or HybridRouter (new).
    /// </summary>
    public class SignOverlayHandler : MonoBehaviour
    {
        [Header("Components")]
        public HybridCommunicationManager hybridManager;  // Legacy path
        public TextMeshProUGUI overlayText;
        public GameObject bubbleObject;

        [Header("Settings")]
        public float displayDuration = 3.0f;
        public bool  followHead      = true;
        public float lerpSpeed       = 5.0f;

        private float     _hideTime = 0f;
        private Transform _mainCameraTransform;
        private string    _lastText = "";

        void Start()
        {
            _mainCameraTransform = Camera.main != null ? Camera.main.transform : transform;
            if (bubbleObject != null) bubbleObject.SetActive(false);
        }

        void Update()
        {
            // Legacy HybridCommunicationManager path
            if (hybridManager != null)
            {
                string text = hybridManager.finalTranslationText;
                if (!string.IsNullOrEmpty(text) && text != _lastText)
                    DisplayText(text);
            }

            // Billboard tracking + auto-hide
            if (bubbleObject != null && bubbleObject.activeSelf)
            {
                if (Time.time > _hideTime)
                {
                    bubbleObject.SetActive(false);
                    _lastText = "";
                }
                else if (followHead)
                {
                    Vector3 targetPos = _mainCameraTransform.position
                                      + _mainCameraTransform.forward * 1.2f
                                      + Vector3.up * 0.1f;
                    transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * lerpSpeed);
                    transform.rotation = Quaternion.LookRotation(transform.position - _mainCameraTransform.position);
                }
            }
        }

        /// <summary>
        /// Called by HybridRouter to push a translation result into the overlay.
        /// </summary>
        public void DisplayText(string text)
        {
            if (string.IsNullOrEmpty(text) || text == _lastText) return;
            _lastText = text;
            if (overlayText != null) overlayText.text = text;
            if (bubbleObject != null)
            {
                bubbleObject.SetActive(true);
                _hideTime = Time.time + displayDuration;
            }
        }
    }
}
