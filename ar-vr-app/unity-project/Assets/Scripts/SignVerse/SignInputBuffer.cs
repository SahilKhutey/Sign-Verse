using UnityEngine;
using UnityEngine.UI;

namespace SignVerse
{
    /// <summary>
    /// SignInputBuffer — Visualizes the "Confidence" of the current sign recognition.
    /// Provides a real-time progress bar floating near the user's hand.
    /// </summary>
    public class SignInputBuffer : MonoBehaviour
    {
        [Header("Components")]
        public GestureReceiver receiver;
        public Image confidenceBar;
        public CanvasGroup canvasGroup;

        [Header("Settings")]
        public float fadeSpeed = 5.0f;
        public float minShowConfidence = 0.1f;

        private float _currentConfidence = 0f;
        private bool _isVisible = false;

        void Start()
        {
            if (receiver != null)
            {
                receiver.OnTranslationReceived += HandleTranslation;
            }
            if (canvasGroup != null) canvasGroup.alpha = 0;
        }

        private void HandleTranslation(GestureReceiver.TranslationResult result)
        {
            _currentConfidence = result.confidence;
            _isVisible = _currentConfidence > minShowConfidence;
        }

        void Update()
        {
            if (confidenceBar != null)
            {
                confidenceBar.fillAmount = Mathf.Lerp(confidenceBar.fillAmount, _currentConfidence, Time.deltaTime * 10f);
            }

            if (canvasGroup != null)
            {
                float targetAlpha = _isVisible ? 1.0f : 0.0f;
                canvasGroup.alpha = Mathf.MoveTowards(canvasGroup.alpha, targetAlpha, Time.deltaTime * fadeSpeed);
            }
        }
    }
}
