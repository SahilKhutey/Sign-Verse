using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

/// <summary>
/// SignVerseUIManager — Main HUD and Interaction Controller for AR/VR.
/// 
/// Provides:
///   - Language Selection (ASL, DGS, TSL, ISL, LSA)
///   - Connection Status Monitoring
///   - Text-to-Sign Input (Voice/Keyboard)
///   - Visual Style: Glassmorphism / Neon Accents
/// </summary>
public class SignVerseUIManager : MonoBehaviour
{
    [Header("References")]
    public GestureReceiver receiver;
    public AvatarController avatar;
    public SignRenderer renderer;
    public SpeechToSignProvider speechProvider;

    [Header("UI Components")]
    public CanvasGroup hudCanvasGroup;
    public TMP_Dropdown languageDropdown;
    public TextMeshProUGUI statusText;
    public TMP_InputField textInput;
    public Button sendButton;
    public Button micButton;
    public Image connectionIndicator;

    [Header("Design Settings")]
    public Color connectedColor = new Color(0.2f, 0.9f, 0.4f); // Neon Green
    public Color disconnectedColor = new Color(0.9f, 0.2f, 0.2f); // Alert Red
    public Color searchingColor = new Color(0.2f, 0.6f, 0.9f); // Tech Blue

    private void Start()
    {
        InitializeDropdown();
        UpdateStatus("Connecting...", searchingColor);
        
        if (sendButton != null)
            sendButton.onClick.AddListener(OnSendTextTriggered);

        if (micButton != null)
        {
            // Simple toggle: Click to start, click to stop is handled via state
            micButton.onClick.AddListener(OnMicClicked);
        }

        // Pulse the HUD on start for "Premium" feel
        StartCoroutine(FadeHUD(0f, 1f, 1.5f));
    }

    private bool isMicActive = false;
    private void OnMicClicked()
    {
        if (!isMicActive)
        {
            speechProvider?.StartListening();
            isMicActive = true;
            micButton.image.color = Color.red; // Record mode
        }
        else
        {
            speechProvider?.StopListening();
            isMicActive = false;
            micButton.image.color = Color.white;
        }
    }

    private IEnumerator FadeHUD(float from, float to, float duration)
    {
        if (hudCanvasGroup == null) yield break;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            hudCanvasGroup.alpha = Mathf.Lerp(from, to, elapsed / duration);
            yield return null;
        }
        hudCanvasGroup.alpha = to;
    }

    private void InitializeDropdown()
    {
        if (languageDropdown == null) return;

        languageDropdown.ClearOptions();
        languageDropdown.AddOptions(new List<string> { "ASL", "DGS", "TSL", "ISL", "LSA" });
        languageDropdown.onValueChanged.AddListener(OnLanguageChanged);
    }

    private void OnLanguageChanged(int index)
    {
        string lang = languageDropdown.options[index].text;
        Debug.Log($"[SignVerseUI] Language changed to {lang}");
        
        if (receiver != null)
            receiver.SetTargetLanguage(lang);
        
        if (avatar != null)
            avatar.ClearQueue();

        UpdateStatus($"Switched to {lang}", connectedColor);
    }

    public void OnSendTextTriggered()
    {
        if (textInput == null || string.IsNullOrEmpty(textInput.text)) return;

        string text = textInput.text;
        Debug.Log($"[SignVerseUI] Text-to-Sign: {text}");
        
        // Strategy: Get glosses from backend REST and play on avatar
        StartCoroutine(TranslateAndPlay(text));
        
        textInput.text = "";
    }

    private IEnumerator TranslateAndPlay(string text)
    {
        string lang = languageDropdown.options[languageDropdown.value].text;
        UpdateStatus("Translating...", searchingColor);

        // This uses the backend REST API to get the correct gloss sequence for the chosen language
        string url = "http://localhost:8000/translate/text-to-sign";
        string json = $"{{\"text\": \"{text}\", \"language\": \"{lang}\"}}";

        using (UnityEngine.Networking.UnityWebRequest request = new UnityEngine.Networking.UnityWebRequest(url, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UnityEngine.Networking.UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new UnityEngine.Networking.DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
            {
                // Parse tokens from: {"tokens": ["HELLO", "WORLD"]}
                string response = request.downloadHandler.text;
                TokensResponse res = JsonUtility.FromJson<TokensResponse>(response);
                
                if (res.tokens != null && res.tokens.Count > 0)
                {
                    avatar?.PlaySignSequence(res.tokens);
                    renderer?.RenderTokens(res.tokens);
                    UpdateStatus("Playing...", connectedColor);
                }
            }
            else
            {
                Debug.LogError($"[SignVerseUI] Translation failed: {request.error}");
                UpdateStatus("Error", disconnectedColor);
            }
        }
    }

    public void UpdateStatus(string message, Color color)
    {
        if (statusText != null)
        {
            statusText.text = message;
            statusText.color = color;
        }

        if (connectionIndicator != null)
            connectionIndicator.color = color;
    }

    [System.Serializable]
    private class TokensResponse
    {
        public List<string> tokens;
    }
}
