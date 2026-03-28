using System.Collections.Generic;
using UnityEngine;
using TMPro;

/// <summary>
/// SignRenderer — Renders sign language text overlay in AR/VR.
///
/// Displays:
///   - Sign gloss tokens as floating text
///   - Translated English sentence
///   - Visual confidence indicator
/// </summary>
public class SignRenderer : MonoBehaviour
{
    [Header("UI References")]
    public TextMeshProUGUI glossText;
    public TextMeshProUGUI translationText;
    public GameObject tokenPrefab;
    public Transform tokenContainer;

    [Header("Settings")]
    public float tokenDisplayTime = 3.0f;
    public Color activeTokenColor = new Color(0.2f, 0.9f, 0.4f);
    public Color inactiveTokenColor = new Color(1f, 1f, 1f, 0.5f);

    private List<GameObject> tokenObjects = new List<GameObject>();
    private float clearTimer;

    private void Update()
    {
        if (tokenObjects.Count > 0)
        {
            clearTimer -= Time.deltaTime;
            if (clearTimer <= 0f)
                ClearTokens();
        }
    }

    /// <summary>
    /// Render a list of sign gloss tokens as floating UI labels.
    /// </summary>
    public void RenderTokens(List<string> tokens)
    {
        ClearTokens();

        string glossLine = string.Join(" ", tokens);

        if (glossText != null)
            glossText.text = glossLine;

        foreach (string token in tokens)
        {
            if (tokenPrefab != null && tokenContainer != null)
            {
                GameObject obj = Instantiate(tokenPrefab, tokenContainer);
                TextMeshProUGUI label = obj.GetComponentInChildren<TextMeshProUGUI>();
                if (label != null) {
                    label.text = token;
                    label.color = activeTokenColor;
                }
                tokenObjects.Add(obj);
            }
        }

        clearTimer = tokenDisplayTime;
    }

    /// <summary>
    /// Update the translated English sentence display.
    /// </summary>
    public void SetTranslation(string sentence)
    {
        if (translationText != null)
            translationText.text = sentence;
    }

    /// <summary>
    /// Highlight the currently active token.
    /// </summary>
    public void HighlightToken(int index)
    {
        for (int i = 0; i < tokenObjects.Count; i++)
        {
            TextMeshProUGUI label = tokenObjects[i].GetComponentInChildren<TextMeshProUGUI>();
            if (label != null)
                label.color = (i == index) ? activeTokenColor : inactiveTokenColor;
        }
    }

    private void ClearTokens()
    {
        foreach (GameObject obj in tokenObjects)
            Destroy(obj);
        tokenObjects.Clear();

        if (glossText != null)
            glossText.text = "";
    }
}
