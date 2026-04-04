using UnityEngine;
using UnityEngine.Events;

public class GestureIntentEngine : MonoBehaviour
{
    public enum UIState { IDLE, MENU, SELECTION, EXECUTION }
    
    [Header("Current State")]
    public UIState currentState = UIState.IDLE;

    [Header("UI References")]
    public GameObject dashboardPanel;
    public GameObject translationBox;

    [Header("Events")]
    public UnityEvent onMenuOpen;
    public UnityEvent onSelect;
    public UnityEvent onExecute;

    /// <summary>
    /// Processes high-level gesture predictions from the AI Engine.
    /// </summary>
    public void ProcessGesture(string gestureLabel)
    {
        switch (gestureLabel.ToUpper())
        {
            case "OPEN_MENU":
                ToggleMenu(true);
                break;
            
            case "CLOSE_MENU":
                ToggleMenu(false);
                break;

            case "SELECT":
                HandleSelection();
                break;

            case "CONFIRM":
                ExecuteAction();
                break;

            default:
                Debug.Log($"Ignored Gesture: {gestureLabel}");
                break;
        }
    }

    private void ToggleMenu(bool open)
    {
        if (open)
        {
            currentState = UIState.MENU;
            if (dashboardPanel != null) dashboardPanel.SetActive(true);
            onMenuOpen?.Invoke();
        }
        else
        {
            currentState = UIState.IDLE;
            if (dashboardPanel != null) dashboardPanel.SetActive(false);
        }
    }

    private void HandleSelection()
    {
        if (currentState == UIState.MENU)
        {
            currentState = UIState.SELECTION;
            onSelect?.Invoke();
            Debug.Log("UI State: Selection Mode");
        }
    }

    private void ExecuteAction()
    {
        if (currentState == UIState.SELECTION)
        {
            currentState = UIState.EXECUTION;
            onExecute?.Invoke();
            Debug.Log("UI State: Action Executed");
            
            // Return to IDLE after a short delay or specific ending gesture
            currentState = UIState.IDLE;
        }
    }
}
