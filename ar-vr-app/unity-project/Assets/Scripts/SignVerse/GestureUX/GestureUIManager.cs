using UnityEngine;
using UnityEngine.XR.Hands;
using System.Collections;
using System.Collections.Generic;
using TMPro;

namespace SignVerse.GestureUX
{
    /// <summary>
    /// GestureUIManager — Context-aware floating panel system.
    ///
    /// Manages:
    ///   - Floating panel spawning/dismissal
    ///   - Menu state machine (Closed → Main → Sub)
    ///   - Gesture command routing to active UI context
    ///   - Smart panel placement in user's FOV
    /// </summary>
    public class GestureUIManager : MonoBehaviour
    {
        // ── Inspector ──────────────────────────────────────────────────────────
        [Header("Panels")]
        public GesturePanel mainMenuPanel;
        public GesturePanel translationPanel;
        public GesturePanel settingsPanel;

        [Header("Placement")]
        public Transform headTransform;   // Usually Camera.main.transform
        public float panelDistance   = 0.8f;
        public float panelHeightOffset = -0.05f;
        public float panelLerpSpeed  = 6f;

        [Header("Visual Feedback")]
        public GameObject gestureHintPrefab;  // Small floating label (e.g. "Pinch to Select")
        public AudioClip selectSound;
        public AudioClip dismissSound;

        // ── State Machine ──────────────────────────────────────────────────────
        private enum MenuState { Closed, Main, Translation, Settings }
        private MenuState _state = MenuState.Closed;

        private GestureInputSystem _input;
        private AudioSource        _audio;
        private GesturePanel       _activePanel;

        // ── Lifecycle ──────────────────────────────────────────────────────────
        void Awake()
        {
            _input = GetComponent<GestureInputSystem>();
            _audio = GetComponent<AudioSource>();

            if (_input != null)
                _input.OnGestureCommand += HandleCommand;

            if (headTransform == null && Camera.main != null)
                headTransform = Camera.main.transform;

            // Start all panels hidden
            SetPanelActive(mainMenuPanel,    false);
            SetPanelActive(translationPanel, false);
            SetPanelActive(settingsPanel,    false);
        }

        void Update()
        {
            // Lazily follow head for active panel (no teleport, smooth lerp)
            if (_activePanel != null && _activePanel.gameObject.activeSelf)
                UpdatePanelPosition(_activePanel);
        }

        // ── Command Handler ────────────────────────────────────────────────────
        private void HandleCommand(GestureCommand cmd)
        {
            switch (_state)
            {
                case MenuState.Closed:
                    if (cmd == GestureCommand.OpenMenu)   OpenMain();
                    break;

                case MenuState.Main:
                    switch (cmd)
                    {
                        case GestureCommand.CloseMenu:    CloseAll();    break;
                        case GestureCommand.Select:       OpenTranslation(); break;
                        case GestureCommand.ScrollDown:   OpenSettings();    break;
                        case GestureCommand.Back:
                        case GestureCommand.DismissPanel: CloseAll();    break;
                    }
                    break;

                case MenuState.Translation:
                    switch (cmd)
                    {
                        case GestureCommand.Back:
                        case GestureCommand.DismissPanel: BackToMain();  break;
                        case GestureCommand.ScrollUp:     mainMenuPanel?.ScrollUp();   break;
                        case GestureCommand.ScrollDown:   mainMenuPanel?.ScrollDown(); break;
                        case GestureCommand.ConfirmAction: ConfirmCurrentItem(); break;
                    }
                    break;

                case MenuState.Settings:
                    switch (cmd)
                    {
                        case GestureCommand.Back:
                        case GestureCommand.DismissPanel: BackToMain();  break;
                        case GestureCommand.ConfirmAction: ConfirmCurrentItem(); break;
                    }
                    break;
            }
        }

        // ── State Transitions ──────────────────────────────────────────────────

        private void OpenMain()
        {
            _state = MenuState.Main;
            SpawnPanel(mainMenuPanel);
            PlaySound(selectSound);
        }

        private void OpenTranslation()
        {
            _state = MenuState.Translation;
            HidePanel(mainMenuPanel);
            SpawnPanel(translationPanel);
            PlaySound(selectSound);
        }

        private void OpenSettings()
        {
            _state = MenuState.Settings;
            HidePanel(mainMenuPanel);
            SpawnPanel(settingsPanel);
            PlaySound(selectSound);
        }

        private void BackToMain()
        {
            HidePanel(_activePanel);
            _state = MenuState.Main;
            SpawnPanel(mainMenuPanel);
            PlaySound(dismissSound);
        }

        private void CloseAll()
        {
            HidePanel(_activePanel);
            _state = MenuState.Closed;
            _activePanel = null;
            PlaySound(dismissSound);
        }

        private void ConfirmCurrentItem()
        {
            _activePanel?.ConfirmSelection();
            PlaySound(selectSound);
        }

        // ── Panel Management ───────────────────────────────────────────────────

        private void SpawnPanel(GesturePanel panel)
        {
            if (panel == null) return;
            PlacePanelInFOV(panel);
            SetPanelActive(panel, true);
            _activePanel = panel;
        }

        private void HidePanel(GesturePanel panel)
        {
            if (panel != null) SetPanelActive(panel, false);
        }

        private void PlacePanelInFOV(GesturePanel panel)
        {
            if (headTransform == null || panel == null) return;
            Vector3 forward = new Vector3(headTransform.forward.x, 0, headTransform.forward.z).normalized;
            Vector3 target  = headTransform.position + forward * panelDistance + Vector3.up * panelHeightOffset;
            panel.transform.position = target;
            panel.transform.LookAt(headTransform.position + headTransform.forward * 2f);
        }

        private void UpdatePanelPosition(GesturePanel panel)
        {
            if (headTransform == null || panel == null) return;
            Vector3 forward = new Vector3(headTransform.forward.x, 0, headTransform.forward.z).normalized;
            Vector3 target  = headTransform.position + forward * panelDistance + Vector3.up * panelHeightOffset;

            panel.transform.position = Vector3.Lerp(panel.transform.position, target, Time.deltaTime * panelLerpSpeed);
            panel.transform.rotation = Quaternion.Lerp(
                panel.transform.rotation,
                Quaternion.LookRotation(panel.transform.position - headTransform.position),
                Time.deltaTime * panelLerpSpeed
            );
        }

        private void SetPanelActive(GesturePanel panel, bool active)
        {
            if (panel != null) panel.gameObject.SetActive(active);
        }

        private void PlaySound(AudioClip clip)
        {
            if (_audio != null && clip != null) _audio.PlayOneShot(clip, 0.5f);
        }
    }
}
