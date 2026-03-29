using System;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace SignVerse
{
    /// <summary>
    /// GestureUIManager — Orchestrates high-level gesture triggers.
    /// Maps hand positions and signs (from backend) to UI actions.
    /// </summary>
    public class GestureUIManager : MonoBehaviour
    {
        [Header("Components")]
        public XRHandSetup handSetup;
        public GestureReceiver receiver;
        public GameObject mainMenuPanel;

        [Header("Thresholds")]
        [Range(0, 180)] public float palmFacingAngle = 30f;
        public float pinchSelectionThreshold = 0.5f;

        private bool _menuActive = false;
        private Camera _mainCamera;

        void Start()
        {
            _mainCamera = Camera.main;
            if (receiver != null)
            {
                receiver.OnTranslationReceived += HandleIntent;
            }
        }

        void Update()
        {
            CheckPalmUpMenu();
        }

        private void CheckPalmUpMenu()
        {
            if (handSetup == null) return;

            // Check Left Hand Palm Direction
            if (handSetup.IsHandTracked(Handedness.Left))
            {
                var hand = handSetup.GetHand(Handedness.Left);
                if (hand.rootPose.HasValue)
                {
                    Vector3 palmNormal = hand.rootPose.Value.up; // Standard Unity hand up is palm-out normally
                    // For many XR plugins, Pose.up is palm normal.
                    
                    Vector3 toCamera = (_mainCamera.transform.position - hand.rootPose.Value.position).normalized;
                    float angle = Vector3.Angle(palmNormal, toCamera);

                    if (angle < palmFacingAngle)
                    {
                        if (!_menuActive) ToggleMenu(true);
                    }
                    else
                    {
                        if (_menuActive) ToggleMenu(false);
                    }
                }
            }
        }

        private void HandleIntent(GestureReceiver.TranslationResult result)
        {
            // If the AI detects a specific sign like "MENU", override.
            if (result.intent == "UI_ACTIVATE")
            {
                ToggleMenu(true);
            }
            else if (result.intent == "UI_DEACTIVATE")
            {
                ToggleMenu(false);
            }
        }

        private void ToggleMenu(bool active)
        {
            _menuActive = active;
            if (mainMenuPanel != null)
            {
                mainMenuPanel.SetActive(active);
                // Position it relative to the hand
                if (active)
                {
                    var hand = handSetup.GetHand(Handedness.Left);
                    mainMenuPanel.transform.position = hand.rootPose.Value.position + Vector3.up * 0.2f;
                    mainMenuPanel.transform.LookAt(_mainCamera.transform);
                }
            }
        }
    }
}
