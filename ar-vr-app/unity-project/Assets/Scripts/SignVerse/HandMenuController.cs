using UnityEngine;
using UnityEngine.XR.Hands;

namespace SignVerse
{
    /// <summary>
    /// HandMenuController — Manages the anchoring of UI to the user's hand.
    /// Uses OpenXR Hand joint data for high-precision placement.
    /// </summary>
    public class HandMenuController : MonoBehaviour
    {
        [Header("Components")]
        public XRHandSetup handSetup;
        public Handedness handToAttach = Handedness.Left;
        public GameObject menuContent;

        [Header("Orientation Thresholds")]
        public float faceUserThreshold = 0.5f;

        private Camera _mainCamera;

        void Start()
        {
            _mainCamera = Camera.main;
            if (menuContent != null) menuContent.SetActive(false);
        }

        void Update()
        {
            if (handSetup == null || menuContent == null) return;

            if (handSetup.IsHandTracked(handToAttach))
            {
                var hand = handSetup.GetHand(handToAttach);
                if (hand.rootPose.HasValue)
                {
                    // Check if palm is facing camera
                    Vector3 palmNormal = hand.rootPose.Value.up; 
                    Vector3 toCamera = (_mainCamera.transform.position - hand.rootPose.Value.position).normalized;
                    float dot = Vector3.Dot(palmNormal, toCamera);

                    bool shouldShow = dot > faceUserThreshold;
                    menuContent.SetActive(shouldShow);

                    if (shouldShow)
                    {
                        // Anchor menu to the palm/wrist area
                        transform.position = hand.rootPose.Value.position + Vector3.up * 0.1f;
                        transform.LookAt(_mainCamera.transform.position);
                        transform.Rotate(0, 180, 0); // Correct for face-user
                    }
                }
            }
            else
            {
                menuContent.SetActive(false);
            }
        }
    }
}
